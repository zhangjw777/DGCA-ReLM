"""
DGCA-ReLM Model: Detector-Guided Confusion-Aware ReLM
实现检测分支、候选注意力头、门控融合等核心组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM
from typing import Optional, Tuple, Dict


class DetectorHead(nn.Module):
    """
    检测分支：预测每个位置是否为错误
    输入源句hidden states，输出每个位置的错误概率
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) 源句的hidden states
            
        Returns:
            detection_probs: (batch, seq_len) 每个位置的错误概率
        """
        # (batch, seq_len, hidden_size)
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # (batch, seq_len, 1) -> (batch, seq_len)
        logits = self.classifier(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return probs


class CandidateHead(nn.Module):
    """
    候选注意力头：在候选集范围内预测目标字符
    计算hidden state与候选embeddings的相似度，输出候选集上的分布
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) mask位置的hidden states
            candidate_embeddings: (batch, seq_len, cand_size, hidden_size) 候选集的embeddings
            candidate_mask: (batch, seq_len, cand_size) 候选集的mask（pad位置为False）
            
        Returns:
            candidate_probs: (batch, seq_len, cand_size) 候选集上的概率分布
        """
        # Transform hidden states
        # (batch, seq_len, hidden_size)
        h = self.transform(hidden_states)
        h = self.activation(h)
        h = self.dropout(h)
        
        # 计算相似度得分
        # h: (batch, seq_len, hidden_size, 1)
        # candidate_embeddings: (batch, seq_len, cand_size, hidden_size)
        # scores: (batch, seq_len, cand_size)
        h_expanded = h.unsqueeze(2)  # (batch, seq_len, 1, hidden_size)
        scores = torch.matmul(h_expanded, candidate_embeddings.transpose(-2, -1)).squeeze(2)
        
        # 应用mask（将pad位置设为-inf）
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, float('-inf'))
        
        # Softmax得到概率分布
        probs = F.softmax(scores, dim=-1)
        
        return probs


class GatedFusion(nn.Module):
    """
    门控融合模块：动态融合全词表分布和候选集分布
    使用检测信号和hidden state计算门控权重α
    
    优化：
    1. 避免每次创建全词表大小的零张量
    2. 直接在候选集空间计算，最后再映射
    """
    
    def __init__(self, hidden_size: int, init_bias: float = 0.0):
        super().__init__()
        # 输入: [hidden_state; detection_prob]
        self.gate_linear = nn.Linear(hidden_size + 1, 1)
        
        # 初始化bias控制初始倾向
        nn.init.constant_(self.gate_linear.bias, init_bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        detection_probs: torch.Tensor,
        vocab_logits: torch.Tensor,
        candidate_probs: torch.Tensor,
        candidate_ids: torch.Tensor,
        vocab_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            detection_probs: (batch, seq_len) 检测概率
            vocab_logits: (batch, seq_len, vocab_size) 全词表logits
            candidate_probs: (batch, seq_len, cand_size) 候选集概率
            candidate_ids: (batch, seq_len, cand_size) 候选集token ids
            vocab_size: 词表大小
            
        Returns:
            fused_logits: (batch, seq_len, vocab_size) 融合后的logits
            gate_weights: (batch, seq_len) 门控权重α
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算门控权重 α
        # (batch, seq_len, hidden_size+1)
        gate_input = torch.cat([hidden_states, detection_probs.unsqueeze(-1)], dim=-1)
        # (batch, seq_len)
        gate_logits = self.gate_linear(gate_input).squeeze(-1)
        gate_weights = torch.sigmoid(gate_logits)
        
        # 优化：直接在vocab_logits上修改，避免创建新的大张量
        # 计算vocab概率
        vocab_probs = F.softmax(vocab_logits, dim=-1)
        
        # 计算融合后的概率
        # 对于候选集中的token：p_fused = (1-α) * p_vocab + α * p_cand
        # 对于非候选集token：p_fused = (1-α) * p_vocab
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # 先将vocab_probs乘以(1-α)
        fused_probs = (1 - gate_weights_expanded) * vocab_probs
        
        # 然后使用scatter_add_将候选概率加到对应位置
        # 这比创建全词表大小的零张量再scatter更高效
        weighted_candidate_probs = gate_weights_expanded * candidate_probs
        fused_probs.scatter_add_(
            dim=-1,
            index=candidate_ids,
            src=weighted_candidate_probs
        )
        
        # 转回logits（避免log(0)）
        fused_logits = torch.log(fused_probs + 1e-10)
        
        return fused_logits, gate_weights


class DGCAReLMWrapper(nn.Module):
    """
    DGCA-ReLM完整模型包装器
    在ReLM（BertForMaskedLM + P-tuning）基础上添加DGCA模块
    """
    
    def __init__(
        self,
        bert_model: BertForMaskedLM,
        confusion_set,
        config,
        prompt_length: int = 3
    ):
        """
        Args:
            bert_model: BertForMaskedLM基座模型
            confusion_set: ConfusionSet实例
            config: DGCAConfig配置
            prompt_length: prompt长度（沿用ReLM）
        """
        super().__init__()
        
        self.config = config
        self.bert_config = bert_model.config
        self.confusion_set = confusion_set
        self.prompt_length = prompt_length
        
        # 基座模型
        self.model = bert_model
        self.model_type = self.bert_config.model_type.split("-")[0]
        self.word_embeddings = getattr(self.model, self.model_type).embeddings.word_embeddings
        self.hidden_size = self.bert_config.hidden_size
        self.vocab_size = self.bert_config.vocab_size
        
        # P-tuning（沿用ReLM）
        self.prompt_embeddings = nn.Embedding(2 * prompt_length, self.hidden_size)
        self.prompt_lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.prompt_linear = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # DGCA模块
        if config.use_detector:
            self.detector_head = DetectorHead(
                self.hidden_size,
                dropout=config.detector_dropout
            )
        else:
            self.detector_head = None
        
        if config.use_candidate_head:
            self.candidate_head = CandidateHead(
                self.hidden_size,
                dropout=config.candidate_dropout
            )
        else:
            self.candidate_head = None
        
        if config.use_gated_fusion:
            self.gated_fusion = GatedFusion(
                self.hidden_size,
                init_bias=config.gate_init_bias
            )
        else:
            self.gated_fusion = None
    
    def _apply_prompt(
        self,
        inputs_embeds: torch.Tensor,
        prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        应用P-tuning prompt（优化版：使用向量化操作替代Python循环）
        """
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        # 生成prompt embeddings
        replace_embeds = self.prompt_embeddings(
            torch.arange(2 * self.prompt_length, device=device)
        )
        replace_embeds = replace_embeds.unsqueeze(0)  # (1, 2*prompt_length, hidden)
        replace_embeds = self.prompt_lstm(replace_embeds)[0]  # (1, 2*prompt_length, 2*hidden)
        replace_embeds = self.prompt_linear(replace_embeds).squeeze(0)  # (2*prompt_length, hidden)
        
        # 优化：使用向量化操作替代双重for循环
        # prompt_mask: (batch, seq_len), 值为1的位置是需要替换的prompt位置
        # 假设每个样本的prompt位置是相同的（在固定位置）
        
        # 找到第一个样本的prompt位置作为参考
        prompt_positions = (prompt_mask[0] == 1).nonzero(as_tuple=True)[0]  # (2*prompt_length,)
        
        # 使用高级索引一次性替换所有样本的prompt位置
        # replace_embeds: (2*prompt_length, hidden)
        # 扩展到 (batch, 2*prompt_length, hidden)
        replace_embeds_expanded = replace_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 使用index_copy_进行批量替换
        # inputs_embeds[:, prompt_positions, :] = replace_embeds_expanded
        inputs_embeds.index_copy_(1, prompt_positions, replace_embeds_expanded)
        
        return inputs_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        labels: torch.Tensor,
        candidate_ids: Optional[torch.Tensor] = None,
        error_labels: Optional[torch.Tensor] = None,
        apply_prompt: bool = True,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            prompt_mask: (batch, seq_len) prompt位置标记
            labels: (batch, seq_len) 目标token ids（-100为ignore）
            candidate_ids: (batch, seq_len, cand_size) 候选集token ids
            error_labels: (batch, seq_len) 错误位置标签（1=错误，0=正确，-100=ignore）
            apply_prompt: 是否应用prompt
            return_dict: 是否返回字典
            
        Returns:
            字典包含: loss, logits, detection_probs, gate_weights等
        """
        batch_size, seq_len = input_ids.shape
        
        # 获取inputs embeds并应用prompt
        inputs_embeds = self.word_embeddings(input_ids)
        if apply_prompt:
            inputs_embeds = self._apply_prompt(inputs_embeds, prompt_mask)
        
        # BERT前向传播，获取hidden states
        outputs = self.model.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # ========== 检测分支 ==========
        detection_probs = None
        detection_loss = None
        if self.detector_head is not None:
            detection_probs = self.detector_head(sequence_output)
            
            # 计算检测损失
            if error_labels is not None and self.training:
                valid_mask = (error_labels != -100)
                if valid_mask.any():
                    pos_weight = torch.tensor(
                        [self.config.detector_pos_weight],
                        device=detection_probs.device
                    )
                    detection_loss_fn = nn.BCEWithLogitsLoss(
                        pos_weight=pos_weight,
                        reduction='none'
                    )
                    # 需要logits，重新计算
                    det_logits = torch.log(detection_probs / (1 - detection_probs + 1e-10))
                    det_loss = detection_loss_fn(det_logits, error_labels.float())
                    detection_loss = (det_loss * valid_mask.float()).sum() / valid_mask.float().sum()
        
        # ========== MLM logits（全词表） ==========
        vocab_logits = self.model.cls(sequence_output)  # (batch, seq_len, vocab_size)
        
        # ========== 候选头 ==========
        candidate_probs = None
        gate_weights = None
        final_logits = vocab_logits
        
        if self.candidate_head is not None and candidate_ids is not None:
            # 获取候选集embeddings
            candidate_embeddings = self.word_embeddings(candidate_ids)
            # (batch, seq_len, cand_size, hidden)
            
            # 候选集mask（pad位置）
            candidate_mask = (candidate_ids != self.confusion_set.pad_token_id)
            
            # 计算候选概率
            candidate_probs = self.candidate_head(
                sequence_output,
                candidate_embeddings,
                candidate_mask
            )  # (batch, seq_len, cand_size)
            
            # ========== 门控融合 ==========
            if self.gated_fusion is not None:
                if detection_probs is None:
                    # 如果没有检测头，用全0（即不使用检测信号）
                    detection_probs = torch.zeros(
                        batch_size, seq_len,
                        device=sequence_output.device
                    )
                
                final_logits, gate_weights = self.gated_fusion(
                    sequence_output,
                    detection_probs,
                    vocab_logits,
                    candidate_probs,
                    candidate_ids,
                    self.vocab_size
                )
            else:
                # 不使用门控，直接用候选头
                # 将候选概率映射到全词表
                candidate_probs_ext = torch.zeros(
                    batch_size, seq_len, self.vocab_size,
                    dtype=candidate_probs.dtype,
                    device=candidate_probs.device
                )
                candidate_probs_ext.scatter_(
                    dim=-1,
                    index=candidate_ids,
                    src=candidate_probs
                )
                final_logits = torch.log(candidate_probs_ext + 1e-10)
        
        # ========== 计算纠错损失 ==========
        correction_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            correction_loss_all = loss_fct(
                final_logits.view(-1, self.vocab_size),
                labels.view(-1)
            )  # (batch * seq_len,)
            
            # 错误位置加权
            if self.config.error_position_weight > 0 and error_labels is not None:
                weight = torch.ones_like(labels, dtype=torch.float)
                weight[error_labels == 1] = 1 + self.config.error_position_weight
                weight = weight.view(-1)
                correction_loss_all = correction_loss_all * weight
            
            valid_mask = (labels.view(-1) != -100)
            if valid_mask.any():
                correction_loss = correction_loss_all[valid_mask].mean()
        
        # ========== 候选排序损失（可选） ==========
        rank_loss = None
        if (self.config.rank_loss_weight > 0 and 
            candidate_probs is not None and 
            labels is not None and
            self.training):
            # 找到target在候选集中的位置
            # (batch, seq_len, cand_size)
            target_mask = (candidate_ids == labels.unsqueeze(-1))
            
            if target_mask.any():
                # 获取正确候选的分数
                target_scores = (candidate_probs * target_mask.float()).sum(dim=-1)  # (batch, seq_len)
                
                # 获取其他候选的分数
                other_scores = candidate_probs.masked_fill(target_mask, 0.0)  # (batch, seq_len, cand_size)
                
                # Margin ranking loss
                valid_positions = (labels != -100) & target_mask.any(dim=-1)
                if valid_positions.any():
                    target_scores_exp = target_scores.unsqueeze(-1)  # (batch, seq_len, 1)
                    margin = self.config.rank_margin
                    rank_loss_all = F.relu(margin - target_scores_exp + other_scores)
                    rank_loss_all = rank_loss_all.sum(dim=-1)  # (batch, seq_len)
                    rank_loss = rank_loss_all[valid_positions].mean()
        
        # ========== 总损失 ==========
        loss = None
        if correction_loss is not None:
            loss = correction_loss
            
            if detection_loss is not None:
                loss = loss + self.config.detector_loss_weight * detection_loss
            
            if rank_loss is not None:
                loss = loss + self.config.rank_loss_weight * rank_loss
        
        # 返回结果
        if not return_dict:
            return (loss, final_logits, detection_probs, gate_weights)
        
        return {
            'loss': loss,
            'logits': final_logits,
            'detection_probs': detection_probs,
            'gate_weights': gate_weights,
            'correction_loss': correction_loss,
            'detection_loss': detection_loss,
            'rank_loss': rank_loss,
        }
