"""
混淆集处理模块
加载混淆集json并为每个字符构建候选集
支持多文件合并：pinyin_sim.json, stroke.json, word_freq.txt
"""

import json
import os
from typing import Dict, List, Optional, Set
import torch


class ConfusionSet:
    """混淆集管理类（支持多文件合并）"""
    
    def __init__(
        self,
        confusion_dir: str = None,
        confusion_file: str = None,
        tokenizer=None,
        candidate_size: int = 16,
        include_original: bool = True,
    ):
        """
        Args:
            confusion_dir: 混淆集目录（包含pinyin_sim.json, stroke.json, word_freq.txt）
            confusion_file: 单个混淆集json文件路径（兼容旧接口）
            tokenizer: transformers tokenizer
            candidate_size: 候选集大小K
            include_original: 是否包含原字符
        """
        self.tokenizer = tokenizer
        self.candidate_size = candidate_size
        self.include_original = include_original
        self.vocab_size = tokenizer.vocab_size
        
        # 加载混淆集（合并多个来源）
        self.confusion_dict: Dict[str, List[str]] = {}
        if confusion_dir:
            self._load_from_dir(confusion_dir)
        elif confusion_file:
            self._load_confusion_file(confusion_file)
        
        # 构建token id到候选token ids的映射
        self.token_to_candidates: Dict[int, List[int]] = {}
        self._build_candidate_mapping()
        
        # 特殊token的id
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
    
    def _load_from_dir(self, confusion_dir: str):
        """从目录加载多个混淆集文件并合并"""
        pinyin_path = os.path.join(confusion_dir, "pinyin_sim.json")
        stroke_path = os.path.join(confusion_dir, "stroke.json")
        freq_path = os.path.join(confusion_dir, "word_freq.txt")
        
        # 加载拼音相近混淆集
        if os.path.exists(pinyin_path):
            with open(pinyin_path, 'r', encoding='utf-8') as f:
                pinyin_dict = json.load(f)
            self._merge_dict(pinyin_dict)
            print(f"加载拼音混淆集: {len(pinyin_dict)} 条")
        
        # 加载字形相似混淆集
        if os.path.exists(stroke_path):
            with open(stroke_path, 'r', encoding='utf-8') as f:
                stroke_dict = json.load(f)
            self._merge_dict(stroke_dict)
            print(f"加载字形混淆集: {len(stroke_dict)} 条")
        
        # 加载高频词（所有字符都可替换为高频词）
        if os.path.exists(freq_path):
            with open(freq_path, 'r', encoding='utf-8') as f:
                freq_words = [line.strip() for line in f if line.strip()]
            # 高频词作为通用候选，但不加入confusion_dict
            # 保持候选集精准度
            print(f"加载高频词: {len(freq_words)} 个")
        
        # 兼容：如果目录下有confusion_dict.json，也加载
        legacy_path = os.path.join(confusion_dir, "confusion_dict.json")
        if os.path.exists(legacy_path):
            with open(legacy_path, 'r', encoding='utf-8') as f:
                legacy_dict = json.load(f)
            self._merge_dict(legacy_dict)
            print(f"加载legacy混淆集: {len(legacy_dict)} 条")
        
        print(f"合并后混淆集总计: {len(self.confusion_dict)} 条")
    
    def _merge_dict(self, new_dict: Dict[str, List[str]]):
        """合并混淆集（去重）"""
        for char, confusions in new_dict.items():
            if char not in self.confusion_dict:
                self.confusion_dict[char] = []
            existing = set(self.confusion_dict[char])
            for c in confusions:
                if c not in existing:
                    self.confusion_dict[char].append(c)
                    existing.add(c)
        
    def _load_confusion_file(self, confusion_file: str):
        """加载混淆集json文件"""
        if not os.path.exists(confusion_file):
            print(f"Warning: Confusion file {confusion_file} not found. Using empty confusion set.")
            return
        
        with open(confusion_file, 'r', encoding='utf-8') as f:
            self.confusion_dict = json.load(f)
        
        print(f"Loaded confusion set with {len(self.confusion_dict)} entries.")
    
    def _build_candidate_mapping(self):
        """构建token id到候选集的映射"""
        unk_id = self.tokenizer.unk_token_id
        skipped_unk = 0
        
        for char, confusions in self.confusion_dict.items():
            # 获取字符的token id
            char_ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(char_ids) != 1:
                continue  # 跳过多字符token
            char_id = char_ids[0]
            
            # 跳过 UNK token（避免多个OOV字覆盖同一个映射）
            if char_id == unk_id:
                skipped_unk += 1
                continue
            
            # 构建候选列表
            candidates = []
            
            # 如果包含原字符，先加入原字符
            if self.include_original:
                candidates.append(char_id)
            
            # 加入混淆字符
            for conf_char in confusions:
                conf_ids = self.tokenizer.encode(conf_char, add_special_tokens=False)
                if len(conf_ids) == 1:
                    conf_id = conf_ids[0]
                    # 跳过 UNK 混淆字符
                    if conf_id == unk_id:
                        continue
                    if conf_id not in candidates:
                        candidates.append(conf_id)
                
                # 达到候选集大小限制
                if len(candidates) >= self.candidate_size:
                    break
            
            self.token_to_candidates[char_id] = candidates
        
        if skipped_unk > 0:
            print(f"跳过了 {skipped_unk} 个 UNK 字符的映射")
    
    def get_candidates(self, token_id: int) -> List[int]:
        """
        获取指定token的候选集
        
        Args:
            token_id: token的id
            
        Returns:
            候选token id列表，如果没有混淆集则返回只包含原token的列表
        """
        if token_id in self.token_to_candidates:
            return self.token_to_candidates[token_id]
        
        # 对于没有混淆集的token，返回只包含自身的列表
        if self.include_original:
            return [token_id]
        return []
    
    def get_candidates_batch(
        self,
        token_ids: torch.Tensor,
        max_candidates: Optional[int] = None
    ) -> torch.Tensor:
        """
        批量获取候选集
        
        Args:
            token_ids: (batch_size, seq_len) token id tensor
            max_candidates: 最大候选数，默认使用self.candidate_size
            
        Returns:
            (batch_size, seq_len, candidate_size) 候选集tensor，不足的位置用pad_token_id填充
        """
        if max_candidates is None:
            max_candidates = self.candidate_size
        
        batch_size, seq_len = token_ids.shape
        candidates = torch.full(
            (batch_size, seq_len, max_candidates),
            self.pad_token_id,
            dtype=torch.long,
            device=token_ids.device
        )
        
        token_ids_list = token_ids.cpu().tolist()
        
        for b in range(batch_size):
            for s in range(seq_len):
                tid = token_ids_list[b][s]
                cands = self.get_candidates(tid)
                for c, cand_id in enumerate(cands[:max_candidates]):
                    candidates[b, s, c] = cand_id
        
        return candidates
    
    def get_candidate_mask(
        self,
        token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        max_candidates: Optional[int] = None
    ) -> torch.Tensor:
        """
        获取目标token在候选集中的位置mask
        
        Args:
            token_ids: (batch_size, seq_len) 源token ids
            target_ids: (batch_size, seq_len) 目标token ids
            max_candidates: 最大候选数
            
        Returns:
            (batch_size, seq_len, candidate_size) bool tensor，目标token位置为True
        """
        if max_candidates is None:
            max_candidates = self.candidate_size
        
        candidates = self.get_candidates_batch(token_ids, max_candidates)
        target_expanded = target_ids.unsqueeze(-1).expand_as(candidates)
        
        return candidates == target_expanded
    
    def get_candidate_embeddings(
        self,
        embedding_layer: torch.nn.Embedding,
        candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        获取候选集的embedding
        
        Args:
            embedding_layer: word embedding层
            candidate_ids: (batch_size, seq_len, candidate_size) 候选集tensor
            
        Returns:
            (batch_size, seq_len, candidate_size, hidden_size) embedding tensor
        """
        return embedding_layer(candidate_ids)
    
    def is_in_candidates(self, token_id: int, candidate_id: int) -> bool:
        """检查candidate_id是否在token_id的候选集中"""
        candidates = self.get_candidates(token_id)
        return candidate_id in candidates
