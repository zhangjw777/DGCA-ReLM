"""
DGCA-ReLM Configuration Module
支持通过yaml配置文件控制消融实验开关
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DGCAConfig:
    """DGCA-ReLM 配置类"""
    
    # ============ 模型结构开关（消融实验） ============
    # 是否使用检测分支
    use_detector: bool = True
    # 是否使用候选注意力头
    use_candidate_head: bool = True
    # 是否使用门控融合（如果False，则直接使用候选头或vocab头）
    use_gated_fusion: bool = True
    
    # ============ 候选集相关 ============
    # 候选集大小K
    candidate_size: int = 16
    # 混淆集文件路径（单文件模式，兼容旧配置）
    confusion_file: str = "confusion/confusion_dict.json"
    # 混淆集目录（多文件模式，包含pinyin_sim.json, stroke.json, word_freq.txt）
    confusion_dir: str = "confusion"
    # 是否在候选集中包含原字符（允许"不改"）
    include_original_char: bool = True
    # 是否使用语言模型top-k候选补充混淆集
    use_lm_candidates: bool = False
    lm_candidate_topk: int = 5
    
    # ============ 检测分支相关 ============
    # 检测分支的dropout
    detector_dropout: float = 0.1
    # 检测损失的权重 λ_det
    detector_loss_weight: float = 1.0
    # 检测分支正样本权重（β，用于加权BCE，鼓励少漏检）
    detector_pos_weight: float = 3.0
    
    # ============ 候选头相关 ============
    # 候选头的dropout
    candidate_dropout: float = 0.1
    
    # ============ 门控融合相关 ============
    # 门控初始偏置（负值倾向于vocab head，正值倾向于candidate head）
    gate_init_bias: float = 0.0
    
    # ============ 损失函数相关 ============
    # 辅助MLM损失权重 λ_aux
    aux_mlm_loss_weight: float = 1.0
    # 候选排序损失权重 λ_rank
    rank_loss_weight: float = 0.5
    # 候选排序损失的margin
    rank_margin: float = 0.5
    # 错误位加权 γ（错误位权重 = 1 + γ）
    error_position_weight: float = 2.0
    
    # ============ 推理相关 ============
    # 检测阈值（d_i > τ 才考虑修改）
    detect_threshold: float = 0.5
    # 门控阈值（α_i > τ 才使用候选头）
    gate_threshold: float = 0.3
    # 是否使用两次迭代修正
    use_iterative_refinement: bool = False
    
    # ============ 数据增强相关 ============
    # 是否使用上下文错误增强
    use_contextual_error_aug: bool = False
    # 是否使用多错共存增强
    use_multi_typo_aug: bool = False
    # 多错数量范围
    multi_typo_min: int = 2
    multi_typo_max: int = 3
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DGCAConfig":
        """从yaml文件加载配置"""
        if not os.path.exists(yaml_path):
            print(f"Warning: Config file {yaml_path} not found, using default config.")
            return cls()
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            return cls()
        
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def to_yaml(self, yaml_path: str):
        """保存配置到yaml文件"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f, allow_unicode=True, default_flow_style=False)
    
    def __str__(self):
        return yaml.dump(self.__dict__, allow_unicode=True, default_flow_style=False)
    
    def get_ablation_name(self) -> str:
        """根据配置生成消融实验名称"""
        parts = []
        if self.use_detector:
            parts.append("det")
        if self.use_candidate_head:
            parts.append("cand")
        if self.use_gated_fusion:
            parts.append("gate")
        if self.error_position_weight > 0:
            parts.append(f"ew{self.error_position_weight}")
        
        if not parts:
            return "baseline"
        return "_".join(parts)


# 预定义的消融实验配置
ABLATION_CONFIGS = {
    # 完整DGCA-ReLM
    "full": DGCAConfig(
        use_detector=True,
        use_candidate_head=True,
        use_gated_fusion=True,
    ),
    # 仅检测分支（无候选头）
    "detector_only": DGCAConfig(
        use_detector=True,
        use_candidate_head=False,
        use_gated_fusion=False,
    ),
    # 仅候选头（无检测门控）
    "candidate_only": DGCAConfig(
        use_detector=False,
        use_candidate_head=True,
        use_gated_fusion=False,
    ),
    # ReLM基线（无DGCA模块）
    "baseline": DGCAConfig(
        use_detector=False,
        use_candidate_head=False,
        use_gated_fusion=False,
    ),
}


def get_ablation_config(name: str) -> DGCAConfig:
    """获取预定义的消融实验配置"""
    if name in ABLATION_CONFIGS:
        return ABLATION_CONFIGS[name]
    raise ValueError(f"Unknown ablation config: {name}. Available: {list(ABLATION_CONFIGS.keys())}")
