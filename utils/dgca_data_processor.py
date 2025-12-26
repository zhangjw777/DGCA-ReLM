"""
DGCA-ReLM数据处理模块
扩展原ReLM的数据处理，添加error_labels和candidate_ids生成

针对百万级数据优化：
1. 使用多进程并行处理
2. 支持流式加载（Dataset类）
3. 预计算候选集映射表
"""

import torch
from torch.utils.data import TensorDataset, Dataset
from typing import List, Tuple, Optional, Dict
import logging
import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_error_labels(
    src_ids: List[int],
    trg_ids: List[int],
    trg_ref_ids: List[int],
    tokenizer
) -> List[int]:
    """
    生成错误位置标签（用于检测分支 DetectorHead 训练）
    
    设计原则：
    检测器应从源句侧预测哪些位置有错误，因此 error_labels 在源句部分生成 0/1 标签。
    
    策略：
    1. prompt位置（CLS/SEP）、padding位置：-100（ignore）
    2. 源句部分（SEP之前的实际字符）：比较 src 和 trg_ref（正确答案），不同为1，相同为0
    3. 目标句部分（SEP之后的Mask区域）：-100（ignore，纠错由CandidateHead负责）
    
    数据对齐说明：
    - trg_ref_ids 的前半部分（源句区域）存储的是 trg（正确答案）
    - trg_ref_ids 的后半部分（目标区域）存储的是 src（原始错误句）
    
    Args:
        src_ids: 源句token ids（含prompt和mask），格式: [CLS]*P + src + [SEP]*P + [MASK]*n
        trg_ids: 目标句token ids（含prompt和正确答案），格式: [CLS]*P + src + [SEP]*P + trg
        trg_ref_ids: 参考ids，格式: [CLS]*P + trg + [SEP]*P + src
        tokenizer: tokenizer
        
    Returns:
        error_labels: 错误标签列表（1=错误，0=正确，-100=ignore）
    """
    error_labels = []
    sep_token_id = tokenizer.sep_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    
    passed_sep = False
    
    for i, (s, t, r) in enumerate(zip(src_ids, trg_ids, trg_ref_ids)):
        # Padding位置
        if s == pad_token_id:
            error_labels.append(-100)
            continue
        
        # SEP标记分界
        if s == sep_token_id:
            passed_sep = True
            error_labels.append(-100)  # SEP本身ignore
            continue
        
        # 源句部分（SEP之前）—— 检测器在此预测
        if not passed_sep:
            # CLS token 跳过
            if s == tokenizer.cls_token_id:
                error_labels.append(-100)
            else:
                # 在源句位置，比较 src(s) 和 trg_ref(r)（正确答案）
                # 若 s != r，说明该位置有错，标记为1；否则为0
                is_error = 1 if s != r else 0
                error_labels.append(is_error)
        else:
            # 目标句部分（Mask区域）—— 检测器不需要预测，设为-100
            # 纠错任务由 CandidateHead 在此区域完成
            error_labels.append(-100)
    
    return error_labels

def convert_examples_to_prompts(
    src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2
):
    """
    ReLM的prompt转换逻辑
    
    输出格式：
    - prompt_src: [CLS]*P + src + [SEP]*P + [MASK]*len(trg)  (模型输入)
    - prompt_trg: [CLS]*P + src + [SEP]*P + trg              (目标输出)
    - block_flag: 标记哪些位置是prompt（用于prompt embedding）
    - trg_ref:    [CLS]*P + src + [SEP]*P + src              (参考，用于生成error_labels)
    
    注意：trg_ref 的后半段是 src（原始错误句），这样在 mask 段比较 trg_ref vs trg 
    才能正确判断哪些位置有错（src[i] != trg[i] 表示有错）
    """
    def truncate(x, max_length):
        return x[:max_length]
    
    src = truncate(src, max_seq_length - prompt_length)
    trg = truncate(trg, max_seq_length - prompt_length)
    assert len(src) == len(trg)
    
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + \
                     [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + anchor + \
                     [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + \
                     [1] * prompt_length + [0 for _ in trg]
        # trg_ref 设计：
        # - 前半部分（源句区域）：trg（正确答案），用于检测器判断源句哪些位置有错
        # - 后半部分（目标区域）：src（原始错误句），用于纠错损失判断哪些mask位置需要修正
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + \
                  [tokenizer.sep_token] * prompt_length + src
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + \
                     [1] * prompt_length + [0 for _ in trg]
        # trg_ref 设计：
        # - 前半部分（源句区域）：trg（正确答案），用于检测器判断源句哪些位置有错
        # - 后半部分（目标区域）：src（原始错误句），用于纠错损失判断哪些mask位置需要修正
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + \
                  [tokenizer.sep_token] * prompt_length + src
    
    return prompt_src, prompt_trg, block_flag, trg_ref

class PreprocessedDataset(Dataset):
    """加载预处理好的 jsonl 数据集"""
    
    def __init__(self, preprocessed_file: str):
        """
        Args:
            preprocessed_file: 预处理好的 jsonl 文件路径
        """
        logger.info(f"Loading preprocessed data from {preprocessed_file}")
        
        from datasets import load_dataset
        
        # 使用 datasets 库加载 jsonl（自动使用内存映射）
        ds = load_dataset('json', data_files=preprocessed_file, split='train')
        
        # 设置 torch 格式
        ds.set_format(type='torch', columns=[
            'input_ids', 'attention_mask', 'labels', 
            'trg_ref_ids', 'block_flag', 'error_labels', 'candidate_ids'
        ])
        
        self._hf_dataset = ds
        self.size = len(ds)
        
        logger.info(f"Loaded {self.size} samples from jsonl")
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self._hf_dataset[idx]
