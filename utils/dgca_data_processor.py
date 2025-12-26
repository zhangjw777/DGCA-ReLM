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
    生成错误位置标签
    
    策略：
    1. prompt位置、padding位置：-100（ignore）
    2. 源句部分（SEP之前）：比较src和trg_ref，不同为1
    3. 目标句部分（SEP之后的mask）：比较trg_ref和trg，不同为1
    
    Args:
        src_ids: 源句token ids（含prompt和mask）
        trg_ids: 目标句token ids（含prompt和正确答案）
        trg_ref_ids: 参考（用于判断原始是否有错）
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
        
        # 源句部分（SEP之前）
        if not passed_sep:
            # 源句部分不需要预测，但可以用于检测
            # 这里标记为-100，因为不参与纠错训练
            error_labels.append(-100)
        else:
            # 目标句部分（mask位置）
            if s == mask_token_id:
                # 这是需要预测的位置
                # 判断是否有错：trg_ref对应位置 vs trg对应位置
                if r != t:
                    error_labels.append(1)  # 有错
                else:
                    error_labels.append(0)  # 无错
            else:
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
        # trg_ref后半段是src，用于判断哪些位置有错
        trg_ref = [tokenizer.cls_token] * prompt_length + src + anchor + \
                  [tokenizer.sep_token] * prompt_length + src
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + \
                     [1] * prompt_length + [0 for _ in trg]
        # trg_ref后半段是src，用于判断哪些位置有错
        trg_ref = [tokenizer.cls_token] * prompt_length + src + \
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
