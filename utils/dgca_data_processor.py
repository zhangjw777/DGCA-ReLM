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


class DGCAInputFeatures:
    """DGCA-ReLM的输入特征"""
    
    def __init__(
        self,
        src_ids: List[int],
        attention_mask: List[int],
        trg_ids: List[int],
        trg_ref_ids: List[int],
        block_flag: List[int],
        error_labels: List[int],
        candidate_ids: List[List[int]]
    ):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag
        self.error_labels = error_labels
        self.candidate_ids = candidate_ids


def convert_examples_to_dgca_features(
    examples,
    max_seq_length: int,
    tokenizer,
    confusion_set,
    prompt_length: int,
    anchor=None,
    mask_rate: float = 0.2
) -> List[DGCAInputFeatures]:
    """
    将examples转换为DGCA-ReLM的features
    在ReLM基础上添加error_labels和candidate_ids
    
    Args:
        examples: 输入样本列表
        max_seq_length: 最大序列长度
        tokenizer: tokenizer
        confusion_set: ConfusionSet实例
        prompt_length: prompt长度
        anchor: anchor tokens
        mask_rate: aux MLM的mask比例
        
    Returns:
        DGCAInputFeatures列表
    """
    features = []
    
    for i, example in enumerate(examples):
        # 沿用ReLM的prompt转换
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(
            example.src, example.trg, prompt_length, 
            max_seq_length // 2, tokenizer, anchor, mask_rate
        )
        
        example.src = src
        example.trg = trg
        
        # Tokenize
        encoded_inputs = tokenizer(
            example.src,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True
        )
        
        trg_ids = tokenizer(
            example.trg,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True
        )["input_ids"]
        
        trg_ref_ids = tokenizer(
            trg_ref,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True
        )["input_ids"]
        
        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        
        # 处理block_flag
        block_flag = ([0] + block_flag)[:max_seq_length]
        if len(block_flag) < max_seq_length:
            block_flag = block_flag + [0] * (max_seq_length - len(block_flag))
        
        # ========== 生成error_labels ==========
        # 比较src和trg，不同的位置标记为错误
        error_labels = []
        for s, t in zip(src_ids, trg_ids):
            if s == t or s == tokenizer.pad_token_id:
                # 相同或padding位置标记为-100（ignore）
                error_labels.append(-100)
            else:
                # mask位置：如果src是mask，需要检查对应的原始token
                # 这里简化处理：mask位置根据原始src_ids和trg_ids判断
                # 实际上mask位置应该标记为可能有错
                if s == tokenizer.mask_token_id:
                    # 查找这个位置对应的原始字符（在trg_ref中）
                    # 简化：假设mask位置都可能有错，标记为需要检测
                    # 但实际错误取决于原始src和trg的对比
                    # 这里使用trg_ref_ids来判断
                    error_labels.append(1 if trg_ref_ids[len(error_labels)] != t else 0)
                else:
                    error_labels.append(1)  # 不同即为错误
        
        # 更精确的error_labels生成：
        # 重新生成，基于原始src和trg的对比
        error_labels = generate_error_labels(
            src_ids, trg_ids, trg_ref_ids, tokenizer
        )
        
        # ========== 生成candidate_ids ==========
        # 为每个位置生成候选集
        candidate_ids_list = []
        for src_id in src_ids:
            candidates = confusion_set.get_candidates(src_id)
            # Pad到candidate_size
            padded_candidates = candidates + [tokenizer.pad_token_id] * (
                confusion_set.candidate_size - len(candidates)
            )
            candidate_ids_list.append(padded_candidates[:confusion_set.candidate_size])
        
        # 验证长度
        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(trg_ids) == max_seq_length
        assert len(trg_ref_ids) == max_seq_length
        assert len(block_flag) == max_seq_length
        assert len(error_labels) == max_seq_length
        assert len(candidate_ids_list) == max_seq_length
        
        if i < 5:
            logger.info("*** DGCA Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("src_tokens: %s" % " ".join(example.src))
            logger.info("trg_tokens: %s" % " ".join(example.trg))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
            logger.info("error_labels: %s" % " ".join([str(x) for x in error_labels]))
            logger.info("candidate_size: %d" % len(candidate_ids_list[0]))
        
        features.append(
            DGCAInputFeatures(
                src_ids=src_ids,
                attention_mask=attention_mask,
                trg_ids=trg_ids,
                trg_ref_ids=trg_ref_ids,
                block_flag=block_flag,
                error_labels=error_labels,
                candidate_ids=candidate_ids_list
            )
        )
    
    return features


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
    沿用ReLM的prompt转换逻辑
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
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + \
                  [tokenizer.sep_token] * prompt_length + trg
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + [tokenizer.mask_token for _ in trg]
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + \
                     [tokenizer.sep_token] * prompt_length + trg
        block_flag = [1] * prompt_length + [0 for _ in src] + \
                     [1] * prompt_length + [0 for _ in trg]
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + \
                  [tokenizer.sep_token] * prompt_length + trg
    
    return prompt_src, prompt_trg, block_flag, trg_ref


def create_dgca_dataset(features: List[DGCAInputFeatures]) -> TensorDataset:
    """
    将features转换为TensorDataset
    
    Args:
        features: DGCAInputFeatures列表
        
    Returns:
        TensorDataset
    """
    all_input_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.trg_ids for f in features], dtype=torch.long)
    all_trg_ref_ids = torch.tensor([f.trg_ref_ids for f in features], dtype=torch.long)
    all_block_flag = torch.tensor([f.block_flag for f in features], dtype=torch.long)
    all_error_labels = torch.tensor([f.error_labels for f in features], dtype=torch.long)
    
    # candidate_ids是2D list，需要转为3D tensor
    all_candidate_ids = torch.tensor([f.candidate_ids for f in features], dtype=torch.long)
    
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_label_ids,
        all_trg_ref_ids,
        all_block_flag,
        all_error_labels,
        all_candidate_ids
    )


# ============================================================================
# 以下是针对百万级数据的优化版本
# ============================================================================

class DGCALazyDataset(Dataset):
    """
    懒加载Dataset，用于百万级大规模数据
    不一次性将所有数据加载到内存，而是按需加载
    支持预处理缓存，加速后续训练
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        confusion_set,
        max_seq_length: int = 128,
        prompt_length: int = 3,
        anchor: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        precompute: bool = True
    ):
        """
        Args:
            data_file: 数据文件路径（格式：src\ttrg，每行一个样本）
            tokenizer: tokenizer
            confusion_set: ConfusionSet实例
            max_seq_length: 最大序列长度
            prompt_length: prompt长度
            anchor: anchor tokens
            cache_dir: 缓存目录，用于存储预处理结果
            precompute: 是否预计算（首次加载较慢，后续加载快）
        """
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.confusion_set = confusion_set
        self.max_seq_length = max_seq_length
        self.prompt_length = prompt_length
        self.anchor = anchor
        self.cache_dir = cache_dir
        
        # 预计算候选集映射表（token_id -> candidate_ids）
        self._precompute_candidates()
        
        # 加载数据索引
        self.line_offsets = []
        self._build_index()
        
        # 缓存
        self.cache = {}
        self.cache_path = None
        if cache_dir and precompute:
            self._try_load_cache()
    
    def _precompute_candidates(self):
        """预计算所有token的候选集，避免运行时查找"""
        self.candidate_lookup: Dict[int, List[int]] = {}
        candidate_size = self.confusion_set.candidate_size
        pad_id = self.tokenizer.pad_token_id
        
        for token_id in range(self.tokenizer.vocab_size):
            candidates = self.confusion_set.get_candidates(token_id)
            # Pad到固定长度
            padded = candidates + [pad_id] * (candidate_size - len(candidates))
            self.candidate_lookup[token_id] = padded[:candidate_size]
    
    def _build_index(self):
        """构建文件行索引，用于随机访问"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode('utf-8'))
        logger.info(f"Indexed {len(self.line_offsets)} samples from {self.data_file}")
    
    def _try_load_cache(self):
        """尝试加载预处理缓存"""
        if not self.cache_dir:
            return
        
        cache_name = os.path.basename(self.data_file).replace('.txt', '_cache.pt')
        self.cache_path = os.path.join(self.cache_dir, cache_name)
        
        if os.path.exists(self.cache_path):
            logger.info(f"Loading cache from {self.cache_path}")
            self.cache = torch.load(self.cache_path)
            logger.info(f"Loaded {len(self.cache)} cached samples")
    
    def save_cache(self):
        """保存预处理缓存"""
        if self.cache_path and self.cache:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            torch.save(self.cache, self.cache_path)
            logger.info(f"Saved cache to {self.cache_path}")
    
    def __len__(self):
        return len(self.line_offsets)
    
    def _read_line(self, idx: int) -> Tuple[List[str], List[str]]:
        """读取指定行"""
        with open(self.data_file, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().strip()
            parts = line.split('\t')
            if len(parts) == 2:
                src = parts[0].split()
                trg = parts[1].split()
                return src, trg
            return [], []
    
    def _process_sample(self, src: List[str], trg: List[str]) -> Dict[str, torch.Tensor]:
        """处理单个样本"""
        # Prompt转换
        prompt_src, prompt_trg, block_flag, trg_ref = convert_examples_to_prompts(
            src, trg, self.prompt_length,
            self.max_seq_length // 2, self.tokenizer, self.anchor
        )
        
        # Tokenize
        encoded_src = self.tokenizer(
            prompt_src,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            is_split_into_words=True
        )
        
        src_ids = encoded_src["input_ids"]
        attention_mask = encoded_src["attention_mask"]
        
        trg_ids = self.tokenizer(
            prompt_trg,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True
        )["input_ids"]
        
        trg_ref_ids = self.tokenizer(
            trg_ref,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            is_split_into_words=True
        )["input_ids"]
        
        # Block flag处理
        block_flag = ([0] + block_flag)[:self.max_seq_length]
        block_flag = block_flag + [0] * (self.max_seq_length - len(block_flag))
        
        # Error labels
        error_labels = generate_error_labels(
            src_ids, trg_ids, trg_ref_ids, self.tokenizer
        )
        
        # Candidate ids（使用预计算的查找表）
        candidate_ids = [self.candidate_lookup.get(tid, [self.tokenizer.pad_token_id] * self.confusion_set.candidate_size) 
                        for tid in src_ids]
        
        return {
            'input_ids': torch.tensor(src_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(trg_ids, dtype=torch.long),
            'trg_ref_ids': torch.tensor(trg_ref_ids, dtype=torch.long),
            'block_flag': torch.tensor(block_flag, dtype=torch.long),
            'error_labels': torch.tensor(error_labels, dtype=torch.long),
            'candidate_ids': torch.tensor(candidate_ids, dtype=torch.long)
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 检查缓存
        if idx in self.cache:
            return self.cache[idx]
        
        # 读取并处理
        src, trg = self._read_line(idx)
        if not src or not trg:
            # 返回空样本
            return self._get_empty_sample()
        
        sample = self._process_sample(src, trg)
        
        # 缓存（内存允许时）
        if len(self.cache) < 100000:  # 最多缓存10万样本
            self.cache[idx] = sample
        
        return sample
    
    def _get_empty_sample(self):
        """返回空样本（用于处理损坏数据）"""
        pad_id = self.tokenizer.pad_token_id
        cand_size = self.confusion_set.candidate_size
        
        return {
            'input_ids': torch.zeros(self.max_seq_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_seq_length, dtype=torch.long),
            'labels': torch.full((self.max_seq_length,), -100, dtype=torch.long),
            'trg_ref_ids': torch.zeros(self.max_seq_length, dtype=torch.long),
            'block_flag': torch.zeros(self.max_seq_length, dtype=torch.long),
            'error_labels': torch.full((self.max_seq_length,), -100, dtype=torch.long),
            'candidate_ids': torch.full((self.max_seq_length, cand_size), pad_id, dtype=torch.long)
        }


def preprocess_large_dataset_parallel(
    data_file: str,
    output_file: str,
    tokenizer,
    confusion_set,
    max_seq_length: int = 128,
    prompt_length: int = 3,
    anchor: Optional[List[str]] = None,
    num_workers: int = None
):
    """
    多进程并行预处理大规模数据集
    将处理结果保存为torch格式，供后续直接加载
    
    Args:
        data_file: 输入数据文件
        output_file: 输出文件路径（.pt格式）
        tokenizer: tokenizer
        confusion_set: ConfusionSet
        max_seq_length: 最大序列长度
        prompt_length: prompt长度
        anchor: anchor tokens
        num_workers: 并行进程数，默认为CPU核心数的一半
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)
    
    logger.info(f"Preprocessing with {num_workers} workers...")
    
    # 读取所有数据
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    logger.info(f"Loaded {len(lines)} lines")
    
    # 准备处理函数
    def process_line(args):
        idx, line = args
        try:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                return None
            src = parts[0].split()
            trg = parts[1].split()
            
            # Prompt转换
            prompt_src, prompt_trg, block_flag, trg_ref = convert_examples_to_prompts(
                src, trg, prompt_length, max_seq_length // 2, tokenizer, anchor
            )
            
            # Tokenize
            encoded_src = tokenizer(
                prompt_src,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                is_split_into_words=True
            )
            
            src_ids = encoded_src["input_ids"]
            attention_mask = encoded_src["attention_mask"]
            
            trg_ids = tokenizer(
                prompt_trg,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                is_split_into_words=True
            )["input_ids"]
            
            trg_ref_ids = tokenizer(
                trg_ref,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                is_split_into_words=True
            )["input_ids"]
            
            block_flag = ([0] + block_flag)[:max_seq_length]
            block_flag = block_flag + [0] * (max_seq_length - len(block_flag))
            
            error_labels = generate_error_labels(src_ids, trg_ids, trg_ref_ids, tokenizer)
            
            # Candidate ids
            candidate_ids = []
            for tid in src_ids:
                cands = confusion_set.get_candidates(tid)
                padded = cands + [tokenizer.pad_token_id] * (confusion_set.candidate_size - len(cands))
                candidate_ids.append(padded[:confusion_set.candidate_size])
            
            return {
                'src_ids': src_ids,
                'attention_mask': attention_mask,
                'trg_ids': trg_ids,
                'trg_ref_ids': trg_ref_ids,
                'block_flag': block_flag,
                'error_labels': error_labels,
                'candidate_ids': candidate_ids
            }
        except Exception as e:
            logger.warning(f"Error processing line {idx}: {e}")
            return None
    
    # 并行处理
    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(process_line, enumerate(lines), chunksize=1000),
            total=len(lines),
            desc="Processing"
        ):
            if result is not None:
                results.append(result)
    
    logger.info(f"Successfully processed {len(results)} samples")
    
    # 转换为tensor并保存
    all_data = {
        'input_ids': torch.tensor([r['src_ids'] for r in results], dtype=torch.long),
        'attention_mask': torch.tensor([r['attention_mask'] for r in results], dtype=torch.long),
        'labels': torch.tensor([r['trg_ids'] for r in results], dtype=torch.long),
        'trg_ref_ids': torch.tensor([r['trg_ref_ids'] for r in results], dtype=torch.long),
        'block_flag': torch.tensor([r['block_flag'] for r in results], dtype=torch.long),
        'error_labels': torch.tensor([r['error_labels'] for r in results], dtype=torch.long),
        'candidate_ids': torch.tensor([r['candidate_ids'] for r in results], dtype=torch.long)
    }
    
    torch.save(all_data, output_file)
    logger.info(f"Saved preprocessed data to {output_file}")


class PreprocessedDataset(Dataset):
    """
    加载预处理好的数据集
    
    优化：
    1. 将数据转为连续的内存布局，加速索引
    2. 支持返回元组模式，减少字典构造开销
    3. 预先将数据移到共享内存（可选，用于多worker）
    """
    
    def __init__(self, preprocessed_file: str, use_shared_memory: bool = True):
        """
        Args:
            preprocessed_file: 预处理好的.pt文件路径
            use_shared_memory: 是否将数据放到共享内存（推荐True，支持多worker高效访问）
        """
        logger.info(f"Loading preprocessed data from {preprocessed_file}")
        raw_data = torch.load(preprocessed_file)
        
        # 将数据转为连续内存布局，加速索引
        # 并可选地移到共享内存，避免多worker时的数据复制
        self.input_ids = raw_data['input_ids'].contiguous()
        self.attention_mask = raw_data['attention_mask'].contiguous()
        self.labels = raw_data['labels'].contiguous()
        self.trg_ref_ids = raw_data['trg_ref_ids'].contiguous()
        self.block_flag = raw_data['block_flag'].contiguous()
        self.error_labels = raw_data['error_labels'].contiguous()
        self.candidate_ids = raw_data['candidate_ids'].contiguous()
        
        if use_shared_memory:
            # 移到共享内存，多worker可以直接访问而无需复制
            self.input_ids = self.input_ids.share_memory_()
            self.attention_mask = self.attention_mask.share_memory_()
            self.labels = self.labels.share_memory_()
            self.trg_ref_ids = self.trg_ref_ids.share_memory_()
            self.block_flag = self.block_flag.share_memory_()
            self.error_labels = self.error_labels.share_memory_()
            self.candidate_ids = self.candidate_ids.share_memory_()
        
        self.size = self.input_ids.shape[0]
        logger.info(f"Loaded {self.size} samples (shared_memory={use_shared_memory})")
        
        # 释放原始数据引用
        del raw_data
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 直接返回字典，Tensor切片操作非常快
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'trg_ref_ids': self.trg_ref_ids[idx],
            'block_flag': self.block_flag[idx],
            'error_labels': self.error_labels[idx],
            'candidate_ids': self.candidate_ids[idx]
        }
