#!/usr/bin/env python
"""
数据预处理脚本（v2）
干净句子 -> 多字造错 -> 预处理为jsonl格式 -> 划分train/dev/test

使用方法：
python preprocess_data.py \
    --input_file data/clean_sentences.txt \
    --output_dir data/processed \
    --model_path bert-base-chinese \
    --confusion_dir confusion

输入：每行一个干净句子的txt文件
输出：output_dir下生成 train.jsonl, dev.jsonl, test.jsonl
"""

import argparse
import os
import sys
import random
import json
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
from confusion.confusion_utils import ConfusionSet
from config.dgca_config import DGCAConfig


class ErrorGenerator:
    """基于混淆集的造错器 - 支持多字造错"""
    
    def __init__(self, confusion_dir: str, seed: int = 42):
        """
        Args:
            confusion_dir: 混淆集目录，包含pinyin_sam.json, pinyin_sim.json, stroke.json, word_freq.txt
            seed: 随机种子
        """
        self.confusion_dir = confusion_dir
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载四种混淆集
        self.pinyin_sam = self._load_json("pinyin_sam.json")  # 完全同音
        self.pinyin_sim = self._load_json("pinyin_sim.json")  # 发音相似
        self.stroke = self._load_json("stroke.json")          # 字形相似
        self.word_freq = self._load_word_freq("word_freq.txt") # 高频词（fallback）
        
        print(f"混淆集加载完成: pinyin_sam={len(self.pinyin_sam)}, "
              f"pinyin_sim={len(self.pinyin_sim)}, stroke={len(self.stroke)}, "
              f"word_freq={len(self.word_freq)}")
    
    def _load_json(self, filename: str) -> dict:
        path = os.path.join(self.confusion_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"警告: {path} 不存在，使用空字典")
        return {}
    
    def _load_word_freq(self, filename: str) -> List[str]:
        path = os.path.join(self.confusion_dir, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        print(f"警告: {path} 不存在，使用空列表")
        return []
    
    @staticmethod
    def is_chinese(char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'
    
    def _get_replacement(self, original_char: str, max_retries: int = 5) -> str:
        """
        获取替换字符，按优先级尝试：
        40% pinyin_sam -> 30% pinyin_sim -> 20% stroke -> 10% word_freq
        如果采样到相同字符，重试
        """
        for _ in range(max_retries):
            rn = random.random()
            new_char = None
            
            if rn < 0.4 and original_char in self.pinyin_sam and self.pinyin_sam[original_char]:
                # 40% 完全同音
                new_char = random.choice(self.pinyin_sam[original_char])
            elif rn < 0.7 and original_char in self.pinyin_sim and self.pinyin_sim[original_char]:
                # 30% 发音相似
                new_char = random.choice(self.pinyin_sim[original_char])
            elif rn < 0.9 and original_char in self.stroke and self.stroke[original_char]:
                # 20% 字形相似
                new_char = random.choice(self.stroke[original_char])
            elif self.word_freq:
                # 10% 高频词fallback
                new_char = random.choice(self.word_freq)
            
            # 如果采样到有效且不同的字符，返回
            if new_char is not None and new_char != original_char:
                return new_char
        
        # 重试多次仍然失败，返回原字符（不造错）
        return original_char
    
    def generate_error(
        self, 
        sentence: str, 
        avg_errors: float = 1.5,
        no_error_ratio: float = 0.15
    ) -> Tuple[str, str]:
        """
        对干净句子造错，返回(错误句, 正确句)
        
        造错策略：
        - 15%概率不造错（让模型学会"不改"）
        - 剩余85%使用泊松分布确定错误数量（lambda=avg_errors）
        - 每个错误位置：40%同音 + 30%近音 + 20%字形 + 10%高频词fallback
        
        Args:
            sentence: 干净句子
            avg_errors: 平均错误数（泊松分布的lambda参数）
            no_error_ratio: 不造错的比例
        """
        # 一定比例不造错
        if random.random() < no_error_ratio:
            return sentence, sentence
        
        tokens = list(sentence)
        
        # 找出可造错的位置（中文字符）
        chinese_indices = [i for i, t in enumerate(tokens) if self.is_chinese(t)]
        
        if not chinese_indices:
            return sentence, sentence
        
        # 使用泊松分布确定错误数量，最少1个，最多不超过可造错位置数
        num_errors = min(
            max(1, np.random.poisson(avg_errors)),
            len(chinese_indices)
        )
        
        # 随机选择造错位置
        error_positions = random.sample(chinese_indices, num_errors)
        
        # 对每个位置造错
        for pos in error_positions:
            original_char = tokens[pos]
            new_char = self._get_replacement(original_char)
            tokens[pos] = new_char
        
        src = ''.join(tokens)
        trg = sentence
        return src, trg


def split_data(
    data: List, 
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), 
    seed: int = 42
) -> Tuple[List, List, List]:
    """划分数据集"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * ratios[0])
    dev_end = train_end + int(n * ratios[1])
    
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def process_sample(
    src_chars: List[str],
    trg_chars: List[str],
    tokenizer,
    confusion_set,
    max_seq_length: int,
    prompt_length: int
) -> Dict[str, Any]:
    """处理单个样本，返回预计算的特征字典"""
    from utils.dgca_data_processor import convert_examples_to_prompts, generate_error_labels
    
    # 截断到合适长度
    half_len = max_seq_length // 2 - prompt_length
    src_chars = src_chars[:half_len]
    trg_chars = trg_chars[:half_len]
    
    # 确保长度一致
    min_len = min(len(src_chars), len(trg_chars))
    src_chars = src_chars[:min_len]
    trg_chars = trg_chars[:min_len]
    
    if len(src_chars) == 0:
        return None
    
    # Prompt转换（返回的序列已经包含CLS/SEP等special tokens）
    prompt_src, prompt_trg, block_flag, trg_ref = convert_examples_to_prompts(
        src_chars, trg_chars, prompt_length, max_seq_length // 2, tokenizer, anchor=None
    )
    
    # Tokenize时不添加special tokens（因为prompt序列中已包含）
    # 使用 add_special_tokens=False 确保 block_flag 与 token 序列完全对齐
    encoded_src = tokenizer(
        prompt_src, max_length=max_seq_length,
        padding="max_length", truncation=True, 
        is_split_into_words=True, add_special_tokens=False
    )
    src_ids = encoded_src["input_ids"]
    attention_mask = encoded_src["attention_mask"]
    
    trg_ids = tokenizer(
        prompt_trg, max_length=max_seq_length,
        padding="max_length", truncation=True, 
        is_split_into_words=True, add_special_tokens=False
    )["input_ids"]
    
    trg_ref_ids = tokenizer(
        trg_ref, max_length=max_seq_length,
        padding="max_length", truncation=True, 
        is_split_into_words=True, add_special_tokens=False
    )["input_ids"]
    
    # Block flag处理：直接使用，确保长度对齐
    # 因为 add_special_tokens=False，所以 block_flag 与 token 序列一一对应
    block_flag = block_flag[:max_seq_length]
    block_flag = block_flag + [0] * (max_seq_length - len(block_flag))
    
    # Error labels
    error_labels = generate_error_labels(src_ids, trg_ids, trg_ref_ids, tokenizer)
    
    # Candidate ids：PAD位置全部填充pad_token_id
    pad_id = tokenizer.pad_token_id
    candidate_ids = []
    for i, tid in enumerate(src_ids):
        if attention_mask[i] == 0:  # PAD位置
            # PAD位置的候选集全部填充pad_token_id
            candidate_ids.append([pad_id] * confusion_set.candidate_size)
        else:
            cands = confusion_set.get_candidates(tid)
            padded = cands + [pad_id] * (confusion_set.candidate_size - len(cands))
            candidate_ids.append(padded[:confusion_set.candidate_size])
    
    return {
        'input_ids': src_ids,
        'attention_mask': attention_mask,
        'labels': trg_ids,
        'trg_ref_ids': trg_ref_ids,
        'block_flag': block_flag,
        'error_labels': error_labels,
        'candidate_ids': candidate_ids
    }


def preprocess_and_save_jsonl(
    pairs: List[Tuple[List[str], List[str]]],
    tokenizer,
    confusion_set,
    max_seq_length: int,
    prompt_length: int,
    output_path: str
):
    """预处理数据并保存为jsonl格式（流式写入，避免OOM）"""
    count = 0
    
    # 流式写入，边处理边写文件，避免内存爆炸
    with open(output_path, 'w', encoding='utf-8') as f:
        for src, trg in tqdm(pairs, desc=f"处理 {os.path.basename(output_path)}"):
            try:
                sample = process_sample(
                    src, trg, tokenizer, confusion_set, max_seq_length, prompt_length
                )
                if sample is not None:
                    # 直接写入jsonl，每行一个json对象
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    count += 1
            except Exception as e:
                continue
    
    print(f"保存: {output_path}, 样本数: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description="干净句子造错+预处理为jsonl")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入文件（每行一个干净句子）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，生成train.jsonl, dev.jsonl, test.jsonl")
    parser.add_argument("--model_path", type=str, default="bert-base-chinese",
                        help="预训练模型路径")
    parser.add_argument("--confusion_dir", type=str, default="confusion",
                        help="混淆集目录")
    parser.add_argument("--dgca_config", type=str, default="config/default_config.yaml",
                        help="DGCA配置文件")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--prompt_length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=str, default="0.8,0.1,0.1",
                        help="train,dev,test划分比例")
    parser.add_argument("--avg_errors", type=float, default=1.5,
                        help="平均每句错误数（泊松分布lambda）")
    parser.add_argument("--no_error_ratio", type=float, default=0.15,
                        help="无错样本比例，默认0.15")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析划分比例
    ratios = tuple(map(float, args.split_ratio.split(',')))
    assert abs(sum(ratios) - 1.0) < 1e-6, "划分比例之和应为1"
    
    print(f"加载Tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"加载DGCA配置: {args.dgca_config}")
    dgca_config = DGCAConfig.from_yaml(args.dgca_config)
    
    print(f"加载混淆集...")
    confusion_set = ConfusionSet(
        confusion_dir=args.confusion_dir,
        tokenizer=tokenizer,
        candidate_size=dgca_config.candidate_size,
        include_original=dgca_config.include_original_char
    )
    
    # 读取干净句子
    print(f"读取干净句子: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"共 {len(sentences)} 个句子")
    
    # 造错
    print(f"开始造错（平均错误数: {args.avg_errors}, 无错比例: {args.no_error_ratio*100:.0f}%）...")
    error_gen = ErrorGenerator(args.confusion_dir, seed=args.seed)
    
    pairs = []
    error_stats = {0: 0, 1: 0, 2: 0, 3: 0, '4+': 0}
    
    for sentence in tqdm(sentences, desc="造错"):
        src, trg = error_gen.generate_error(
            sentence, 
            avg_errors=args.avg_errors,
            no_error_ratio=args.no_error_ratio
        )
        pairs.append((list(src), list(trg)))
        
        # 统计错误数
        num_errors = sum(1 for s, t in zip(src, trg) if s != t)
        if num_errors >= 4:
            error_stats['4+'] += 1
        else:
            error_stats[num_errors] += 1
    
    print(f"\n造错统计:")
    for k, v in sorted(error_stats.items(), key=lambda x: str(x[0])):
        print(f"  {k}个错误: {v} ({v/len(pairs)*100:.1f}%)")
    
    # 划分数据集
    print(f"\n划分数据集 (比例: {ratios})...")
    train_pairs, dev_pairs, test_pairs = split_data(pairs, ratios, args.seed)
    print(f"Train: {len(train_pairs)}, Dev: {len(dev_pairs)}, Test: {len(test_pairs)}")
    
    # 预处理并保存
    for name, data_pairs in [("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)]:
        print(f"\n预处理 {name}...")
        output_path = os.path.join(args.output_dir, f"{name}.jsonl")
        preprocess_and_save_jsonl(
            data_pairs, tokenizer, confusion_set,
            args.max_seq_length, args.prompt_length, output_path
        )
    
    print("\n预处理完成！")


if __name__ == "__main__":
    main()
