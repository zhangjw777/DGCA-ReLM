#!/usr/bin/env python
"""
大规模数据预处理脚本
将百万级干净句子 -> 造错 -> 预处理为.pt格式 -> 划分train/dev/test

使用方法：
python preprocess_data.py \
    --input_file data/clean_sentences.txt \
    --output_dir data/processed \
    --model_path bert-base-chinese \
    --confusion_dir confusion \
    --num_workers 16

输入：每行一个干净句子的txt文件
输出：output_dir下生成 train.pt, dev.pt, test.pt
"""

import argparse
import os
import sys
import random
import json
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from confusion.confusion_utils import ConfusionSet
from config.dgca_config import DGCAConfig


class ErrorGenerator:
    """基于混淆集的造错器（参考lemon的ConfusDataset）"""
    
    def __init__(self, confusion_dir: str, seed: int = 42):
        """
        Args:
            confusion_dir: 混淆集目录，包含pinyin_sim.json, stroke.json, word_freq.txt
            seed: 随机种子
        """
        self.confusion_dir = confusion_dir
        random.seed(seed)
        
        # 加载三种混淆集
        self.pinyin_sim = self._load_json("pinyin_sim.json")
        self.stroke = self._load_json("stroke.json")
        self.word_freq = self._load_word_freq("word_freq.txt")
        
        print(f"混淆集加载完成: pinyin_sim={len(self.pinyin_sim)}, "
              f"stroke={len(self.stroke)}, word_freq={len(self.word_freq)}")
    
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
    
    def generate_error(self, sentence: str, no_error_ratio: float = 0.2) -> Tuple[str, str]:
        """
        对干净句子造错，返回(错误句, 正确句)
        
        造错策略（参考lemon）:
        - 20%概率不造错，直接返回原句（让模型学会"不改"）
        - 剩余80%中:
          - 50%拼音相近替换
          - 30%字形相似替换
          - 20%随机高频词替换
        
        Args:
            sentence: 干净句子
            no_error_ratio: 不造错的比例，默认0.2（20%）
        """
        # 20%概率不造错，让模型学会"不改"
        if random.random() < no_error_ratio:
            return sentence, sentence
        
        tokens = list(sentence)
        chinese_indices = [i for i, t in enumerate(tokens) if self.is_chinese(t)]
        
        if not chinese_indices:
            return sentence, sentence  # 无中文，不造错
        
        # 随机选一个位置造错
        pos = random.choice(chinese_indices)
        original_char = tokens[pos]
        
        rn = random.random()
        if rn < 0.5 and original_char in self.pinyin_sim:
            # 拼音相近
            tokens[pos] = random.choice(self.pinyin_sim[original_char])
        elif rn < 0.8 and original_char in self.stroke:
            # 字形相似
            tokens[pos] = random.choice(self.stroke[original_char])
        elif self.word_freq:
            # 随机高频词
            tokens[pos] = random.choice(self.word_freq)
        # 如果都没命中，保持不变（无错样本）
        
        src = ''.join(tokens)
        trg = sentence
        return src, trg


def split_data(data: List, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42):
    """划分数据集"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * ratios[0])
    dev_end = train_end + int(n * ratios[1])
    
    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


def process_batch(args):
    """处理一批数据（用于多进程）"""
    lines, tokenizer_name, confusion_dir, max_seq_length, prompt_length, seed = args
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    error_gen = ErrorGenerator(confusion_dir, seed=seed)
    
    results = []
    for line in lines:
        sentence = line.strip()
        if not sentence:
            continue
        
        src, trg = error_gen.generate_error(sentence)
        # 格式: src_chars \t trg_chars (空格分隔字符)
        src_chars = list(src)
        trg_chars = list(trg)
        results.append((src_chars, trg_chars))
    
    return results


def preprocess_to_pt(
    pairs: List[Tuple[List[str], List[str]]],
    tokenizer,
    confusion_set,
    max_seq_length: int,
    prompt_length: int,
    num_workers: int = None,
    batch_size: int = None
) -> dict:
    """将src/trg对预处理为pt格式
    
    Args:
        batch_size: 分批处理大小，None表示一次性处理所有数据。
                    建议大数据集设置为10000-50000以避免OOM
    """
    from utils.dgca_data_processor import convert_examples_to_prompts, generate_error_labels
    
    if num_workers is None:
        num_workers = max(1, cpu_count() // 2)
    
    # 如果未指定batch_size或数据量较小，一次性处理
    if batch_size is None or len(pairs) <= batch_size:
        return _process_batch_to_dict(
            pairs, tokenizer, confusion_set, max_seq_length, prompt_length
        )
    
    # 大数据集分批处理
    print(f"数据量较大({len(pairs)}样本)，使用分批处理(batch_size={batch_size})")
    all_batches = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="分批预处理"):
        batch_pairs = pairs[i:i+batch_size]
        batch_dict = _process_batch_to_dict(
            batch_pairs, tokenizer, confusion_set, max_seq_length, prompt_length
        )
        all_batches.append(batch_dict)
    
    # 合并所有批次
    print("合并所有批次...")
    merged = {
        'input_ids': torch.cat([b['input_ids'] for b in all_batches], dim=0),
        'attention_mask': torch.cat([b['attention_mask'] for b in all_batches], dim=0),
        'labels': torch.cat([b['labels'] for b in all_batches], dim=0),
        'trg_ref_ids': torch.cat([b['trg_ref_ids'] for b in all_batches], dim=0),
        'block_flag': torch.cat([b['block_flag'] for b in all_batches], dim=0),
        'error_labels': torch.cat([b['error_labels'] for b in all_batches], dim=0),
        'candidate_ids': torch.cat([b['candidate_ids'] for b in all_batches], dim=0)
    }
    return merged


def _process_batch_to_dict(
    pairs: List[Tuple[List[str], List[str]]],
    tokenizer,
    confusion_set,
    max_seq_length: int,
    prompt_length: int
) -> dict:
    """处理一批数据并返回dict格式"""
    from utils.dgca_data_processor import convert_examples_to_prompts, generate_error_labels
    
    results = []
    for src, trg in pairs:
        try:
            # 截断
            half_len = max_seq_length // 2 - prompt_length
            src = src[:half_len]
            trg = trg[:half_len]
            if len(src) != len(trg):
                trg = trg[:len(src)]
            
            # Prompt转换
            prompt_src, prompt_trg, block_flag, trg_ref = convert_examples_to_prompts(
                src, trg, prompt_length, max_seq_length // 2, tokenizer, anchor=None
            )
            
            # Tokenize
            encoded_src = tokenizer(
                prompt_src, max_length=max_seq_length,
                padding="max_length", truncation=True, is_split_into_words=True
            )
            src_ids = encoded_src["input_ids"]
            attention_mask = encoded_src["attention_mask"]
            
            trg_ids = tokenizer(
                prompt_trg, max_length=max_seq_length,
                padding="max_length", truncation=True, is_split_into_words=True
            )["input_ids"]
            
            trg_ref_ids = tokenizer(
                trg_ref, max_length=max_seq_length,
                padding="max_length", truncation=True, is_split_into_words=True
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
            
            results.append({
                'src_ids': src_ids,
                'attention_mask': attention_mask,
                'trg_ids': trg_ids,
                'trg_ref_ids': trg_ref_ids,
                'block_flag': block_flag,
                'error_labels': error_labels,
                'candidate_ids': candidate_ids
            })
        except Exception as e:
            continue
    
    return {
        'input_ids': torch.tensor([r['src_ids'] for r in results], dtype=torch.long),
        'attention_mask': torch.tensor([r['attention_mask'] for r in results], dtype=torch.long),
        'labels': torch.tensor([r['trg_ids'] for r in results], dtype=torch.long),
        'trg_ref_ids': torch.tensor([r['trg_ref_ids'] for r in results], dtype=torch.long),
        'block_flag': torch.tensor([r['block_flag'] for r in results], dtype=torch.long),
        'error_labels': torch.tensor([r['error_labels'] for r in results], dtype=torch.long),
        'candidate_ids': torch.tensor([r['candidate_ids'] for r in results], dtype=torch.long)
    }


def main():
    parser = argparse.ArgumentParser(description="干净句子造错+预处理为.pt")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入文件（每行一个干净句子）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录，生成train.pt, dev.pt, test.pt")
    parser.add_argument("--model_path", type=str, default="bert-base-chinese",
                        help="预训练模型路径")
    parser.add_argument("--confusion_dir", type=str, default="confusion",
                        help="混淆集目录（包含pinyin_sim.json, stroke.json, word_freq.txt）")
    parser.add_argument("--dgca_config", type=str, default="config/default_config.yaml",
                        help="DGCA配置文件")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--prompt_length", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=str, default="0.8,0.1,0.1",
                        help="train,dev,test划分比例")
    parser.add_argument("--no_error_ratio", type=float, default=0.2,
                        help="无错样本比例，默认0.2（20%不造错，让模型学会不改）")
    parser.add_argument("--batch_size", type=int, default=50000,
                        help="分批处理大小，避免OOM。默认50000，可根据内存调整")
    
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
    
    print(f"加载混淆集（用于推理/训练候选集）...")
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
    print(f"开始造错（无错样本比例: {args.no_error_ratio*100:.0f}%）...")
    error_gen = ErrorGenerator(args.confusion_dir, seed=args.seed)
    pairs = []
    no_error_count = 0
    for sentence in tqdm(sentences, desc="造错"):
        src, trg = error_gen.generate_error(sentence, no_error_ratio=args.no_error_ratio)
        pairs.append((list(src), list(trg)))
        if src == trg:
            no_error_count += 1
    
    print(f"造错完成: 总样本 {len(pairs)}, 无错样本 {no_error_count} ({no_error_count/len(pairs)*100:.1f}%)")
    
    # 划分数据集
    print(f"划分数据集 (比例: {ratios})...")
    train_pairs, dev_pairs, test_pairs = split_data(pairs, ratios, args.seed)
    print(f"Train: {len(train_pairs)}, Dev: {len(dev_pairs)}, Test: {len(test_pairs)}")
    
    # 预处理并保存
    for name, data_pairs in [("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)]:
        print(f"\n预处理 {name}...")
        pt_data = preprocess_to_pt(
            data_pairs, tokenizer, confusion_set,
            args.max_seq_length, args.prompt_length, args.num_workers,
            batch_size=args.batch_size
        )
        output_path = os.path.join(args.output_dir, f"{name}.pt")
        torch.save(pt_data, output_path)
        print(f"保存: {output_path}, 样本数: {pt_data['input_ids'].shape[0]}")
    
    print("\n预处理完成！")


if __name__ == "__main__":
    main()
