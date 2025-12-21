"""
GPU性能诊断脚本
逐步测试各个环节，找出瓶颈所在

使用方法：
    python diagnose_gpu.py --test 1   # 测试1：纯BERT推理
    python diagnose_gpu.py --test 2   # 测试2：BERT + 数据加载
    python diagnose_gpu.py --test 3   # 测试3：BERT + 数据加载 + 混合精度
    python diagnose_gpu.py --test 4   # 测试4：完整DGCA模型
    python diagnose_gpu.py --test 5   # 测试5：完整训练循环
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForMaskedLM, AutoTokenizer
from tqdm import tqdm


def test_pure_bert_inference(device, batch_size=64, seq_len=128, num_iters=100):
    """测试1：纯BERT模型推理性能"""
    print("\n" + "="*60)
    print("测试1：纯BERT模型推理（无数据加载开销）")
    print("="*60)
    
    model = BertForMaskedLM.from_pretrained(
        "bert-base-chinese",
        return_dict=True
    ).to(device)
    model.eval()
    
    # 预先生成假数据
    input_ids = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    torch.cuda.synchronize()
    
    # 正式测试
    start = time.time()
    for _ in tqdm(range(num_iters), desc="纯推理"):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/num_iters*1000:.1f}ms/iter")
    print(f"吞吐量: {num_iters * batch_size / elapsed:.1f} samples/s")
    
    del model
    torch.cuda.empty_cache()


def test_bert_with_training(device, batch_size=64, seq_len=128, num_iters=100):
    """测试2：BERT训练（含反向传播）"""
    print("\n" + "="*60)
    print("测试2：BERT训练（含反向传播，无数据加载）")
    print("="*60)
    
    model = BertForMaskedLM.from_pretrained(
        "bert-base-chinese",
        return_dict=True
    ).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 预先生成假数据
    input_ids = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    
    # 预热
    for _ in range(5):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # 正式测试
    start = time.time()
    for _ in tqdm(range(num_iters), desc="训练"):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/num_iters*1000:.1f}ms/iter")
    print(f"吞吐量: {num_iters * batch_size / elapsed:.1f} samples/s")
    
    del model, optimizer
    torch.cuda.empty_cache()


def test_bert_with_fp16(device, batch_size=64, seq_len=128, num_iters=100):
    """测试3：BERT训练 + FP16混合精度"""
    print("\n" + "="*60)
    print("测试3：BERT训练 + FP16混合精度")
    print("="*60)
    
    model = BertForMaskedLM.from_pretrained(
        "bert-base-chinese",
        return_dict=True
    ).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # 预先生成假数据
    input_ids = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(1, 21128, (batch_size, seq_len), device=device)
    
    # 预热
    for _ in range(5):
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # 正式测试
    start = time.time()
    for _ in tqdm(range(num_iters), desc="FP16训练"):
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/num_iters*1000:.1f}ms/iter")
    print(f"吞吐量: {num_iters * batch_size / elapsed:.1f} samples/s")
    
    del model, optimizer, scaler
    torch.cuda.empty_cache()


def test_dataloader_speed(batch_size=64, num_workers=4, num_iters=100):
    """测试4：DataLoader加载速度"""
    print("\n" + "="*60)
    print(f"测试4：DataLoader速度测试 (num_workers={num_workers})")
    print("="*60)
    
    # 创建假数据集
    data_size = 10000
    seq_len = 128
    
    dataset = TensorDataset(
        torch.randint(1, 21128, (data_size, seq_len)),
        torch.ones(data_size, seq_len, dtype=torch.long),
        torch.randint(1, 21128, (data_size, seq_len)),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 预热
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # 正式测试
    start = time.time()
    count = 0
    for i, batch in enumerate(tqdm(dataloader, desc="数据加载", total=min(num_iters, len(dataloader)))):
        count += 1
        if count >= num_iters:
            break
    
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/count*1000:.1f}ms/batch")


def test_real_dataset(preprocessed_file, device, batch_size=64, num_workers=4, num_iters=100):
    """测试5：真实数据集加载 + BERT训练"""
    print("\n" + "="*60)
    print(f"测试5：真实数据集 + BERT训练")
    print("="*60)
    
    from utils.dgca_data_processor import PreprocessedDataset
    
    print("加载数据集...")
    load_start = time.time()
    dataset = PreprocessedDataset(preprocessed_file)
    print(f"数据集加载时间: {time.time() - load_start:.2f}s")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    model = BertForMaskedLM.from_pretrained(
        "bert-base-chinese",
        return_dict=True
    ).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # 正式测试
    start = time.time()
    count = 0
    
    for batch in tqdm(dataloader, desc="训练", total=min(num_iters, len(dataloader))):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        scaler.scale(outputs.loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        count += 1
        if count >= num_iters:
            break
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/count*1000:.1f}ms/iter")
    print(f"吞吐量: {count * batch_size / elapsed:.1f} samples/s")


def test_dgca_model(preprocessed_file, device, batch_size=64, num_workers=4, num_iters=100):
    """测试6：完整DGCA模型"""
    print("\n" + "="*60)
    print(f"测试6：完整DGCA模型训练")
    print("="*60)
    
    from utils.dgca_data_processor import PreprocessedDataset
    from config.dgca_config import DGCAConfig
    from confusion.confusion_utils import ConfusionSet
    from multiTask.DGCAModel import DGCAReLMWrapper
    
    # 加载配置
    dgca_config = DGCAConfig.from_yaml("config/default_config.yaml")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 加载混淆集
    confusion_set = ConfusionSet(
        confusion_dir=dgca_config.confusion_dir,
        confusion_file=dgca_config.confusion_file,
        tokenizer=tokenizer,
        candidate_size=dgca_config.candidate_size,
        include_original=dgca_config.include_original_char
    )
    
    # 加载数据
    print("加载数据集...")
    load_start = time.time()
    dataset = PreprocessedDataset(preprocessed_file)
    print(f"数据集加载时间: {time.time() - load_start:.2f}s")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 创建模型
    bert_model = BertForMaskedLM.from_pretrained("bert-base-chinese", return_dict=True)
    model = DGCAReLMWrapper(
        bert_model=bert_model,
        confusion_set=confusion_set,
        config=dgca_config,
        prompt_length=3
    ).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # 正式测试
    start = time.time()
    count = 0
    
    for batch in tqdm(dataloader, desc="DGCA训练", total=min(num_iters, len(dataloader))):
        src_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        trg_ids = batch['labels'].to(device)
        block_flag = batch['block_flag'].to(device)
        error_labels = batch['error_labels'].to(device)
        candidate_ids = batch['candidate_ids'].to(device)
        
        labels = trg_ids.clone()
        labels[src_ids == trg_ids] = -100
        
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=src_ids,
                attention_mask=attention_mask,
                prompt_mask=block_flag,
                labels=labels,
                candidate_ids=candidate_ids,
                error_labels=error_labels,
                apply_prompt=True
            )
        
        scaler.scale(outputs['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        count += 1
        if count >= num_iters:
            break
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print(f"总时间: {elapsed:.2f}s, 平均: {elapsed/count*1000:.1f}ms/iter")
    print(f"吞吐量: {count * batch_size / elapsed:.1f} samples/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, required=True, 
                        help="测试编号：1-6")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--data_file", type=str, default="data/train.pt",
                        help="预处理数据文件路径")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if args.test == 1:
        test_pure_bert_inference(device, args.batch_size, num_iters=args.num_iters)
    elif args.test == 2:
        test_bert_with_training(device, args.batch_size, num_iters=args.num_iters)
    elif args.test == 3:
        test_bert_with_fp16(device, args.batch_size, num_iters=args.num_iters)
    elif args.test == 4:
        test_dataloader_speed(args.batch_size, args.num_workers, args.num_iters)
    elif args.test == 5:
        test_real_dataset(args.data_file, device, args.batch_size, args.num_workers, args.num_iters)
    elif args.test == 6:
        test_dgca_model(args.data_file, device, args.batch_size, args.num_workers, args.num_iters)
    else:
        print("未知的测试编号！请使用 1-6")
        print("1: 纯BERT推理")
        print("2: BERT训练（含反向传播）")
        print("3: BERT训练 + FP16")
        print("4: DataLoader速度测试")
        print("5: 真实数据集 + BERT训练")
        print("6: 完整DGCA模型训练")


if __name__ == "__main__":
    main()
