"""
DDP双卡性能诊断脚本
用于分析WSL2环境下双4090训练的瓶颈

使用方法:
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 diagnose_ddp.py

如果只想运行通信测试:
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 diagnose_ddp.py --comm_only
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 测试配置
BATCH_SIZE = 64  # 每卡的batch size
SEQ_LEN = 128
NUM_ITERS = 20
WARMUP_ITERS = 5

# 设置超时时间（避免卡住）
NCCL_TIMEOUT = 120  # 秒


def setup():
    """初始化DDP"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    # 设置NCCL超时
    dist.init_process_group(
        backend='nccl',
        timeout=torch.distributed.distributed_c10d.timedelta(seconds=NCCL_TIMEOUT)
    )
    return local_rank


def cleanup():
    dist.destroy_process_group()


def print_rank0(msg, local_rank):
    if local_rank == 0:
        print(msg)


def measure_time(fn, warmup=5, iterations=20):
    """测量函数执行时间"""
    # Warmup
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        fn()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def test_nccl_bandwidth(local_rank):
    """测试1: NCCL通信带宽"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试1: NCCL AllReduce 通信带宽", local_rank)
    print_rank0("="*60, local_rank)
    
    # 测试不同大小的tensor
    sizes = [
        (1024 * 1024, "1MB"),       # 1M floats = 4MB
        (10 * 1024 * 1024, "10MB"), # 10M floats = 40MB
        (50 * 1024 * 1024, "50MB"), # 50M floats = 200MB (约等于BERT梯度大小)
    ]
    
    for size, name in sizes:
        tensor = torch.randn(size, device=f'cuda:{local_rank}')
        
        def allreduce():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        
        time_ms = measure_time(allreduce, warmup=3, iterations=10)
        bandwidth = (size * 4 * 2) / (time_ms / 1000) / 1e9  # GB/s (双向)
        
        print_rank0(f"  {name}: {time_ms:.2f}ms, 带宽: {bandwidth:.2f} GB/s", local_rank)
    
    dist.barrier()


def test_pure_bert_ddp(local_rank):
    """测试2: 纯BERT DDP训练（不含DGCA模块）- 使用本地模型避免下载"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0(f"测试2: 纯BERT DDP训练 (batch_size={BATCH_SIZE}/卡)", local_rank)
    print_rank0("="*60, local_rank)
    
    # 延迟导入，避免影响通信测试
    from transformers import BertForMaskedLM, BertConfig
    
    # 使用本地配置创建模型，避免网络下载
    try:
        model = BertForMaskedLM.from_pretrained('bert-base-chinese', local_files_only=True).cuda(local_rank)
    except:
        print_rank0("  无法加载bert-base-chinese，使用随机初始化的BERT模型", local_rank)
        config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12, 
                           num_attention_heads=12, intermediate_size=3072)
        model = BertForMaskedLM(config).cuda(local_rank)
    
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    # 创建假数据
    input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    def train_step():
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    time_ms = measure_time(train_step, warmup=WARMUP_ITERS, iterations=NUM_ITERS)
    throughput = BATCH_SIZE * dist.get_world_size() / (time_ms / 1000)
    
    print_rank0(f"  时间: {time_ms:.1f}ms/iter", local_rank)
    print_rank0(f"  吞吐: {throughput:.1f} samples/s (双卡总和)", local_rank)
    print_rank0(f"  GPU功率: 请查看nvidia-smi", local_rank)
    
    del model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()


def test_bert_ddp_bucket_size(local_rank):
    """测试3: 不同bucket_cap_mb设置的影响"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试3: DDP bucket_cap_mb参数影响", local_rank)
    print_rank0("="*60, local_rank)
    
    bucket_sizes = [25, 50, 100, 200]  # MB
    
    for bucket_mb in bucket_sizes:
        model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
        model = DDP(
            model, 
            device_ids=[local_rank],
            bucket_cap_mb=bucket_mb
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        def train_step():
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        time_ms = measure_time(train_step, warmup=3, iterations=10)
        print_rank0(f"  bucket_cap_mb={bucket_mb}: {time_ms:.1f}ms/iter", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_gradient_compression(local_rank):
    """测试4: 梯度累积对通信的影响"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试4: Gradient Accumulation效果", local_rank)
    print_rank0("="*60, local_rank)
    
    accum_steps_list = [1, 2, 4]
    
    for accum_steps in accum_steps_list:
        micro_batch = BATCH_SIZE // accum_steps
        
        model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        input_ids = torch.randint(100, 21000, (micro_batch, SEQ_LEN), device=f'cuda:{local_rank}')
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        def train_step_with_accum():
            optimizer.zero_grad()
            for i in range(accum_steps):
                with model.no_sync() if i < accum_steps - 1 else torch.enable_grad():
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / accum_steps
                    scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        time_ms = measure_time(train_step_with_accum, warmup=3, iterations=10)
        effective_batch = micro_batch * accum_steps * dist.get_world_size()
        throughput = effective_batch / (time_ms / 1000)
        
        print_rank0(f"  accum_steps={accum_steps}: {time_ms:.1f}ms/iter, 吞吐: {throughput:.0f} samples/s", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_find_unused_parameters(local_rank):
    """测试5: find_unused_parameters开销"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试5: find_unused_parameters开销", local_rank)
    print_rank0("="*60, local_rank)
    
    for find_unused in [False, True]:
        model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
        model = DDP(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=find_unused
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        def train_step():
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        time_ms = measure_time(train_step, warmup=3, iterations=10)
        print_rank0(f"  find_unused_parameters={find_unused}: {time_ms:.1f}ms/iter", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_static_graph(local_rank):
    """测试6: static_graph优化 (PyTorch 1.11+)"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试6: static_graph优化", local_rank)
    print_rank0("="*60, local_rank)
    
    for static_graph in [False, True]:
        model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
        model = DDP(
            model, 
            device_ids=[local_rank],
            static_graph=static_graph
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        def train_step():
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        time_ms = measure_time(train_step, warmup=WARMUP_ITERS, iterations=NUM_ITERS)
        print_rank0(f"  static_graph={static_graph}: {time_ms:.1f}ms/iter", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_different_batch_sizes(local_rank):
    """测试7: 不同batch size下的扩展效率"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试7: 不同batch size的扩展效率", local_rank)
    print_rank0("="*60, local_rank)
    
    batch_sizes = [32, 64, 96, 128]
    
    for bs in batch_sizes:
        try:
            model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
            model = DDP(model, device_ids=[local_rank])
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scaler = torch.amp.GradScaler('cuda')
            
            input_ids = torch.randint(100, 21000, (bs, SEQ_LEN), device=f'cuda:{local_rank}')
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            
            def train_step():
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            time_ms = measure_time(train_step, warmup=3, iterations=10)
            total_samples = bs * dist.get_world_size()
            throughput = total_samples / (time_ms / 1000)
            
            # 获取显存使用
            mem_used = torch.cuda.max_memory_allocated(local_rank) / 1024**3
            
            print_rank0(f"  batch_size={bs}/卡: {time_ms:.1f}ms, 吞吐: {throughput:.0f} samples/s, 显存: {mem_used:.1f}GB", local_rank)
            
            del model, optimizer
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(local_rank)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print_rank0(f"  batch_size={bs}/卡: OOM!", local_rank)
                torch.cuda.empty_cache()
            else:
                raise
    
    dist.barrier()


def test_compute_comm_overlap(local_rank):
    """测试8: 计算与通信重叠度分析"""
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("测试8: 计算与通信重叠分析", local_rank)
    print_rank0("="*60, local_rank)
    
    model = BertForMaskedLM.from_pretrained('bert-base-chinese').cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # 测量纯前向传播时间
    def forward_only():
        with torch.amp.autocast('cuda'):
            outputs = model.module(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs.loss
    
    forward_time = measure_time(forward_only, warmup=3, iterations=10)
    
    # 测量前向+反向（不含DDP通信）
    def forward_backward_no_sync():
        optimizer.zero_grad()
        with model.no_sync():
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            loss.backward()
    
    fb_no_sync_time = measure_time(forward_backward_no_sync, warmup=3, iterations=10)
    
    # 测量完整训练步骤（含DDP通信）
    def full_train_step():
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    full_time = measure_time(full_train_step, warmup=3, iterations=10)
    
    backward_time = fb_no_sync_time - forward_time
    comm_overhead = full_time - fb_no_sync_time
    
    print_rank0(f"  前向传播: {forward_time:.1f}ms", local_rank)
    print_rank0(f"  反向传播(无通信): {backward_time:.1f}ms", local_rank)
    print_rank0(f"  DDP通信开销: {comm_overhead:.1f}ms", local_rank)
    print_rank0(f"  通信占比: {comm_overhead/full_time*100:.1f}%", local_rank)
    
    del model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    local_rank = setup()
    
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("DGCA-ReLM DDP性能诊断", local_rank)
    print_rank0(f"CUDA: {torch.version.cuda}", local_rank)
    print_rank0(f"PyTorch: {torch.__version__}", local_rank)
    print_rank0(f"World Size: {dist.get_world_size()}", local_rank)
    print_rank0(f"NCCL Version: {torch.cuda.nccl.version()}", local_rank)
    print_rank0("="*60, local_rank)
    
    # 运行所有测试
    test_nccl_bandwidth(local_rank)
    test_pure_bert_ddp(local_rank)
    test_bert_ddp_bucket_size(local_rank)
    test_gradient_compression(local_rank)
    test_find_unused_parameters(local_rank)
    test_static_graph(local_rank)
    test_different_batch_sizes(local_rank)
    test_compute_comm_overlap(local_rank)
    
    print_rank0("\n" + "="*60, local_rank)
    print_rank0("诊断完成！", local_rank)
    print_rank0("="*60, local_rank)
    
    cleanup()


if __name__ == "__main__":
    main()
