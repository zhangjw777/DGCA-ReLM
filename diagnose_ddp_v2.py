"""
DDP双卡性能诊断脚本 v2
针对WSL2环境下低NCCL带宽场景的优化测试

使用方法:
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 diagnose_ddp_v2.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

# 配置
BATCH_SIZE = 64
SEQ_LEN = 128


def setup():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # 设置较长的超时时间
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=5))
    return local_rank


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, local_rank):
    if local_rank == 0:
        print(msg, flush=True)


def measure_time(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000


def test_nccl_bandwidth(local_rank):
    """测试NCCL通信带宽"""
    print_rank0("\n[测试1] NCCL AllReduce带宽", local_rank)
    
    sizes = [
        (1024 * 1024, "4MB"),
        (10 * 1024 * 1024, "40MB"),
        (50 * 1024 * 1024, "200MB"),
    ]
    
    for num_floats, name in sizes:
        tensor = torch.randn(num_floats, device=f'cuda:{local_rank}')
        
        def allreduce():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
        
        time_ms = measure_time(allreduce, warmup=2, iterations=5)
        bandwidth = (num_floats * 4 * 2) / (time_ms / 1000) / 1e9
        print_rank0(f"  {name}: {time_ms:.1f}ms, 带宽: {bandwidth:.2f} GB/s", local_rank)
        del tensor
    
    torch.cuda.empty_cache()
    dist.barrier()


def test_gradient_accumulation(local_rank):
    """测试梯度累积策略"""
    print_rank0("\n[测试2] 梯度累积效果 (关键优化!)", local_rank)
    
    from transformers import BertConfig, BertForMaskedLM
    
    config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072)
    
    # 测试不同的梯度累积步数
    accum_configs = [
        (64, 1, "64x1 (每步通信)"),
        (32, 2, "32x2 (2步累积)"),
        (16, 4, "16x4 (4步累积)"),
    ]
    
    for micro_bs, accum_steps, desc in accum_configs:
        model = BertForMaskedLM(config).cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scaler = torch.amp.GradScaler('cuda')
        
        input_ids = torch.randint(100, 21000, (micro_bs, SEQ_LEN), device=f'cuda:{local_rank}')
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        def train_step():
            optimizer.zero_grad()
            for i in range(accum_steps):
                # 使用no_sync()跳过中间步骤的梯度同步
                ctx = model.no_sync() if i < accum_steps - 1 else torch.enable_grad()
                with ctx:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss / accum_steps
                    scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        time_ms = measure_time(train_step, warmup=2, iterations=5)
        effective_bs = micro_bs * accum_steps * dist.get_world_size()
        throughput = effective_bs / (time_ms / 1000)
        
        print_rank0(f"  {desc}: {time_ms:.0f}ms, 吞吐: {throughput:.0f} samples/s", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_ddp_options(local_rank):
    """测试DDP配置选项"""
    print_rank0("\n[测试3] DDP配置选项", local_rank)
    
    from transformers import BertConfig, BertForMaskedLM
    
    config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072)
    
    options = [
        {"bucket_cap_mb": 25, "desc": "bucket=25MB"},
        {"bucket_cap_mb": 100, "desc": "bucket=100MB"},
        {"bucket_cap_mb": 200, "desc": "bucket=200MB"},
        {"bucket_cap_mb": 100, "gradient_as_bucket_view": True, "desc": "bucket=100MB+bucket_view"},
        {"bucket_cap_mb": 100, "static_graph": True, "desc": "bucket=100MB+static_graph"},
    ]
    
    for opt in options:
        desc = opt.pop("desc")
        model = BertForMaskedLM(config).cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], **opt)
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
        
        time_ms = measure_time(train_step, warmup=2, iterations=5)
        print_rank0(f"  {desc}: {time_ms:.0f}ms", local_rank)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    dist.barrier()


def test_comm_overlap_analysis(local_rank):
    """分析通信开销占比"""
    print_rank0("\n[测试4] 通信开销分析", local_rank)
    
    from transformers import BertConfig, BertForMaskedLM
    
    config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072)
    
    model = BertForMaskedLM(config).cuda(local_rank)
    model_ddp = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # 纯前向
    def forward_only():
        with torch.amp.autocast('cuda'):
            model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    # 前向+反向，无通信
    def fb_no_comm():
        optimizer.zero_grad()
        with model_ddp.no_sync():
            with torch.amp.autocast('cuda'):
                outputs = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
    
    # 完整训练步骤
    def full_step():
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    t_forward = measure_time(forward_only, warmup=2, iterations=5)
    t_fb_no_comm = measure_time(fb_no_comm, warmup=2, iterations=5)
    t_full = measure_time(full_step, warmup=2, iterations=5)
    
    t_backward = t_fb_no_comm - t_forward
    t_comm = t_full - t_fb_no_comm
    
    print_rank0(f"  前向: {t_forward:.0f}ms", local_rank)
    print_rank0(f"  反向: {t_backward:.0f}ms", local_rank)
    print_rank0(f"  通信+优化器: {t_comm:.0f}ms", local_rank)
    print_rank0(f"  通信占比: {t_comm/t_full*100:.0f}%", local_rank)
    
    if t_comm/t_full > 0.3:
        print_rank0("\n  ⚠️ 通信开销超过30%，建议使用梯度累积减少通信频率！", local_rank)
    
    del model, model_ddp, optimizer
    torch.cuda.empty_cache()
    dist.barrier()


def test_single_vs_ddp(local_rank):
    """对比单卡和双卡的扩展效率"""
    print_rank0("\n[测试5] 单卡/双卡效率对比", local_rank)
    
    from transformers import BertConfig, BertForMaskedLM
    
    config = BertConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                       num_attention_heads=12, intermediate_size=3072)
    
    # 单机单卡（不使用DDP）
    model_single = BertForMaskedLM(config).cuda(local_rank)
    optimizer_single = torch.optim.AdamW(model_single.parameters(), lr=5e-5)
    scaler_single = torch.amp.GradScaler('cuda')
    
    input_ids = torch.randint(100, 21000, (BATCH_SIZE, SEQ_LEN), device=f'cuda:{local_rank}')
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    def single_step():
        optimizer_single.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model_single(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        scaler_single.scale(outputs.loss).backward()
        scaler_single.step(optimizer_single)
        scaler_single.update()
    
    t_single = measure_time(single_step, warmup=2, iterations=5)
    
    del model_single, optimizer_single
    torch.cuda.empty_cache()
    
    # DDP双卡
    model_ddp = BertForMaskedLM(config).cuda(local_rank)
    model_ddp = DDP(model_ddp, device_ids=[local_rank])
    optimizer_ddp = torch.optim.AdamW(model_ddp.parameters(), lr=5e-5)
    scaler_ddp = torch.amp.GradScaler('cuda')
    
    def ddp_step():
        optimizer_ddp.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model_ddp(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        scaler_ddp.scale(outputs.loss).backward()
        scaler_ddp.step(optimizer_ddp)
        scaler_ddp.update()
    
    t_ddp = measure_time(ddp_step, warmup=2, iterations=5)
    
    # 计算扩展效率
    # 理想情况：双卡时间 = 单卡时间 / 2，吞吐翻倍
    single_throughput = BATCH_SIZE / (t_single / 1000)
    ddp_throughput = BATCH_SIZE * 2 / (t_ddp / 1000)  # 双卡总batch
    scaling_efficiency = ddp_throughput / (single_throughput * 2) * 100
    
    print_rank0(f"  单卡: {t_single:.0f}ms, {single_throughput:.0f} samples/s", local_rank)
    print_rank0(f"  双卡: {t_ddp:.0f}ms, {ddp_throughput:.0f} samples/s (总)", local_rank)
    print_rank0(f"  扩展效率: {scaling_efficiency:.0f}%", local_rank)
    
    if scaling_efficiency < 70:
        print_rank0("\n  ⚠️ 扩展效率低于70%，通信是主要瓶颈！", local_rank)
        print_rank0("  建议: 使用梯度累积(--gradient_accumulation_steps 4)", local_rank)
    
    del model_ddp, optimizer_ddp
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    local_rank = setup()
    
    print_rank0("="*60, local_rank)
    print_rank0("DGCA-ReLM DDP性能诊断 v2", local_rank)
    print_rank0(f"PyTorch: {torch.__version__}", local_rank)
    print_rank0(f"CUDA: {torch.version.cuda}", local_rank)
    print_rank0(f"World Size: {dist.get_world_size()}", local_rank)
    print_rank0("="*60, local_rank)
    
    try:
        test_nccl_bandwidth(local_rank)
        test_comm_overlap_analysis(local_rank)
        test_gradient_accumulation(local_rank)
        test_ddp_options(local_rank)
        test_single_vs_ddp(local_rank)
        
        print_rank0("\n" + "="*60, local_rank)
        print_rank0("诊断完成！", local_rank)
        print_rank0("="*60, local_rank)
        
    except Exception as e:
        print(f"Rank {local_rank} 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
