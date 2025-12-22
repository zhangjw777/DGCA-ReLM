# GPU性能优化报告

**日期**: 2025年12月22日 (更新)  
**项目**: DGCA-ReLM  
**问题**: 训练时GPU利用率低，双卡DDP性能瓶颈

---

## 📊 WSL2双卡环境诊断

### 硬件连接情况
```
GPU0    GPU1    连接方式
 X      SYS     跨NUMA PCIe连接（无NVLink）
```

### NCCL通信带宽测试
| 数据量 | 延迟 | 带宽 |
|--------|------|------|
| 4MB | 9ms | 0.92 GB/s |
| 40MB | 83ms | 1.01 GB/s |
| 200MB | 390ms | 1.07 GB/s |

⚠️ **关键发现**: WSL2 + NCCL_SHM_DISABLE导致NCCL带宽只有**1 GB/s**，正常PCIe应有20+ GB/s

---

## 📊 单卡诊断测试数据

使用 `diagnose_gpu.py` 进行分步测试（batch_size=128, seq_len=128）：

| 测试项 | 速度 | 吞吐量 | GPU功率 | 说明 |
|--------|------|--------|---------|------|
| 测试1: 纯BERT推理 | 111.6ms/iter | 1147 samples/s | **400W+** | GPU满载 |
| 测试2: BERT训练 | 335.3ms/iter | 381.7 samples/s | **400W+** | GPU满载 |
| 测试3: BERT+FP16 | 162.9ms/iter | 785.5 samples/s | 350W | FP16减少计算 |
| 测试4: DataLoader | 12.4ms/batch | - | N/A | 数据加载很快 |
| 测试5: 真实数据+BERT | 186.4ms/iter | 686.7 samples/s | 350W | 略有下降 |
| 测试6: 完整DGCA模型 | 579.4ms/iter | 220.9 samples/s | **200W** | ⚠️ GPU等待 |

---

## 🔍 瓶颈分析

### 1. 已排除的因素

- ❌ **数据加载瓶颈**: 测试4显示DataLoader速度很快(12.4ms/batch)
- ❌ **mmap I/O问题**: 使用`--preload_data`预加载到内存后速度无变化
- ❌ **num_workers竞争**: num_workers=2/4/8速度相同

### 2. 找到的真正瓶颈

#### ⚠️ 核心问题: `_apply_prompt`中的双重Python for循环

```python
# 原始代码 - 每次forward执行768次Python循环！
for i in range(batch_size):           # 128次
    for j in range(2 * self.prompt_length):  # 6次
        inputs_embeds[i, blocked_indices[i, j], :] = replace_embeds[j, :]
```

**问题根源**: 
- 每次循环都是一次CUDA kernel launch
- Python循环开销 + CUDA kernel启动开销叠加
- batch_size=128时，每个forward要768次kernel launch

---

## ✅ 已完成的优化

### 优化1: `_apply_prompt` 向量化（最关键！）

**文件**: `multiTask/DGCAModel.py`

```python
# 优化后 - 单次向量化操作
prompt_positions = (prompt_mask[0] == 1).nonzero(as_tuple=True)[0]
replace_embeds_expanded = replace_embeds.unsqueeze(0).expand(batch_size, -1, -1)
inputs_embeds.index_copy_(1, prompt_positions, replace_embeds_expanded)
```

### 优化2: `dynamic_mask_token` GPU化

**文件**: `run_dgca_relm.py`

- 移除`tokenizer.get_special_tokens_mask()`的CPU调用
- 全部使用GPU上的向量化操作

### 优化3: `GatedFusion` 内存优化

**文件**: `multiTask/DGCAModel.py`

- 使用`scatter_add_`原地操作替代创建vocab_size大小的零张量
- 减少约1.3GB临时内存分配

### 优化4: `PreprocessedDataset` 优化

**文件**: `utils/dgca_data_processor.py`

- `clone()` -> `contiguous()`减少不必要拷贝
- 新增`preload_to_memory`选项

### 优化5: DataLoader配置

**文件**: `run_dgca_relm.py`

- 新增`--preload_data`参数
- 新增`--prefetch_factor`参数
- 添加`drop_last=True`

---

## 📈 单卡优化效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 训练速度 | 2.2 it/s | **4.9 it/s** | **2.2x** |
| GPU功率 | 200W | **320W** | +60% |
| 利用率估算 | ~44% | ~71% | +27% |

---

## 🔧 双卡DDP优化（2025-12-22新增）

### 问题诊断

**现象**: 
- 单卡: 4.9 it/s，功率320W
- 双卡: 1.4 it/s，每卡功率只有100W

**根本原因**: WSL2环境下NCCL带宽极低（1 GB/s vs 正常20+ GB/s）
- `NCCL_SHM_DISABLE=1` 禁用共享内存
- `NCCL_IB_DISABLE=1` 禁用InfiniBand
- 两张4090通过跨NUMA PCIe连接（SYS），无NVLink

### 已实施的DDP优化

#### 优化6: `no_sync()` 梯度累积优化（最关键！）

**文件**: `run_dgca_relm.py`

在梯度累积的中间步骤跳过AllReduce通信：

```python
# 在累积步骤跳过梯度同步
if args.local_rank != -1 and is_accumulation_step and not is_last_step:
    sync_context = model.no_sync()
else:
    sync_context = torch.enable_grad()

with sync_context:
    scaler.scale(loss).backward()
```

**效果**: 使用`--gradient_accumulation_steps 4`时，通信频率降低4倍

#### 优化7: DDP配置优化

**文件**: `run_dgca_relm.py`

```python
model = DDP(model, 
    device_ids=[local_rank],
    bucket_cap_mb=args.ddp_bucket_cap_mb,  # 默认100MB
    gradient_as_bucket_view=True,  # 减少内存拷贝
    static_graph=args.ddp_static_graph  # 可选
)
```

新增命令行参数：
- `--ddp_bucket_cap_mb`: DDP bucket大小，低带宽环境建议200
- `--ddp_static_graph`: 启用static_graph优化

#### 优化8: 评估频率优化

**文件**: `run_dgca_relm.py`

- `--save_steps` 默认值从500改为1000
- 新增 `--eval_steps` 参数，可独立控制评估频率

---

## 📝 双卡推荐配置

### WSL2低带宽环境推荐

```bash
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 \
torchrun --nproc_per_node=2 --master_port=29500 run_dgca_relm.py \
    --do_train --do_eval --do_test \
    --preprocessed_train data/train.pt \
    --preprocessed_eval data/dev.pt \
    --preprocessed_test data/test.pt \
    --fp16 --apply_prompt --mft \
    --train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --ddp_bucket_cap_mb 200 \
    --eval_steps 2000 \
    --num_workers 2
```

**关键参数说明**:
- `--gradient_accumulation_steps 4`: 每4步通信1次，减少75%通信开销
- `--ddp_bucket_cap_mb 200`: 更大的bucket减少AllReduce次数
- `--eval_steps 2000`: 减少评估频率，减少阻塞
- `--num_workers 2`: 双卡时减少worker数量避免竞争

### 单卡推荐配置

```bash
CUDA_VISIBLE_DEVICES=1 python run_dgca_relm.py \
    --do_train --do_eval --do_test \
    --preprocessed_train data/train.pt \
    --preprocessed_eval data/dev.pt \
    --preprocessed_test data/test.pt \
    --fp16 --apply_prompt --mft \
    --train_batch_size 128 \
    --num_workers 4 \
    --prefetch_factor 2
```

---

## 🔮 待继续优化的问题

### 1. 双卡仍需验证优化效果

运行诊断脚本验证优化：
```bash
NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 diagnose_ddp_v2.py
```

### 2. 功率仍未满载（320W vs 450W）

**可能原因**:
- DetectorHead、CandidateHead额外计算开销
- 混合精度下某些操作fallback到FP32
- 内存带宽瓶颈

### 3. batch_size > 128 OOM问题

双卡模式下batch_size超过128就OOM，但单卡可以跑128。可能原因：
- DDP需要额外显存存储梯度bucket
- `gradient_as_bucket_view=True`可能有帮助

---

## 🛠️ 下次继续的方向

1. **验证双卡优化效果**: 运行新的推荐配置测试速度提升
2. **进一步提升单卡利用率**: 使用torch.profiler找出剩余瓶颈
3. **编译优化**: 尝试`torch.compile()`（PyTorch 2.0+）
4. **尝试其他分布式策略**: 如FSDP可能比DDP更适合低带宽环境
