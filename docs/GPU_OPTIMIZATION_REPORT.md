# GPU性能优化报告

**日期**: 2025年12月21日  
**项目**: DGCA-ReLM  
**问题**: 训练时GPU利用率低，功率只有200W（满载450W）

---

## 📊 诊断测试数据

使用 `diagnose_gpu.py` 进行分步测试（batch_size=128, seq_len=128）：

| 测试项 | 速度 | 吞吐量 | GPU功率 | 说明 |
|--------|------|--------|---------|------|
| 测试1: 纯BERT推理 | 111.6ms/iter | 1147 samples/s | **400W+** | GPU满载 |
| 测试2: BERT训练 | 335.3ms/iter | 381.7 samples/s | **400W+** | GPU满载 |
| 测试3: BERT+FP16 | 162.9ms/iter | 785.5 samples/s | 350W | FP16减少计算 |
| 测试4: DataLoader | 12.4ms/batch | - | N/A | 数据加载很快 |
| 测试5: 真实数据+BERT | 186.4ms/iter | 686.7 samples/s | 350W | 略有下降 |
| 测试6: 完整DGCA模型 | 579.4ms/iter | 220.9 samples/s | **200W** | ⚠️ GPU等待 |

**关键发现**: 测试6(DGCA)比测试3(BERT+FP16)慢了3.5倍，功率低了150W

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

## 📈 优化效果

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 训练速度 | 2.2 it/s | **4.9 it/s** | **2.2x** |
| GPU功率 | 200W | **320W** | +60% |
| 利用率估算 | ~44% | ~71% | +27% |

---

## 🔮 待继续优化的问题

### 1. 双卡DDP训练反而更慢

**现象**: 
- 单卡: 2.2 it/s (优化前) / 4.9 it/s (优化后)
- 双卡: 1.5 it/s，每卡功率只有100W

**可能原因**:
- NCCL通信开销
- 梯度同步瓶颈
- batch_size=128在双卡下每卡只有64，计算效率下降

**待测试**:
- 双卡时增大batch_size到256或更高
- 检查gradient_accumulation_steps设置
- 使用`torch.distributed.barrier()`定位同步开销

### 2. 功率仍未满载（320W vs 450W）

**可能原因**:
- DetectorHead、CandidateHead额外计算开销
- 混合精度下某些操作fallback到FP32
- 内存带宽瓶颈（candidate_embeddings lookup）

**待分析**:
- 使用`torch.profiler`详细分析各操作耗时
- 检查是否有CUDA同步点导致的等待

### 3. diagnose_gpu.py 测试6 待更新

需要更新测试6使用优化后的代码重新测试基准。

---

## 📝 配置建议

### 当前推荐配置（单卡4090）

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

### 待验证配置（双卡）

```bash
torchrun --nproc_per_node=2 --master_port=29500 run_dgca_relm.py \
    --do_train --do_eval --do_test \
    --preprocessed_train data/train.pt \
    --preprocessed_eval data/dev.pt \
    --preprocessed_test data/test.pt \
    --fp16 --apply_prompt --mft \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --num_workers 4
```

---

## 🛠️ 下次继续的方向

1. **双卡训练优化**: 分析DDP通信开销，尝试gradient_accumulation
2. **进一步提升单卡利用率**: 使用torch.profiler找出剩余瓶颈
3. **编译优化**: 尝试`torch.compile()`（PyTorch 2.0+）
4. **更大batch_size**: 测试batch_size=192/256的效果
