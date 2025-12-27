# DGCA-ReLM 训练流程详解

## 1. 数据格式详解

### 1.1 原始数据

假设原始拼写纠错数据为：
- **src（源句/错误句）**: "我好高心"
- **trg（目标句/正确句）**: "我好高兴"

### 1.2 预处理后的数据字段

每个训练样本包含以下7个字段：

| 字段名 | 形状 | 示例（prompt_length=3） | 说明 |
|--------|------|-------------------------|------|
| `input_ids` | (seq_len,) | `[CLS,CLS,CLS, 我,好,高,心, SEP,SEP,SEP, MASK,MASK,MASK,MASK]` | 模型输入 |
| `attention_mask` | (seq_len,) | `[1,1,1, 1,1,1,1, 1,1,1, 1,1,1,1]` | 注意力掩码 |
| `labels` | (seq_len,) | `[CLS,CLS,CLS, 我,好,高,心, SEP,SEP,SEP, 我,好,高,兴]` | 目标输出（纠错标签） |
| `trg_ref_ids` | (seq_len,) | `[CLS,CLS,CLS, 我,好,高,兴, SEP,SEP,SEP, 我,好,高,心]` | 参考序列（特殊用途） |
| `block_flag` | (seq_len,) | `[1,1,1, 0,0,0,0, 1,1,1, 0,0,0,0]` | Prompt位置标记 |
| `error_labels` | (seq_len,) | `[-100,-100,-100, 0,0,0,1, -100,-100,-100, -100,-100,-100,-100]` | 检测标签 |
| `candidate_ids` | (seq_len, K) | 每个位置的K个候选字符ID | 混淆候选集 |

### 1.3 字段详细说明

#### 1.3.1 `input_ids`（模型输入）

**格式**: `[CLS]*P + src + [SEP]*P + [MASK]*n`

```
位置:     |  Prompt前  |   源句区域   |  Prompt后  |   目标区域（Mask）   |
内容:     | CLS CLS CLS | 我 好 高 心 | SEP SEP SEP | MASK MASK MASK MASK |
索引:     |   0  1  2   |  3  4  5  6 |   7  8  9   |  10   11   12   13  |
```

**设计理由**:
- ReLM 的核心思想是将 CSC 任务转化为"重写式语言建模"
- 源句放在前面提供上下文，MASK 区域让模型预测正确字符
- 这避免了传统 tagging 方法容易学到的"字到字映射捷径"

#### 1.3.2 `labels`（纠错标签）

**格式**: `[CLS]*P + src + [SEP]*P + trg`

```
位置:     |  Prompt前  |   源句区域   |  Prompt后  |   目标区域        |
内容:     | CLS CLS CLS | 我 好 高 心 | SEP SEP SEP | 我 好 高 兴      |
训练:     |   ignore   |   ignore    |   ignore   | 计算纠错loss     |
```

**训练时的处理**:
```python
labels = trg_ids.clone()
labels[src_ids == trg_ids] = -100  # ignore index
```

在 DGCA-ReLM 中：
- 源句区域 `src == trg`（因为 `labels` 源句侧存的是 src），全部被 ignore
- 目标区域计算纠错 loss：只有 `MASK != trg` 的位置（即错误位）有梯度

**为什么 DGCA 源句侧不参与纠错 loss？**
- DGCA 设计了专门的检测分支处理源句侧
- 源句侧用于 DetectorHead 训练，目标侧用于 CandidateHead 训练
- 这种分工让模型学习更清晰

#### 1.3.3 `trg_ref_ids`（参考序列）

**格式**: `[CLS]*P + trg + [SEP]*P + src`

```
位置:     |  Prompt前  |   源句区域   |  Prompt后  |   目标区域     |
内容:     | CLS CLS CLS | 我 好 高 兴 | SEP SEP SEP | 我 好 高 心   |
```

这是一个"镜像"序列，有三个关键用途：

1. **源句侧用于生成 `error_labels`**:
   - 比较 `input_ids` 源句侧（src）与 `trg_ref_ids` 源句侧（trg）
   - `input_ids[i] != trg_ref_ids[i]` → 错误位（error_label=1）
   - 用于训练 DetectorHead

2. **目标侧用于纠错位加权**:
   - 比较 `trg_ref_ids` 目标侧（src）与 `labels` 目标侧（trg）
   - `trg_ref_ids[i] != labels[i]` → 错误位，需要更高权重
   - 这是你之前发现的 bug 的修复方案

3. **推理时用于"保留原字"策略**:
   - 目标侧存的是原始 src 字符
   - 当阈值判断不需要修改时，应该保留原字符 `trg_ref_ids[i]`
   - 而不是 `input_ids[i]`（那是 MASK）

#### 1.3.4 `block_flag`（Prompt位置标记）

**格式**: Prompt 位置为 1，其他位置为 0

```
位置:     |  Prompt前  |   源句区域   |  Prompt后  |   目标区域     |
内容:     |  1  1  1   |  0  0  0  0 |   1  1  1  |  0  0  0  0   |
```

**用途**: P-tuning（Prompt Tuning）
- 值为 1 的位置会被替换为可学习的 Prompt Embedding
- 这些 Prompt 经过 LSTM + Linear 层处理后注入模型

#### 1.3.5 `error_labels`（检测标签）

**格式**: 源句区域为 0/1，其他位置为 -100

```
位置:     |  Prompt前  |   源句区域   |  Prompt后  |   目标区域     |
内容:     | -100 -100 -100 | 0 0 0 1 | -100 -100 -100 | -100 -100 -100 -100 |
含义:     |    ignore    | 正确正确正确错误 |   ignore   |     ignore      |
```

**用途**: 训练 DetectorHead（检测分支）
- DetectorHead 只在源句区域预测每个位置是否有错
- 为什么目标区域是 -100？因为目标区域的纠错由 CandidateHead 负责

**生成逻辑**:
```python
# 源句区域
if input_ids[i] != trg_ref_ids[i]:  # src[i] != trg[i]
    error_labels[i] = 1  # 错误
else:
    error_labels[i] = 0  # 正确
```

#### 1.3.6 `candidate_ids`（混淆候选集）

**格式**: (seq_len, K) 每个位置 K 个候选字符

```
位置 "心" 的候选集（K=8）: [心, 兴, 欣, 新, 芯, 薪, 馨, PAD]
位置 "我" 的候选集（K=8）: [我, 窝, 沃, 握, 卧, 喔, PAD, PAD]
```

**用途**: CandidateHead（候选注意力头）
- 每个位置的候选集包含：
  1. 原字符（允许"不改"）
  2. 同音/近音候选
  3. 形近候选
- CandidateHead 在候选集范围内预测，比全词表 softmax 更容易命中正确替换

---

## 2. 模型前向传播流程

### 2.1 输入预处理

```python
# 1. 获取 word embeddings
inputs_embeds = word_embeddings(input_ids)

# 2. P-tuning: 替换 Prompt 位置的 embedding
if apply_prompt:
    prompt_embeds = prompt_embeddings(range(2*P))
    prompt_embeds = prompt_lstm(prompt_embeds)
    prompt_embeds = prompt_linear(prompt_embeds)
    inputs_embeds[block_flag == 1] = prompt_embeds
```

### 2.2 BERT 编码

```python
# 获取 hidden states
outputs = bert(inputs_embeds, attention_mask)
hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
```

### 2.3 检测分支（DetectorHead）

```python
# 只在源句区域预测是否错误
detection_logits = detector_head(hidden_states)  # (batch, seq_len)
detection_probs = sigmoid(detection_logits)

# 计算检测损失（只在 error_labels != -100 的位置）
detection_loss = BCE(detection_logits, error_labels, pos_weight=β)
```

**参数 `pos_weight=β`（如 β=3）**:
- 让模型更重视"检测到错误"（少漏检）
- 与追求 Recall/F2 的目标一致

### 2.4 MLM Head（全词表预测）

```python
# 标准 BERT MLM head
vocab_logits = bert.cls(hidden_states)  # (batch, seq_len, vocab_size)
```

### 2.5 候选注意力头（CandidateHead）

```python
# 获取候选集 embeddings
candidate_embeds = word_embeddings(candidate_ids)  # (batch, seq_len, K, hidden)

# 计算候选分布
h_transformed = candidate_head.transform(hidden_states)
scores = h_transformed @ candidate_embeds.T  # (batch, seq_len, K)
candidate_probs = softmax(scores)
```

### 2.6 门控融合（GatedFusion）

```python
# 计算门控权重 α
gate_input = concat([hidden_states, detection_probs])
alpha = sigmoid(gate_linear(gate_input))  # (batch, seq_len)

# 融合分布
# p_fused = (1-α) * p_vocab + α * p_candidate
vocab_probs = softmax(vocab_logits)
fused_probs = (1 - alpha) * vocab_probs
fused_probs.scatter_add_(candidate_ids, alpha * candidate_probs)
```

**门控直觉**:
- `detection_probs` 高（可能有错）→ `α` 大 → 候选头权重大 → 更敢改 → Recall↑
- `detection_probs` 低（没错）→ `α` 小 → 全词表权重大 → 倾向保留 → Precision↑

---

## 3. 损失函数计算

### 3.1 纠错损失（核心）

```python
# 交叉熵损失
correction_loss = CrossEntropy(fused_logits, labels, reduction='none')

# 错误位加权（关键修复！）
if error_position_weight > 0 and trg_ref_ids is not None:
    weight = ones_like(labels)
    # 错误位：trg_ref_ids（目标侧=src）!= labels（目标侧=trg）
    error_positions = (trg_ref_ids != labels) & (labels != -100)
    weight[error_positions] = 1 + error_position_weight  # 如 1+3=4倍权重
    correction_loss = correction_loss * weight

# 只在有效位置计算
correction_loss = correction_loss[labels != -100].mean()
```

**为什么要用 `trg_ref_ids` 判断错误位？**
- `labels` 在目标侧是 trg（正确答案）
- `trg_ref_ids` 在目标侧是 src（原始错误句）
- `trg_ref_ids != labels` 就是 `src != trg`，正确找到错误位

### 3.2 检测损失

```python
detection_loss = BCEWithLogits(
    detection_logits, 
    error_labels.float(),
    pos_weight=detector_pos_weight  # 如 3.0
)
# 只在 error_labels != -100 的位置计算
```

### 3.3 候选排序损失（可选）

```python
# Margin ranking: 让正确候选分数高于其他候选
# L_rank = max(0, margin - score(y) + score(c))
rank_loss = relu(margin - target_score + other_scores).mean()
```

### 3.4 辅助 MLM 损失（抵抗对齐捷径）

```python
# 在源句非错误位置随机 mask，让模型重建
if mft:
    # 只 mask 正确位置（mask_mode="noerror"）
    masked_input_ids, aux_mask = dynamic_mask_token(input_ids, trg_ref_ids)
    
    # 构建辅助标签：被 mask 位置的原始字符
    aux_mlm_labels = full_like(input_ids, -100)
    aux_mlm_labels[aux_mask] = original_input_ids[aux_mask]
    
    aux_mlm_loss = CrossEntropy(vocab_logits, aux_mlm_labels)
```

**为什么这样做？**
- 防止模型学到"字到字的对齐捷径"
- 强迫模型利用全局语义来预测被 mask 的字符

### 3.5 总损失

```python
loss = correction_loss
loss += detector_loss_weight * detection_loss
loss += rank_loss_weight * rank_loss
loss += aux_mlm_loss_weight * aux_mlm_loss
```

---

## 4. 推理流程

### 4.1 前向传播

```python
with torch.no_grad():
    outputs = model(input_ids, attention_mask, ...)
    logits = outputs['logits']
    detection_probs = outputs['detection_probs']
    gate_weights = outputs['gate_weights']

# 取最大概率的预测
_, prd_ids = torch.max(logits, dim=-1)
```

### 4.2 选择性改动策略

```python
# 只有超过阈值才执行修改
should_modify = (detection_probs > detect_threshold) | (gate_weights > gate_threshold)

# 不修改的位置保留原字符（关键修复！）
# 使用 trg_ref_ids（目标侧是原字符），而不是 input_ids（目标侧是 MASK）
prd_ids = torch.where(should_modify, prd_ids, trg_ref_ids)
```

**为什么用 `trg_ref_ids` 而不是 `input_ids`？**
- `input_ids` 在目标侧是 `[MASK]` token
- `trg_ref_ids` 在目标侧是原始字符 src
- "保留原字"应该保留的是原始字符，不是 MASK

### 4.3 解码输出

```python
# 只解码目标区域（SEP 后）
for src, trg, prd in zip(src_ids, trg_ids, prd_ids):
    output_text = []
    in_target_region = False
    for s, t, p in zip(src, trg, prd):
        if s == SEP_ID:
            in_target_region = True
            continue
        if in_target_region:
            output_text.append(tokenizer.decode(p))
```

---

## 5. 关键设计决策总结

| 设计决策 | 原因 | 影响 |
|---------|------|------|
| 源句侧不参与纠错 loss | 源句由 DetectorHead 处理，分工明确 | 训练更稳定 |
| `trg_ref_ids` 目标侧存 src | 用于判断错误位 + 推理时保留原字 | Recall↑ |
| 错误位加权 | 让模型更关注错误位的纠正 | Recall/F2↑ |
| 检测 pos_weight | 减少漏检 | Recall↑ |
| 门控融合 | 平衡"敢改"与"不过纠" | Recall↑ & Precision 稳定 |
| 辅助 MLM（MFT） | 防止对齐捷径 | 泛化能力↑ |

---

## 6. 常见问题

### Q1: 为什么不直接用 `error_labels` 做纠错位加权？

因为 `error_labels` 只在源句区域有 0/1 标签，目标区域全是 -100。而纠错 loss 是在目标区域计算的，所以必须用 `trg_ref_ids` 来判断目标区域的错误位。

### Q2: 为什么推理时不能用 `input_ids` 保留原字？

因为 `input_ids` 在目标区域是 `[MASK]` token，不是原始字符。decode 时 `[MASK]` 会被跳过或产生乱序。

### Q3: `trg_ref_ids` 为什么要这样设计（前半 trg，后半 src）？

这是一个巧妙的设计，一个字段同时满足两个需求：
- 源句区域（前半）存 trg：用于生成 `error_labels`（`src != trg`）
- 目标区域（后半）存 src：用于纠错位加权和推理时保留原字
