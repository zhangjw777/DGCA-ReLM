# DGCA-ReLM 使用说明

## 项目结构

```
DGCA-ReLM/
├── config/
│   ├── __init__.py
│   ├── dgca_config.py        # DGCA配置类定义
│   └── default_config.yaml   # 默认配置文件
├── confusion/
│   ├── __init__.py
│   ├── confusion_utils.py    # 混淆集处理模块
│   ├── pinyin_sam.json       # 完全同音混淆集
│   ├── pinyin_sim.json       # 发音相似混淆集
│   └── stroke.json           # 字形相似混淆集
├── model/
│   └── DGCAModel.py          # DGCA-ReLM模型核心实现
├── utils/
│   ├── dgca_data_processor.py # DGCA数据处理模块
│   └── metrics.py            # 评估指标
├── run_dgca_relm.py          # 训练脚本（支持DDP）
└── preprocess_data.py        # 数据预处理脚本
```

## 快速开始

### 1. 数据预处理

从干净句子生成训练数据（自动造错 + 预处理为jsonl格式）：

```bash
python preprocess_data.py \
    --input_file data/clean_sentences.txt \
    --output_dir data/processed \
    --model_path bert-base-chinese \
    --avg_errors 1.5 \
    --no_error_ratio 0.15 \
    --split_ratio "0.8,0.1,0.1"
```

**输入格式**：每行一个干净句子的txt文件  
**输出**：`data/processed/` 下生成 `train.jsonl`, `dev.jsonl`, `test.jsonl`

### 2. 训练

```bash
CUDA_VISIBLE_DEVICES=1 python run_dgca_relm.py \
    --do_train --do_eval --do_test \
    --preprocessed_train data/processed/train.jsonl \
    --preprocessed_eval data/processed/dev.jsonl \
    --preprocessed_test data/processed/test.jsonl \
    --load_model_path bert-base-chinese \
    --output_dir outputs/dgca_relm/ \
    --train_batch_size 128 --eval_batch_size 128 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --save_steps 9000 --eval_steps 9000 \
    --fp16 --apply_prompt --mft \
    --num_workers 4
```

### 3. DDP双卡训练

```bash
torchrun --nproc_per_node=2 --master_port=29500 run_dgca_relm.py \
    --do_train --do_eval \
    --preprocessed_train data/processed/train.jsonl \
    --preprocessed_eval data/processed/dev.jsonl \
    --load_model_path bert-base-chinese \
    --output_dir outputs/dgca_relm_ddp/ \
    --train_batch_size 256 \
    --num_train_epochs 10 \
    --fp16 --apply_prompt --mft
```

## 核心模块说明

### 1. DetectorHead（检测分支）
- 预测每个位置是否为错误
- 输出检测概率 d_i
- 用于门控融合和检测损失计算

### 2. CandidateHead（候选注意力头）
- 在混淆集候选范围内预测目标字符
- 计算hidden state与候选embeddings的相似度
- 输出候选集上的概率分布

### 3. GatedFusion（门控融合）
- 动态融合全词表分布和候选集分布
- 门控权重 α_i = σ(W[h_i; d_i] + b)
- 最终分布：p_i = (1-α_i) * p_vocab + α_i * p_cand

## 配置说明

### config/default_config.yaml

```yaml
# 模型结构开关（消融实验）
use_detector: true           # 是否使用检测分支
use_candidate_head: true     # 是否使用候选注意力头
use_gated_fusion: true       # 是否使用门控融合

# 候选集相关
candidate_size: 16           # 候选集大小K
include_original_char: true  # 候选集包含原字符

# 损失权重
detector_loss_weight: 1.0    # 检测损失权重
detector_pos_weight: 3.0     # 检测正样本权重（提升召回）
error_position_weight: 2.0   # 错误位权重γ
rank_loss_weight: 0.5        # 排序损失权重
```

### 预定义消融配置

```python
# 在命令行使用 --ablation 参数
--ablation full              # 完整DGCA-ReLM
--ablation detector_only     # 仅检测分支
--ablation candidate_only    # 仅候选头
--ablation baseline          # ReLM基线
```

## 预处理参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--avg_errors` | 1.5 | 平均每句错误数（泊松分布λ） |
| `--no_error_ratio` | 0.15 | 无错样本比例（让模型学会"不改"） |
| `--split_ratio` | 0.8,0.1,0.1 | train/dev/test 划分比例 |
| `--max_seq_length` | 128 | 最大序列长度 |
| `--prompt_length` | 3 | Prompt token数量 |

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_batch_size` | 64 | 训练批次大小（DDP时为总大小） |
| `--eval_batch_size` | 64 | 评估批次大小 |
| `--learning_rate` | 5e-5 | 学习率 |
| `--num_train_epochs` | 10 | 训练轮数 |
| `--save_steps` | 5000 | 保存模型间隔（步数） |
| `--eval_steps` | None | 评估间隔（默认=save_steps） |
| `--fp16` | False | 混合精度训练 |
| `--mft` | False | Masked-FT技术（ReLM核心） |
| `--apply_prompt` | False | 启用P-tuning |
| `--early_stopping_patience` | None | Early stopping耐心值 |

## 混淆集说明

混淆集位于 `confusion/` 目录下：

| 文件 | 说明 | 使用概率 |
|------|------|----------|
| `pinyin_sam.json` | 完全同音混淆集 | 40% |
| `pinyin_sim.json` | 发音相似混淆集 | 30% |
| `stroke.json` | 字形相似混淆集 | 20% |
| `word_freq.txt` | 高频词（fallback） | 10% |

格式示例：
```json
{
  "一": ["壹", "依", "伊", "医", "衣", "移"],
  "不": ["步", "部", "布", "补", "捕"]
}
```

## 输出说明

训练过程会输出：
- `step-{step}_f1-{f1}.bin`: 模型检查点
- TensorBoard日志（自动记录loss和指标）

测试会输出：
- `sents.tp`: True Positive样本
- `sents.fp`: False Positive样本  
- `sents.fn`: False Negative样本
- `sents.wp`: Wrong Prediction样本

## 评估指标

- **Precision**: 纠错精确率
- **Recall**: 纠错召回率
- **F1**: F1分数
- **F2**: F2分数（β=2，更偏向召回）
- **FPR**: 句级误报率

## 注意事项

1. **必须使用预处理数据**：训练/评估/测试均需指定 `--preprocessed_*` 参数
2. DDP训练时 `batch_size` 是总大小，会自动分配到各卡
3. 使用 `--mft` 开启 Masked-FT 技术（ReLM核心）
4. 使用 `--apply_prompt` 开启 P-tuning
5. 建议使用 `--fp16` 加速训练
6. 大数据集建议设置较大的 `--eval_steps`（如每epoch评估一次）
