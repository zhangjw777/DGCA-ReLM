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
│   └── confusion_dict.json   # 混淆集数据（需要替换为完整版）
├── multiTask/
│   └── DGCAModel.py          # DGCA-ReLM模型核心实现
├── utils/
│   └── dgca_data_processor.py # DGCA数据处理模块
├── run_dgca_relm.py          # DDP训练脚本
└── preprocess_data.py        # 大规模数据预处理脚本
```

## 数据格式

训练数据为txt文件，每行一个样本，格式为：
```
源句（空格分隔每个字）\t目标句（空格分隔每个字）
```

**示例：**
```
依 法 制 国 以 构 建 ...	依 法 治 国 以 构 建 ...
法 人 对 其 所 负 债 物 承 担 有 线 责 任	法 人 对 其 所 负 债 务 承 担 有 限 责 任
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
confusion_file: "confusion/confusion_dict.json"
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

## 运行命令

### 数据预处理（百万级数据推荐）

```bash
# 使用预处理脚本（多进程并行，推荐百万级数据）
python preprocess_data.py \
    --input_file data/train_million.txt \
    --output_file data/train_preprocessed.pt \
    --model_path bert-base-chinese \
    --confusion_file confusion/confusion_dict.json \
    --num_workers 16
```

### 单卡训练（小数据，原始txt格式）

```bash
python run_dgca_relm.py \
    --do_train \
    --do_eval \
    --data_dir data/ecspell/ \
    --train_on law \
    --eval_on law \
    --load_model_path bert-base-chinese \
    --output_dir output/dgca_relm_law/ \
    --dgca_config config/default_config.yaml \
    --confusion_file confusion/confusion_dict.json \
    --train_batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --prompt_length 3 \
    --apply_prompt \
    --mft \
    --mask_rate 0.2 \
    --fp16
```

### 单卡训练（大数据，预处理.pt格式）

```bash
python run_dgca_relm.py \
    --do_train \
    --do_eval \
    --preprocessed_train data/train_preprocessed.pt \
    --preprocessed_eval data/eval_preprocessed.pt \
    --load_model_path bert-base-chinese \
    --output_dir output/dgca_relm_large/ \
    --dgca_config config/default_config.yaml \
    --confusion_file confusion/confusion_dict.json \
    --train_batch_size 64 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --prompt_length 3 \
    --apply_prompt \
    --mft \
    --fp16
```

### DDP双卡训练

```bash
# 使用 torchrun 启动DDP训练
torchrun --nproc_per_node=2 --master_port=29500 run_dgca_relm.py \
    --do_train \
    --do_eval \
    --preprocessed_train data/train_preprocessed.pt \
    --preprocessed_eval data/eval_preprocessed.pt \
    --load_model_path bert-base-chinese \
    --output_dir output/dgca_relm_ddp/ \
    --dgca_config config/default_config.yaml \
    --train_batch_size 128 \
    --num_train_epochs 5 \
    --learning_rate 3e-5 \
    --prompt_length 3 \
    --apply_prompt \
    --mft \
    --fp16
```

### 测试

```bash
python run_dgca_relm.py \
    --do_test \
    --data_dir data/ecspell/ \
    --test_on law \
    --load_model_path bert-base-chinese \
    --load_state_dict output/dgca_relm_law/best_model.bin \
    --dgca_config config/default_config.yaml \
    --output_dir output/dgca_relm_law/
```

### 消融实验

```bash
# 运行baseline（原版ReLM，无DGCA模块）
python run_dgca_relm.py \
    --do_train --do_eval \
    --ablation baseline \
    --output_dir output/ablation_baseline/ \
    ...

# 仅检测分支
python run_dgca_relm.py \
    --do_train --do_eval \
    --ablation detector_only \
    --output_dir output/ablation_detector/ \
    ...

# 仅候选头
python run_dgca_relm.py \
    --do_train --do_eval \
    --ablation candidate_only \
    --output_dir output/ablation_candidate/ \
    ...
```

## 混淆集格式

混淆集文件 `confusion/confusion_dict.json` 格式：

```json
{
  "一": ["壹", "依", "伊", "医", "衣", "移"],
  "不": ["步", "部", "布", "补", "捕"],
  ...
}
```

其中key是原字符，value是混淆字符列表（同音/近音/形近字）。

## 输出说明

训练过程会输出：
- `step-{step}_f1-{f1}.bin`: 模型检查点
- `eval_results.txt`: 评估结果日志
- `dgca_config.yaml`: 保存的配置文件

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

1. 混淆集文件需要替换为完整版本
2. DDP训练时batch_size是总batch size，会自动分配到各卡
3. 使用`--mft`开启Masked-FT技术（ReLM核心）
4. 使用`--apply_prompt`开启P-tuning
5. 建议使用`--fp16`加速训练
6. 百万级数据建议先用`preprocess_data.py`预处理
