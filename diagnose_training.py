"""
训练问题诊断脚本
检查数据、模型、损失函数是否正常
"""

import torch
import argparse
from transformers import AutoTokenizer
from config.dgca_config import DGCAConfig
from confusion.confusion_utils import ConfusionSet


def check_preprocessed_data(data_path, tokenizer, max_samples=10):
    """检查预处理数据的正确性"""
    print(f"\n{'='*80}")
    print(f"检查预处理数据: {data_path}")
    print(f"{'='*80}")
    
    try:
        # 加载数据
        data_dict = torch.load(data_path)
        
        print(f"✓ 成功加载数据")
        print(f"  样本数: {len(data_dict['input_ids'])}")
        
        # 检查数据格式
        required_keys = ['input_ids', 'attention_mask', 'labels', 'trg_ref_ids', 
                         'block_flag', 'error_labels', 'candidate_ids']
        for key in required_keys:
            if key in data_dict:
                print(f"  ✓ {key}: {data_dict[key].shape}")
            else:
                print(f"  ✗ 缺少字段: {key}")
                return False
        
        # 统计error_labels
        error_labels = data_dict['error_labels']
        valid_mask = (error_labels != -100)
        error_mask = (error_labels == 1) & valid_mask
        correct_mask = (error_labels == 0) & valid_mask
        
        total_valid = valid_mask.sum().item()
        total_errors = error_mask.sum().item()
        total_correct = correct_mask.sum().item()
        error_rate = total_errors / total_valid if total_valid > 0 else 0
        
        print(f"\n错误标签统计:")
        print(f"  有效位置数: {total_valid}")
        print(f"  错误位置数: {total_errors} ({error_rate*100:.2f}%)")
        print(f"  正确位置数: {total_correct} ({(1-error_rate)*100:.2f}%)")
        
        if error_rate < 0.05:
            print(f"  ⚠️  警告: 错误率过低 ({error_rate*100:.2f}%)，可能数据生成有问题！")
            print(f"      正常应该在15-25%左右")
        
        # 打印前几个样本
        print(f"\n前{max_samples}个样本:")
        for i in range(min(max_samples, len(data_dict['input_ids']))):
            input_ids = data_dict['input_ids'][i]
            labels = data_dict['labels'][i]
            error_label = data_dict['error_labels'][i]
            
            # 解码
            src_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            trg_tokens = tokenizer.convert_ids_to_tokens(labels)
            
            # 找到非pad位置
            valid_pos = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(valid_pos) > 20:  # 只显示前20个token
                valid_pos = valid_pos[:20]
            
            print(f"\n样本 {i}:")
            print(f"  src: {' '.join([src_tokens[j] for j in valid_pos])}")
            print(f"  trg: {' '.join([trg_tokens[j] for j in valid_pos])}")
            
            # 标记错误位置
            error_positions = []
            for j in valid_pos:
                if input_ids[j] != labels[j] and labels[j] != -100:
                    error_positions.append(j.item())
            
            if error_positions:
                print(f"  错误位置: {error_positions}")
                print(f"  error_labels: {[error_label[j].item() for j in error_positions]}")
            else:
                print(f"  无错误（干净样本）")
        
        return True
        
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return False


def check_model_gradient(model_path, data_path, config_path):
    """检查模型梯度是否正常流动"""
    print(f"\n{'='*80}")
    print(f"检查模型梯度流动")
    print(f"{'='*80}")
    
    from transformers import BertForMaskedLM
    from multiTask.DGCAModel import DGCAReLMWrapper
    from utils.dgca_data_processor import PreprocessedDataset
    from torch.utils.data import DataLoader
    
    try:
        # 加载配置
        dgca_config = DGCAConfig.from_yaml(config_path)
        print(f"✓ DGCA配置:")
        print(f"  detector_loss_weight: {dgca_config.detector_loss_weight}")
        print(f"  rank_loss_weight: {dgca_config.rank_loss_weight}")
        print(f"  aux_mlm_loss_weight: {dgca_config.aux_mlm_loss_weight}")
        print(f"  error_position_weight: {dgca_config.error_position_weight}")
        
        # 加载tokenizer和混淆集
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        confusion_set = ConfusionSet(
            confusion_dir=dgca_config.confusion_dir,
            confusion_file=dgca_config.confusion_file,
            tokenizer=tokenizer,
            candidate_size=dgca_config.candidate_size,
            include_original=dgca_config.include_original_char
        )
        
        # 加载模型
        bert_model = BertForMaskedLM.from_pretrained(model_path, return_dict=True)
        model = DGCAReLMWrapper(
            bert_model=bert_model,
            confusion_set=confusion_set,
            config=dgca_config,
            prompt_length=3
        )
        model.cuda()
        model.train()
        
        print(f"✓ 模型加载成功")
        
        # 加载一个batch的数据
        dataset = PreprocessedDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        batch = next(iter(dataloader))
        
        print(f"✓ 数据加载成功，batch_size={len(batch['input_ids'])}")
        
        # 前向传播
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            prompt_mask=batch['block_flag'],
            labels=batch['labels'],
            candidate_ids=batch['candidate_ids'],
            error_labels=batch['error_labels'],
            apply_prompt=True
        )
        
        loss = outputs['loss']
        print(f"\n前向传播结果:")
        print(f"  总loss: {loss.item():.4f}")
        if outputs['correction_loss'] is not None:
            print(f"  correction_loss: {outputs['correction_loss'].item():.4f}")
        if outputs['detection_loss'] is not None:
            print(f"  detection_loss: {outputs['detection_loss'].item():.4f}")
        if outputs['rank_loss'] is not None:
            print(f"  rank_loss: {outputs['rank_loss'].item():.4f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        print(f"\n梯度统计:")
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm
        
        # 检查关键模块的梯度
        key_modules = ['detector_head', 'candidate_head', 'gated_fusion', 
                       'prompt_embeddings', 'bert.encoder.layer.11']
        
        for module_name in key_modules:
            module_grads = {k: v for k, v in grad_stats.items() if module_name in k}
            if module_grads:
                avg_grad = sum(module_grads.values()) / len(module_grads)
                max_grad = max(module_grads.values())
                print(f"  {module_name}:")
                print(f"    平均梯度范数: {avg_grad:.6f}")
                print(f"    最大梯度范数: {max_grad:.6f}")
                
                if avg_grad < 1e-7:
                    print(f"    ⚠️  梯度过小，可能梯度消失！")
                elif avg_grad > 100:
                    print(f"    ⚠️  梯度过大，可能梯度爆炸！")
            else:
                print(f"  {module_name}: 无梯度（可能被冻结或不存在）")
        
        # 检查detection_probs分布
        if outputs['detection_probs'] is not None:
            det_probs = outputs['detection_probs']
            print(f"\n检测概率分布:")
            print(f"  平均值: {det_probs.mean().item():.4f}")
            print(f"  中位数: {det_probs.median().item():.4f}")
            print(f"  最小值: {det_probs.min().item():.4f}")
            print(f"  最大值: {det_probs.max().item():.4f}")
            
            if det_probs.mean().item() > 0.95:
                print(f"  ⚠️  检测概率过高，模型倾向于将所有位置标记为错误！")
            elif det_probs.mean().item() < 0.05:
                print(f"  ⚠️  检测概率过低，模型倾向于将所有位置标记为正确！")
        
        return True
        
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.pt")
    parser.add_argument("--eval_data", type=str, default="data/dev.pt")
    parser.add_argument("--model_path", type=str, default="bert-base-chinese")
    parser.add_argument("--config_path", type=str, default="config/default_config.yaml")
    parser.add_argument("--check_data", action="store_true", help="检查数据")
    parser.add_argument("--check_gradient", action="store_true", help="检查梯度")
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    all_ok = True
    
    # 检查数据
    if args.check_data:
        print("\n" + "="*80)
        print("数据诊断")
        print("="*80)
        
        ok = check_preprocessed_data(args.train_data, tokenizer, max_samples=5)
        all_ok = all_ok and ok
        
        if args.eval_data:
            ok = check_preprocessed_data(args.eval_data, tokenizer, max_samples=3)
            all_ok = all_ok and ok
    
    # 检查梯度
    if args.check_gradient:
        ok = check_model_gradient(args.model_path, args.train_data, args.config_path)
        all_ok = all_ok and ok
    
    # 总结
    print(f"\n{'='*80}")
    if all_ok:
        print("✓ 所有检查通过")
    else:
        print("✗ 发现问题，请根据上述提示修复")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
