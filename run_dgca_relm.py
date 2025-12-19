"""
DGCA-ReLM训练脚本
支持DDP双卡训练，基于原ReLM改造
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import SchedulerType, get_scheduler

# 项目模块
from utils.data_processor import EcspellProcessor
from utils.metrics import Metrics
from utils.dgca_data_processor import (
    convert_examples_to_dgca_features,
    create_dgca_dataset
)
from config.dgca_config import DGCAConfig
from confusion.confusion_utils import ConfusionSet
from multiTask.DGCAModel import DGCAReLMWrapper


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def setup_ddp(args):
    """初始化DDP环境"""
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    return device


def cleanup_ddp():
    """清理DDP环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    """判断是否是主进程（用于日志输出）"""
    return args.local_rank == -1 or args.local_rank == 0


def set_seed(seed, n_gpu):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def dynamic_mask_token(inputs, targets, tokenizer, device, mask_mode="noerror", noise_probability=0.2):
    """
    ReLM的masked-FT技术：在源句中随机遮蔽一部分非错误字符
    """
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    if mask_mode == "noerror":
        probability_matrix.masked_fill_(inputs != targets, value=0.0)
    elif mask_mode == "error":
        probability_matrix.masked_fill_(inputs == targets, value=0.0)
    else:
        assert mask_mode == "all"
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    return inputs


def main():
    parser = argparse.ArgumentParser()
    
    # ============ 数据配置 ============
    parser.add_argument("--data_dir", type=str, default="data/ecspell/",
                        help="数据目录")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese",
                        help="预训练模型路径")
    parser.add_argument("--cache_dir", type=str, default="../../cache/",
                        help="模型缓存目录")
    parser.add_argument("--output_dir", type=str, default="outputs/dgca_relm/",
                        help="输出目录")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="加载已训练的模型权重")
    
    # ============ DGCA配置 ============
    parser.add_argument("--dgca_config", type=str, default="config/default_config.yaml",
                        help="DGCA配置文件路径（包含混淆集路径、候选集大小等）")
    parser.add_argument("--ablation", type=str, default=None,
                        help="使用预定义的消融配置：full/detector_only/candidate_only/baseline")
    
    # ============ 训练配置 ============
    parser.add_argument("--do_train", action="store_true", help="是否训练")
    parser.add_argument("--do_eval", action="store_true", help="是否验证")
    parser.add_argument("--do_test", action="store_true", help="是否测试")
    parser.add_argument("--train_on", type=str, default="law", help="训练集划分：law/med/odw")
    parser.add_argument("--eval_on", type=str, default="law")
    parser.add_argument("--test_on", type=str, default="law")
    
    # ============ 预处理数据（可选，加速训练） ============
    parser.add_argument("--preprocessed_train", type=str, default=None)
    parser.add_argument("--preprocessed_eval", type=str, default=None)
    parser.add_argument("--preprocessed_test", type=str, default=None)
    
    # ============ 模型与数据超参 ============
    parser.add_argument("--prompt_length", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    
    # ============ 训练超参 ============
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    
    # ============ Early Stopping ============
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="Early stopping patience，连续 N 次 eval 指标不提升则停止。"
                             "默认None表示不启用")
    parser.add_argument("--early_stopping_metric", type=str, default="f1",
                        choices=["f1", "f2", "precision", "recall"],
                        help="Early stopping 监控的指标")
    
    # ============ 硬件配置 ============
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    
    # ============ ReLM相关 ============
    parser.add_argument("--mft", action="store_true", help="masked-fine-tuning")
    parser.add_argument("--mask_mode", type=str, default="noerror", 
                        help="遮蔽模式：noerror/error/all")
    parser.add_argument("--mask_rate", type=float, default=0.2)
    parser.add_argument("--anchor", type=str, default=None, help="锚点字符串")
    parser.add_argument("--apply_prompt", action="store_true", help="使用prompt")
    parser.add_argument("--freeze_lm", action="store_true", help="冻结语言模型")
    
    args = parser.parse_args()
    
    # ============ 初始化环境 ============
    device = setup_ddp(args)
    
    n_gpu = 1 if args.local_rank != -1 else torch.cuda.device_count()
    
    if is_main_process(args):
        logger.info(f"Device: {device}, n_gpu: {n_gpu}, DDP: {args.local_rank != -1}, FP16: {args.fp16}")
    
    # 设置随机种子
    set_seed(args.seed, n_gpu)
    
    # 创建输出目录
    if is_main_process(args) and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # ============ 加载DGCA配置 ============
    if args.ablation:
        from config.dgca_config import get_ablation_config
        dgca_config = get_ablation_config(args.ablation)
        if is_main_process(args):
            logger.info(f"Using ablation config: {args.ablation}")
    else:
        dgca_config = DGCAConfig.from_yaml(args.dgca_config)
    
    if is_main_process(args):
        logger.info(f"DGCA Config:\n{dgca_config}")
        # 保存配置
        dgca_config.to_yaml(os.path.join(args.output_dir, "dgca_config.yaml"))
    
    # ============ 加载Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(
        args.load_model_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer
    )
    
    # ============ 加载混淆集 ============
    confusion_set = ConfusionSet(
        confusion_dir=dgca_config.confusion_dir,
        confusion_file=dgca_config.confusion_file,
        tokenizer=tokenizer,
        candidate_size=dgca_config.candidate_size,
        include_original=dgca_config.include_original_char
    )
    
    if is_main_process(args):
        logger.info(f"Loaded confusion set with {len(confusion_set.confusion_dict)} entries")
    
    # ============ 处理anchor ============
    anchor = None
    if args.anchor is not None:
        anchor = [tokenizer.sep_token] + [t for t in args.anchor]
    
    # ============ 数据处理 ============
    processor = EcspellProcessor()
    
    # 导入预处理数据集类
    from utils.dgca_data_processor import PreprocessedDataset
    
    if args.do_train:
        # 优先使用预处理数据
        if args.preprocessed_train:
            if is_main_process(args):
                logger.info(f"Loading preprocessed training data from {args.preprocessed_train}")
            train_dataset = PreprocessedDataset(args.preprocessed_train)
        else:
            # 原始方式：读取txt文件并处理
            train_examples = processor.get_train_examples(args.data_dir, args.train_on)
            train_features = convert_examples_to_dgca_features(
                train_examples,
                args.max_seq_length,
                tokenizer,
                confusion_set,
                args.prompt_length,
                anchor=anchor,
                mask_rate=args.mask_rate
            )
            train_dataset = create_dgca_dataset(train_features)
            if is_main_process(args):
                logger.info(f"Loaded {len(train_examples)} training examples")
        
        if is_main_process(args):
            logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # DDP Sampler
        if args.local_rank != -1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset)
        
        # 计算实际batch size
        per_device_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=per_device_batch_size
        )
        
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    if args.do_eval:
        # 优先使用预处理数据
        if args.preprocessed_eval:
            if is_main_process(args):
                logger.info(f"Loading preprocessed eval data from {args.preprocessed_eval}")
            eval_dataset = PreprocessedDataset(args.preprocessed_eval)
        else:
            eval_examples = processor.get_dev_examples(args.data_dir, args.eval_on)
            eval_features = convert_examples_to_dgca_features(
                eval_examples,
                args.max_seq_length,
                tokenizer,
                confusion_set,
                args.prompt_length,
                anchor=anchor
            )
            eval_dataset = create_dgca_dataset(eval_features)
        
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size
        )
    
    # ============ 创建模型 ============
    bert_model = BertForMaskedLM.from_pretrained(
        args.load_model_path,
        return_dict=True,
        cache_dir=args.cache_dir
    )
    
    model = DGCAReLMWrapper(
        bert_model=bert_model,
        confusion_set=confusion_set,
        config=dgca_config,
        prompt_length=args.prompt_length
    )
    
    model.to(device)
    
    # 加载已有权重
    if args.load_state_dict:
        model.load_state_dict(torch.load(args.load_state_dict, map_location=device))
        if is_main_process(args):
            logger.info(f"Loaded model from {args.load_state_dict}")
    
    # DDP包装
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # ============ 优化器 ============
    if args.do_train:
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.max_train_steps * args.warmup_proportion),
            num_training_steps=args.max_train_steps
        )
        
        # 冻结LM参数
        if args.freeze_lm:
            trainable_params = ["prompt_embeddings", "prompt_lstm", "prompt_linear",
                               "detector_head", "candidate_head", "gated_fusion"]
            for n, p in model.named_parameters():
                if not any(tp in n for tp in trainable_params):
                    p.requires_grad = False
                    if is_main_process(args):
                        logger.info(f"Freeze: {n}")
        
        # 混合精度
        scaler = None
        if args.fp16:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
    
    # ============ 训练 ============
    if args.do_train:
        # 初始化TensorBoard（仅主进程）
        tb_writer = None
        if is_main_process(args):
            tb_log_dir = os.path.join(args.output_dir, "tensorboard")
            os.makedirs(tb_log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info(f"TensorBoard log dir: {tb_log_dir}")
        
        if is_main_process(args):
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Batch size = {args.train_batch_size}")
            logger.info(f"  Num steps = {args.max_train_steps}")
            # 保存训练参数
            torch.save(args, os.path.join(args.output_dir, "train_args.bin"))
        
        global_step = 0
        best_result = []
        wrap = False
                # Early Stopping 相关
        best_metric = 0.0
        patience_counter = 0
                # 用于累积loss分项以便记录平均值
        accumulated_losses = {
            'total': 0.0,
            'correction': 0.0,
            'detection': 0.0,
            'rank': 0.0
        }
        
        progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process(args))
        
        for epoch in range(int(args.num_train_epochs)):
            if wrap:
                break
            
            # DDP每个epoch shuffle
            if args.local_rank != -1:
                train_sampler.set_epoch(epoch)
            
            train_loss = 0
            train_steps = 0
            
            for step, batch in enumerate(train_dataloader):
                model.train()
                
                # 解包batch（兼容TensorDataset元组和自定义Dataset字典两种格式）
                if isinstance(batch, dict):
                    # 自定义Dataset返回字典
                    src_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    trg_ids = batch['labels'].to(device)
                    trg_ref_ids = batch['trg_ref_ids'].to(device)
                    block_flag = batch['block_flag'].to(device)
                    error_labels = batch['error_labels'].to(device)
                    candidate_ids = batch['candidate_ids'].to(device)
                else:
                    # TensorDataset返回元组
                    src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag, error_labels, candidate_ids = \
                        [t.to(device) for t in batch]
                
                # Masked-FT
                if args.mft:
                    src_ids = dynamic_mask_token(
                        src_ids, trg_ref_ids, tokenizer, device,
                        args.mask_mode, args.mask_rate
                    )
                
                # 处理labels（-100 ignore）
                labels = trg_ids.clone()
                labels[src_ids == trg_ids] = -100
                
                # 前向传播
                if args.fp16:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = model(
                            input_ids=src_ids,
                            attention_mask=attention_mask,
                            prompt_mask=block_flag,
                            labels=labels,
                            candidate_ids=candidate_ids,
                            error_labels=error_labels,
                            apply_prompt=args.apply_prompt
                        )
                else:
                    outputs = model(
                        input_ids=src_ids,
                        attention_mask=attention_mask,
                        prompt_mask=block_flag,
                        labels=labels,
                        candidate_ids=candidate_ids,
                        error_labels=error_labels,
                        apply_prompt=args.apply_prompt
                    )
                
                loss = outputs['loss']
                
                # 累积loss分项用于TensorBoard记录
                accumulated_losses['total'] += loss.item()
                if outputs.get('correction_loss') is not None:
                    accumulated_losses['correction'] += outputs['correction_loss'].item()
                if outputs.get('detection_loss') is not None:
                    accumulated_losses['detection'] += outputs['detection_loss'].item()
                if outputs.get('rank_loss') is not None:
                    accumulated_losses['rank'] += outputs['rank_loss'].item()
                
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # 反向传播
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                train_loss += loss.item()
                train_steps += 1
                
                # 梯度更新
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)
                
                # 日志记录
                if global_step % args.logging_steps == 0 and is_main_process(args):
                    avg_loss = train_loss / train_steps
                    logger.info(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    
                    # TensorBoard记录训练指标
                    if tb_writer is not None:
                        # 记录loss分项（取平均）
                        log_steps = args.logging_steps
                        tb_writer.add_scalar('train/loss_total', accumulated_losses['total'] / log_steps, global_step)
                        if accumulated_losses['correction'] > 0:
                            tb_writer.add_scalar('train/loss_correction', accumulated_losses['correction'] / log_steps, global_step)
                        if accumulated_losses['detection'] > 0:
                            tb_writer.add_scalar('train/loss_detection', accumulated_losses['detection'] / log_steps, global_step)
                        if accumulated_losses['rank'] > 0:
                            tb_writer.add_scalar('train/loss_rank', accumulated_losses['rank'] / log_steps, global_step)
                        
                        # 记录学习率
                        current_lr = scheduler.get_last_lr()[0]
                        tb_writer.add_scalar('train/learning_rate', current_lr, global_step)
                        
                        # 重置累积
                        accumulated_losses = {k: 0.0 for k in accumulated_losses}
                
                # 验证和保存
                if args.do_eval and global_step % args.save_steps == 0 and is_main_process(args):
                    eval_result = evaluate(
                        model, eval_dataloader, tokenizer, device, args, dgca_config
                    )
                    
                    logger.info(f"***** Eval results at step {global_step} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value:.4f}")
                    
                    # TensorBoard记录验证指标
                    if tb_writer is not None:
                        tb_writer.add_scalar('eval/loss', eval_result['loss'], global_step)
                        tb_writer.add_scalar('eval/precision', eval_result['precision'], global_step)
                        tb_writer.add_scalar('eval/recall', eval_result['recall'], global_step)
                        tb_writer.add_scalar('eval/f1', eval_result['f1'], global_step)
                        tb_writer.add_scalar('eval/f2', eval_result['f2'], global_step)
                        tb_writer.add_scalar('eval/fpr', eval_result['fpr'], global_step)
                        tb_writer.add_scalar('eval/wpr', eval_result['wpr'], global_step)
                    
                    # 保存模型
                    model_to_save = model.module if hasattr(model, "module") else model
                    output_file = os.path.join(
                        args.output_dir,
                        f"step-{global_step}_f1-{eval_result['f1']:.2f}.bin"
                    )
                    torch.save(model_to_save.state_dict(), output_file)
                    
                    best_result.append((eval_result['f1'], output_file))
                    best_result.sort(key=lambda x: x[0], reverse=True)
                    
                    # 只保留前3个最佳模型
                    if len(best_result) > 3:
                        _, model_to_remove = best_result.pop()
                        if os.path.exists(model_to_remove):
                            os.remove(model_to_remove)
                    
                    # 保存评估结果
                    with open(os.path.join(args.output_dir, "eval_results.txt"), "a") as f:
                        f.write(f"Step {global_step}: P={eval_result['precision']:.2f}, "
                               f"R={eval_result['recall']:.2f}, F1={eval_result['f1']:.2f}, "
                               f"F2={eval_result['f2']:.2f}, FPR={eval_result['fpr']:.2f}\n")
                    
                    # Early Stopping 判断
                    if args.early_stopping_patience is not None:
                        current_metric = eval_result[args.early_stopping_metric]
                        if current_metric > best_metric:
                            best_metric = current_metric
                            patience_counter = 0
                            logger.info(f"New best {args.early_stopping_metric}: {best_metric:.4f}")
                        else:
                            patience_counter += 1
                            logger.info(f"Early stopping patience: {patience_counter}/{args.early_stopping_patience}")
                            
                            if patience_counter >= args.early_stopping_patience:
                                logger.info(f"Early stopping triggered! Best {args.early_stopping_metric}: {best_metric:.4f}")
                                wrap = True
                                break
                
                if global_step >= args.max_train_steps:
                    wrap = True
                    break
        
        # 训练结束，关闭TensorBoard writer
        if tb_writer is not None:
            tb_writer.close()
            logger.info("TensorBoard writer closed")
    
    # ============ 测试 ============
    if args.do_test:
        # 优先使用预处理数据
        if args.preprocessed_test:
            if is_main_process(args):
                logger.info(f"Loading preprocessed test data from {args.preprocessed_test}")
            test_dataset = PreprocessedDataset(args.preprocessed_test)
        else:
            test_examples = processor.get_test_examples(args.data_dir, args.test_on)
            test_features = convert_examples_to_dgca_features(
                test_examples,
                args.max_seq_length,
                tokenizer,
                confusion_set,
                args.prompt_length,
                anchor=anchor
            )
            test_dataset = create_dgca_dataset(test_features)
        
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=args.eval_batch_size
        )
        
        # 如果没有训练，需要重新加载模型
        if not args.do_train and args.load_state_dict:
            bert_model = BertForMaskedLM.from_pretrained(
                args.load_model_path,
                return_dict=True,
                cache_dir=args.cache_dir
            )
            model = DGCAReLMWrapper(
                bert_model=bert_model,
                confusion_set=confusion_set,
                config=dgca_config,
                prompt_length=args.prompt_length
            )
            model.load_state_dict(torch.load(args.load_state_dict, map_location=device))
            model.to(device)
        
        if is_main_process(args):
            logger.info("***** Running test *****")
            logger.info(f"  Num examples = {len(test_examples)}")
        
        test_result = evaluate(
            model, test_dataloader, tokenizer, device, args, dgca_config,
            save_predictions=True,
            output_dir=args.output_dir
        )
        
        if is_main_process(args):
            logger.info("***** Test results *****")
            for key, value in test_result.items():
                logger.info(f"  {key} = {value:.4f}")
    
    # 清理DDP
    cleanup_ddp()


def evaluate(model, dataloader, tokenizer, device, args, config,
             save_predictions=False, output_dir=None):
    """评估函数"""
    model.eval()
    
    all_inputs, all_labels, all_predictions = [], [], []
    eval_loss = 0
    eval_steps = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # 兼容TensorDataset元组和自定义Dataset字典两种格式
        if isinstance(batch, dict):
            src_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            trg_ids = batch['labels'].to(device)
            trg_ref_ids = batch['trg_ref_ids'].to(device)
            block_flag = batch['block_flag'].to(device)
            error_labels = batch['error_labels'].to(device)
            candidate_ids = batch['candidate_ids'].to(device)
        else:
            src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag, error_labels, candidate_ids = \
                [t.to(device) for t in batch]
        
        labels = trg_ids.clone()
        labels[src_ids == trg_ids] = -100
        
        with torch.no_grad():
            outputs = model(
                input_ids=src_ids,
                attention_mask=attention_mask,
                prompt_mask=block_flag,
                labels=labels,
                candidate_ids=candidate_ids,
                error_labels=error_labels,
                apply_prompt=args.apply_prompt
            )
            
            logits = outputs['logits']
            if outputs['loss'] is not None:
                eval_loss += outputs['loss'].item()
        
        # 解码预测结果
        _, prd_ids = torch.max(logits, dim=-1)
        prd_ids = prd_ids.masked_fill(attention_mask == 0, 0)
        
        src_ids_list = src_ids.tolist()
        trg_ids_list = trg_ids.tolist()
        prd_ids_list = prd_ids.tolist()
        
        for s, t, p in zip(src_ids_list, trg_ids_list, prd_ids_list):
            mapped_src, mapped_trg, mapped_prd = [], [], []
            flag = False
            
            for st, tt, pt in zip(s, t, p):
                if st == tokenizer.sep_token_id:
                    flag = True
                
                if not flag:
                    mapped_src.append(st)
                else:
                    mapped_trg.append(tt)
                    if st == tokenizer.mask_token_id:
                        mapped_prd.append(pt)
                    else:
                        mapped_prd.append(st)
            
            # 处理anchor
            if args.anchor is not None:
                anchor_length = len(args.anchor) + 1
                del mapped_trg[:anchor_length]
                del mapped_prd[:anchor_length]
            
            def decode(ids):
                return tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            
            all_inputs.append(decode(mapped_src))
            all_labels.append(decode(mapped_trg))
            all_predictions.append(decode(mapped_prd))
        
        eval_steps += 1
    
    # 计算指标
    p, r, f1, fpr, wpr, tp_sents, fp_sents, fn_sents, wp_sents = \
        Metrics.csc_compute(all_inputs, all_labels, all_predictions)
    
    # 计算F2（β=2，更偏向召回）
    beta = 2
    f2 = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    
    # 保存预测结果
    if save_predictions and output_dir:
        with open(os.path.join(output_dir, "sents.tp"), "w") as f:
            for line in tp_sents:
                f.write(line + "\n")
        
        with open(os.path.join(output_dir, "sents.fp"), "w") as f:
            for line in fp_sents:
                f.write(line + "\n")
        
        with open(os.path.join(output_dir, "sents.fn"), "w") as f:
            for line in fn_sents:
                f.write(line + "\n")
        
        with open(os.path.join(output_dir, "sents.wp"), "w") as f:
            for line in wp_sents:
                f.write(line + "\n")
    
    return {
        'loss': eval_loss / max(eval_steps, 1),
        'precision': p * 100,
        'recall': r * 100,
        'f1': f1 * 100,
        'f2': f2 * 100,
        'fpr': fpr * 100,
        'wpr': wpr * 100
    }


if __name__ == "__main__":
    main()
