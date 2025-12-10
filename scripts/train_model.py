#!/usr/bin/env python3
"""
小红书CTR预估模型 - 统一模型训练入口
支持单任务和多任务训练的统一接口
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.single_task.ctr_trainer import LocalCTRTrainer
from training.multi_task.ple_trainer import PLETrainer
from training.multi_task.mmoe_trainer import MMOETrainer
from training.multi_task.pnn_mmoe_trainer import PNNMMOETrainer
from training.base.base_config import create_config_from_args

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    if log_file is None:
        log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 确保logs目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_file
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return str(log_file)


def train_single_task(args) -> Dict[str, Any]:
    """训练单任务模型"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting single-task CTR model training")

    # LocalCTRTrainer使用旧的**kwargs接口，需要直接传递参数
    # 处理自定义特征配置
    custom_config_args = {}
    if args.feature_config == "custom":
        custom_config_args = {
            'sparse_features': args.sparse_features.split(',') if args.sparse_features else [],
            'dense_features': args.dense_features.split(',') if args.dense_features else [],
            'use_clip_image': args.use_clip_image,
            'use_clip_text': args.use_clip_text
        }

    # 创建训练器
    trainer = LocalCTRTrainer(
        input_path=args.input,
        model_type=args.model,
        feature_config=args.feature_config,
        sample_size=getattr(args, 'sample_size', None),
        output_dir=args.output,
        label_column=getattr(args, 'label_column', 'ctr'),
        # 数据处理参数
        filter_zeros=args.filter_zeros,
        min_impression_threshold=args.min_impression,
        # 特征处理参数
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        use_fp16=getattr(args, 'use_fp16', False),
        # 训练参数
        use_early_stopping=not getattr(args, 'no_early_stopping', False),
        early_stopping_patience=getattr(args, 'early_stopping_patience', 10),
        early_stopping_monitor=getattr(args, 'early_stopping_monitor', 'loss'),
        early_stopping_min_delta=getattr(args, 'early_stopping_min_delta', 0.0001),
        restore_best_weights=getattr(args, 'restore_best_weights', True),
        save_best_model=getattr(args, 'save_best_model', False),
        # 可视化参数
        enable_visualization=getattr(args, 'enable_visualization', True),
        scatter_sample_size=getattr(args, 'scatter_sample_size', 1000),
        plot_dpi=getattr(args, 'plot_dpi', 150),
        # 自定义配置参数
        **custom_config_args
    )

    if args.all_models:
        # 训练所有模型并对比
        results = trainer.run_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=getattr(args, 'learning_rate', 0.001)
        )
    else:
        # 训练单个模型
        results = trainer.run(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=getattr(args, 'learning_rate', 0.001),
            test_size=getattr(args, 'test_size', 0.2),
            val_size=getattr(args, 'val_size', 0.1)
        )

    return results


def train_multi_task(args) -> Dict[str, Any]:
    """训练多任务模型"""
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Starting multi-task {args.multi_task_model} model training")

    # 添加参数映射，以便create_config_from_args正确解析
    args.input_path = args.input
    args.output_path = args.output
    args.model_type = args.multi_task_model

    # 使用create_config_from_args创建配置对象
    config = create_config_from_args(args, config_type="multi")

    # 根据模型类型创建相应的训练器
    if args.multi_task_model == 'PLE':
        trainer = PLETrainer(config)
    elif args.multi_task_model == 'MMOE':
        trainer = MMOETrainer(config)
    elif args.multi_task_model == 'PNN_MMOE':
        trainer = PNNMMOETrainer(config)
    else:
        raise ValueError(f"Unknown multi-task model: {args.multi_task_model}")

    # 运行训练
    results = trainer.run()

    return results


def auto_detect_input_path(stage: str = "multimodal") -> str:
    """自动检测输入路径"""
    logger = logging.getLogger(__name__)
    
    if stage == "multimodal":
        # 查找最新的多模态特征输出
        base_paths = [
            "/Volumes/home/raw_data/image_features_ple_parquet",
            "/Volumes/home/raw_data/image_features_chinese_clip_parquet",
            "/Volumes/home/raw_data/merged_features_parquet"
        ]
        
        for base_path in base_paths:
            if os.path.exists(base_path):
                # 找到最新的输出目录
                subdirs = []
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path):
                        subdirs.append(item_path)
                
                if subdirs:
                    latest_dir = max(subdirs, key=os.path.getmtime)
                    logger.info(f"Auto-detected input path: {latest_dir}")
                    return latest_dir
    
    # 如果自动检测失败，返回默认路径
    default_path = "/Volumes/home/raw_data/image_features_ple_parquet"
    logger.warning(f"Could not auto-detect input path, using default: {default_path}")
    return default_path


def generate_training_summary(results: Dict[str, Any], output_path: str):
    """生成训练摘要报告"""
    logger = logging.getLogger(__name__)
    
    # 创建摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_type': 'single_task' if isinstance(results, dict) and 'test_mse' in results else 'multi_task',
        'results': results
    }
    
    # 保存摘要
    summary_path = Path(output_path) / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📊 Training summary saved: {summary_path}")
    
    # 打印摘要
    logger.info("="*60)
    logger.info("📊 TRAINING SUMMARY")
    logger.info("="*60)
    
    if summary['training_type'] == 'single_task':
        if isinstance(results, dict) and 'test_mse' in results:
            # 单个模型结果
            logger.info(f"Model: {results.get('model_name', 'Unknown')}")
            logger.info(f"Test MSE: {results.get('test_mse', 0):.6f}")
            logger.info(f"Test R²: {results.get('test_r2', 0):.4f}")
            logger.info(f"Test Spearman: {results.get('test_spearman', 0):.4f}")
            logger.info(f"Training time: {results.get('training_time', 0):.2f}s")
        else:
            # 多个模型比较结果
            logger.info("Model Comparison Results:")
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'test_mse' in model_results:
                    logger.info(f"  {model_name}: MSE={model_results['test_mse']:.6f}, "
                               f"R²={model_results['test_r2']:.4f}, "
                               f"Spearman={model_results['test_spearman']:.4f}")
    else:
        # 多任务结果
        if 'summary' in results:
            logger.info(f"Multi-task Model Training")
            logger.info(f"Average Test MSE: {results['summary'].get('avg_test_mse', 0):.6f}")
            logger.info(f"Average Test R²: {results['summary'].get('avg_test_r2', 0):.4f}")
            logger.info(f"Number of tasks: {results['summary'].get('num_tasks', 0)}")
        
        # 各任务详细结果
        if 'tasks' in results:
            logger.info("Task-wise Results:")
            for task, task_results in results['tasks'].items():
                logger.info(f"  {task}: MSE={task_results.get('test_mse', 0):.6f}, "
                           f"R²={task_results.get('test_r2', 0):.4f}, "
                           f"Spearman={task_results.get('test_spearman', 0):.4f}")
    
    logger.info("="*60)
    
    return str(summary_path)


def main():
    parser = argparse.ArgumentParser(description='小红书CTR预估模型 - 统一模型训练入口')
    
    # 基本参数
    parser.add_argument('--task', choices=['single', 'multi'], required=True,
                       help='训练任务类型: single=单任务CTR, multi=多任务学习')
    parser.add_argument('--input', help='输入数据路径 (如不指定则自动检测)')
    parser.add_argument('--output', default='models',
                       help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--log-file', help='日志文件名')
    
    # 单任务参数
    single_group = parser.add_argument_group('单任务训练参数')
    single_group.add_argument('--model', default='DeepFM', 
                             choices=['PNN', 'DeepFM', 'WDL', 'DCN', 'xDeepFM', 'AutoInt', 'FiBiNET', 'CCPM'],
                             help='单任务模型类型')
    single_group.add_argument('--all-models', action='store_true',
                             help='训练所有单任务模型并比较')
    single_group.add_argument('--feature-config', default='all',
                             choices=['basic', 'clip_only', 'no_image', 'no_text', 'all', 'custom'],
                             help='特征配置')
    single_group.add_argument('--no-early-stopping', action='store_true',
                             help='禁用早停机制')
    
    # 自定义特征配置参数
    custom_group = parser.add_argument_group('自定义特征配置参数 (当feature-config=custom时使用)')
    custom_group.add_argument('--sparse-features', help='稀疏特征列表(逗号分隔)')
    custom_group.add_argument('--dense-features', help='密集特征列表(逗号分隔)')
    custom_group.add_argument('--use-clip-image', action='store_true', help='使用CLIP图像特征')
    custom_group.add_argument('--use-clip-text', action='store_true', help='使用CLIP文本特征')
    
    # 多任务参数
    multi_group = parser.add_argument_group('多任务训练参数')
    multi_group.add_argument('--multi-task-model', default='PLE',
                            choices=['PLE', 'MMOE', 'PNN_MMOE'],
                            help='多任务模型类型')
    multi_group.add_argument('--tasks',
                            help='任务列表，逗号分隔 (如: ctr,like_rate,comment_rate)')
    multi_group.add_argument('--shared-expert-num', type=int, default=4,
                            help='共享专家数量 (PLE)')
    multi_group.add_argument('--specific-expert-num', type=int, default=2,
                            help='任务特定专家数量 (PLE)')
    multi_group.add_argument('--num-levels', type=int, default=2,
                            help='PLE层数')
    multi_group.add_argument('--num-experts', type=int, default=4,
                            help='专家数量 (MMOE/PNN_MMOE)')
    multi_group.add_argument('--min-click', type=int, default=100,
                            help='最小点击阈值 (多任务)')
    
    # 特征处理参数
    feature_group = parser.add_argument_group('特征处理参数')
    feature_group.add_argument('--sample-size', type=int, help='随机采样文件数量')
    feature_group.add_argument('--label-column', default='ctr', help='标签列名')
    feature_group.add_argument('--use-pca', action='store_true',
                              help='使用PCA降维CLIP特征')
    feature_group.add_argument('--pca-components', type=int, default=256,
                              help='PCA组件数')
    feature_group.add_argument('--use-fp16', action='store_true', help='使用float16优化内存')
    feature_group.add_argument('--filter-zeros', action='store_true',
                              help='过滤零特征样本')
    feature_group.add_argument('--min-impression', type=int, default=5000,
                              help='最小曝光阈值')
    
    # 训练参数
    training_group = parser.add_argument_group('训练参数')
    training_group.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
    training_group.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    training_group.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    
    # 早停参数
    early_stopping_group = parser.add_argument_group('早停机制参数')
    early_stopping_group.add_argument('--early-stopping-patience', type=int, default=10,
                                     help='早停耐心值')
    early_stopping_group.add_argument('--early-stopping-monitor', default='loss',
                                     choices=['loss', 'val_loss', 'mae', 'val_mae'],
                                     help='早停监控指标')
    early_stopping_group.add_argument('--early-stopping-min-delta', type=float, default=0.0001,
                                     help='早停最小改善幅度')
    early_stopping_group.add_argument('--restore-best-weights', action='store_true', default=True,
                                     help='恢复最佳权重')
    early_stopping_group.add_argument('--save-best-model', action='store_true',
                                     help='保存最佳模型')
    
    # 额外训练参数
    advanced_group = parser.add_argument_group('高级训练参数')
    advanced_group.add_argument('--label-normalization',
                               choices=['none', 'standard', 'minmax', 'robust'],
                               default='none',
                               help='标签归一化方法')
    advanced_group.add_argument('--device',
                               choices=['auto', 'cpu', 'cuda', 'mps'],
                               default='auto',
                               help='训练设备选择')

    # 可视化参数
    visualization_group = parser.add_argument_group('可视化参数 (验证集散点图)')
    visualization_group.add_argument('--enable-visualization', action='store_true', default=True,
                                    help='启用验证集散点图可视化（默认启用）')
    visualization_group.add_argument('--no-visualization', dest='enable_visualization',
                                    action='store_false',
                                    help='禁用验证集散点图可视化')
    visualization_group.add_argument('--scatter-sample-size', type=int, default=1000,
                                    help='散点图采样数量（默认1000）')
    visualization_group.add_argument('--plot-dpi', type=int, default=150,
                                    help='散点图DPI质量（默认150）')

    args = parser.parse_args()
    
    # 验证参数
    if args.epochs <= 0:
        parser.error("Epochs must be positive")
    if args.batch_size <= 0:
        parser.error("Batch size must be positive")
    
    # 设置日志
    log_file = setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # 自动检测输入路径
    if not args.input:
        args.input = auto_detect_input_path()
    
    logger.info("="*80)
    logger.info("🎯 小红书CTR预估模型 - 模型训练")
    logger.info("="*80)
    logger.info(f"训练类型: {args.task}")
    if args.task == 'single':
        if args.all_models:
            logger.info(f"模型: 所有单任务模型")
        else:
            logger.info(f"模型: {args.model}")
    else:
        logger.info(f"模型: {args.multi_task_model}")
        if args.tasks:
            logger.info(f"任务: {args.tasks}")
    logger.info(f"输入路径: {args.input}")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*80)
    
    try:
        # 验证输入路径
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path not found: {args.input}")
        
        # 根据任务类型运行相应的训练
        if args.task == 'single':
            results = train_single_task(args)
        elif args.task == 'multi':
            results = train_multi_task(args)
        else:
            raise ValueError(f"Unknown task type: {args.task}")
        
        # 生成训练摘要
        summary_path = generate_training_summary(results, args.output)
        
        logger.info("="*80)
        logger.info("🎉 模型训练完成!")
        logger.info(f"摘要报告: {summary_path}")
        logger.info(f"完成时间: {datetime.now()}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()