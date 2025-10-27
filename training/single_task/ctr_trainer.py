#!/usr/bin/env python3
"""
重构后的本地CTR模型训练脚本
基于单任务训练基类架构，大幅简化代码结构

支持PNN、FNN、DeepFM、WDL等模型的训练
可从image_features_parquet或merged_features_parquet读取数据

使用示例:
----------
# 1. 训练单个模型 (PNN)
python ctr_trainer_refactored.py \
    --input-path /Volumes/home/raw_data/image_features_chinese_clip_parquet \
    --model PNN \
    --epochs 10

# 2. 训练所有模型并生成对比报告
python ctr_trainer_refactored.py \
    --input-path /Volumes/home/raw_data/image_features_chinese_clip_parquet \
    --all-models \
    --epochs 10

# 3. 使用不同的特征配置
python ctr_trainer_refactored.py \
    --input-path /Volumes/home/raw_data/image_features_chinese_clip_parquet \
    --model DeepFM \
    --feature-config clip_only

支持的模型:
-----------
- PNN: Product-based Neural Network
- DeepFM: Deep Factorization Machine
- WDL: Wide & Deep Learning
- DCN: Deep & Cross Network
- xDeepFM: eXtreme Deep Factorization Machine
- FiBiNET: Feature Importance and Bilinear feature Interaction NETwork
- AutoInt: Automatic Feature Interaction Learning
- CCPM: Convolutional Click Prediction Model

特征配置预设:
-------------
- basic: 基础特征 (无CLIP)
- clip_only: 仅CLIP特征
- no_image: 无图像特征
- no_text: 无文本特征
- all: 所有特征
- default: 默认配置
- custom: 自定义配置
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import List
import torch

# 导入deepctr-torch模型
from deepctr_torch.models import DeepFM, PNN, WDL, DCN, xDeepFM, FiBiNET, AutoInt, CCPM

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入基类和工具
from training.base.single_task_trainer import BaseSingleTaskTrainer
from training.base.feature_selector import FeatureSelector

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if os.environ.get('DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalCTRTrainer(BaseSingleTaskTrainer):
    """本地CTR模型训练器（重构版本）
    
    基于BaseSingleTaskTrainer的轻量级实现，专注于DeepCTR模型的创建和配置。
    大部分通用逻辑已抽象到基类中，此类仅保留模型特定的实现。
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = ['PNN', 'DeepFM', 'WDL', 'DCN', 'xDeepFM', 'FiBiNET', 'AutoInt', 'CCPM']
    
    def __init__(self, **kwargs):
        """初始化LocalCTRTrainer
        
        接受所有BaseSingleTaskTrainer的参数，并处理特征配置。
        """
        # 处理特征配置
        feature_config_name = kwargs.pop('feature_config', 'default')
        if feature_config_name == 'custom':
            # 处理自定义配置
            feature_config = FeatureSelector.create_custom_config(
                sparse_features=kwargs.pop('sparse_features', None),
                dense_features=kwargs.pop('dense_features', None),
                use_clip_image=kwargs.pop('use_clip_image', False),
                use_clip_text=kwargs.pop('use_clip_text', False)
            )
        else:
            # 使用预设配置
            feature_config = FeatureSelector.get_feature_config(feature_config_name)
        
        kwargs['feature_config'] = feature_config
        
        # 调用父类初始化
        super().__init__(**kwargs)
        
        logger.info(f"🚀 LocalCTRTrainer initialized with {feature_config_name} feature config")
        logger.info(f"📊 Config: {feature_config.get('description', 'Unknown')}")
    
    def create_model(self, feature_columns: List) -> torch.nn.Module:
        """创建DeepCTR模型（实现抽象方法）
        
        Args:
            feature_columns: DeepCTR特征列定义
            
        Returns:
            初始化的PyTorch模型
        """
        logger.info(f"🏗️ Creating {self.model_type} model with deepctr-torch...")
        
        device = torch.device(self.device)
        
        # 根据模型类型创建相应的模型
        if self.model_type == 'DEEPFM':
            model = DeepFM(feature_columns, feature_columns, task='regression', device=device)
        elif self.model_type == 'PNN':
            model = PNN(feature_columns, task='regression', device=device)
        elif self.model_type == 'WDL':
            model = WDL(feature_columns, feature_columns, task='regression', device=device)
        elif self.model_type == 'DCN':
            model = DCN(feature_columns, task='regression', device=device)
        elif self.model_type == 'XDEEPFM':
            model = xDeepFM(feature_columns, feature_columns, task='regression', device=device)
        elif self.model_type == 'FIBINET':
            model = FiBiNET(feature_columns, feature_columns, task='regression', device=device)
        elif self.model_type == 'AUTOINT':
            model = AutoInt(feature_columns, task='regression', device=device)
        elif self.model_type == 'CCPM':
            model = CCPM(feature_columns, task='regression', device=device)
        elif self.model_type in ['FNN', 'DNN']:
            # FNN不在deepctr-torch中，使用PNN替代
            logger.warning(f"Model {self.model_type} not in deepctr-torch, using PNN instead")
            model = PNN(feature_columns, task='regression', device=device)
            self.model_type = 'PNN'  # 更新模型类型以便后续处理
        else:
            logger.warning(f"Model {self.model_type} not supported, using DeepFM")
            model = DeepFM(feature_columns, feature_columns, task='regression', device=device)
            self.model_type = 'DEEPFM'
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ {self.model_type} model created successfully")
        logger.info(f"📊 Model parameters:")
        logger.info(f"   - Total: {total_params:,}")
        logger.info(f"   - Trainable: {trainable_params:,}")
        logger.info(f"   - Device: {device}")
        
        return model
    
    def get_model_config(self) -> dict:
        """获取模型配置（覆盖基类方法）"""
        return {
            'model_type': self.model_type,
            'device': self.device,
            'task': 'regression',
            'feature_config': self.feature_config,
            'framework': 'deepctr-torch'
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Local CTR Model Training (Refactored)")
    
    # 基础参数
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Input file or directory path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PNN",
        choices=LocalCTRTrainer.SUPPORTED_MODELS,
        help="Model type (default: PNN)"
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default="default",
        choices=FeatureSelector.get_available_configs(),
        help="Feature configuration preset (default: default)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of files to sample from directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/single_task",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="ctr",
        help="Label column name (default: ctr)"
    )
    
    # 自定义特征选择参数（当feature-config为custom时使用）
    parser.add_argument(
        "--sparse-features",
        type=str,
        default=None,
        help="Comma-separated list of sparse features (for custom config)"
    )
    parser.add_argument(
        "--dense-features",
        type=str,
        default=None,
        help="Comma-separated list of dense features (for custom config)"
    )
    parser.add_argument(
        "--use-clip-image",
        action="store_true",
        help="Use CLIP image features (for custom config)"
    )
    parser.add_argument(
        "--use-clip-text",
        action="store_true",
        help="Use CLIP text features (for custom config)"
    )
    
    # 数据处理参数
    parser.add_argument(
        "--filter-zeros",
        action="store_true",
        help="Filter out samples with all-zero CLIP features (recommended)"
    )
    parser.add_argument(
        "--min-impression",
        type=int,
        default=5000,
        help="Minimum impression threshold for filtering (default: 5000, set to 0 to disable)"
    )
    
    # 特征处理参数
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Apply PCA dimension reduction to CLIP features"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=256,
        help="Number of PCA components (default: 256)"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use float16 data type to reduce memory usage"
    )
    
    # 训练参数
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size ratio (default: 0.2)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set size ratio (default: 0.1)"
    )
    
    # 早停机制相关参数
    parser.add_argument(
        "--use-early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping to prevent overfitting (default: True)"
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience - number of epochs without improvement (default: 10)"
    )
    parser.add_argument(
        "--early-stopping-monitor",
        type=str,
        default="loss",
        choices=["val_loss", "val_mae", "loss", "mae"],
        help="Metric to monitor for early stopping (default: loss)"
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0001,
        help="Minimum improvement required to qualify as improvement (default: 0.0001)"
    )
    parser.add_argument(
        "--restore-best-weights",
        action="store_true",
        default=True,
        help="Restore best weights when early stopping is triggered (default: True)"
    )
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save the best model during training using ModelCheckpoint"
    )
    
    # 模型对比
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train and compare all supported models"
    )
    
    args = parser.parse_args()
    
    # 处理早停参数逻辑
    if args.no_early_stopping:
        use_early_stopping = False
    else:
        use_early_stopping = args.use_early_stopping
    
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
        input_path=args.input_path,
        model_type=args.model,
        feature_config=args.feature_config,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
        label_column=args.label_column,
        # 数据处理参数
        filter_zeros=args.filter_zeros,
        min_impression_threshold=args.min_impression,
        # 特征处理参数
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        use_fp16=args.use_fp16,
        # 训练参数
        use_early_stopping=use_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_monitor=args.early_stopping_monitor,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_weights=args.restore_best_weights,
        save_best_model=args.save_best_model,
        # 自定义配置参数
        **custom_config_args
    )
    
    # 运行训练
    if args.all_models:
        # 训练所有模型并对比
        trainer.run_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:
        # 训练单个模型
        trainer.run(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            test_size=args.test_size,
            val_size=args.val_size
        )


if __name__ == "__main__":
    main()