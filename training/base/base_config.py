#!/usr/bin/env python3
"""
统一训练配置管理系统

提供所有trainer的配置基类，补全所有原始脚本中的配置选项：
- 数据配置：sample-size, filter-zeros, min-impression
- 特征配置：use-pca, pca-components, feature-config, label-normalization  
- 训练配置：epochs, batch-size, learning-rate, use-fp16
- 早停配置：early-stopping, patience, monitor, min-delta, restore-best-weights
- 模型配置：model-name, save-best-model
- 多任务配置：tasks, task-weights

特性覆盖：
✅ sample-size: 数据文件采样
✅ filter-zeros: CLIP特征零值过滤
✅ min-impression: 最小曝光阈值
✅ use-pca: PCA降维支持
✅ use-fp16: 半精度浮点数
✅ label-normalization: 标签归一化
✅ 精细早停控制: monitor, min-delta, restore-weights
✅ save-best-model: ModelCheckpoint
"""

import os
import psutil
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class LabelNormalizationType(Enum):
    """标签归一化类型"""
    NONE = "none"
    STANDARD = "standard"     # StandardScaler (z-score)
    MINMAX = "minmax"        # MinMaxScaler [0,1]
    ROBUST = "robust"        # RobustScaler (median, IQR)


class FeatureConfigType(Enum):
    """特征配置预设类型"""
    ALL = "all"              # 所有特征
    BASIC = "basic"          # 基础特征(无CLIP)
    CLIP_ONLY = "clip_only"  # 仅CLIP特征
    NO_IMAGE = "no_image"    # 无图像特征
    NO_TEXT = "no_text"      # 无文本特征
    CUSTOM = "custom"        # 自定义配置


class EarlyStoppingMonitor(Enum):
    """早停监控指标"""
    VAL_LOSS = "val_loss"
    VAL_MAE = "val_mae"
    LOSS = "loss"
    MAE = "mae"


@dataclass
class BaseTrainingConfig(ABC):
    """训练配置基类
    
    包含所有trainer的通用配置选项，对应原始脚本的所有参数
    """
    
    # ============ 数据配置 ============
    input_path: str                                    # 输入数据路径
    output_path: str = "models/training_output"        # 输出路径
    sample_size: Optional[int] = None                  # 🆕 数据文件采样数量
    
    # ============ 数据过滤配置 ============  
    filter_zeros: bool = True                          # 🆕 过滤零CLIP特征
    min_impression: int = 5000                         # 🆕 最小曝光阈值
    
    # ============ 特征配置 ============
    feature_config: FeatureConfigType = FeatureConfigType.ALL  # 特征配置预设
    use_pca: bool = False                              # 🆕 PCA降维
    pca_components: int = 256                          # 🆕 PCA组件数
    label_normalization: LabelNormalizationType = LabelNormalizationType.STANDARD  # 🆕 标签归一化
    
    # ============ 自定义特征配置 ============
    sparse_features: Optional[List[str]] = None        # 自定义稀疏特征
    dense_features: Optional[List[str]] = None         # 自定义密集特征
    use_clip_image: bool = True                        # 使用CLIP图像特征
    use_clip_text: bool = True                         # 使用CLIP文本特征
    
    # ============ 训练配置 ============
    epochs: int = 20                                   # 训练轮数
    batch_size: int = 256                              # 批次大小
    learning_rate: float = 0.001                       # 学习率
    validation_split: float = 0.2                      # 验证集比例
    use_fp16: bool = False                             # 🆕 半精度浮点数
    
    # ============ 早停配置 ============
    use_early_stopping: bool = True                    # 🆕 启用早停
    early_stopping_patience: int = 10                  # 🆕 早停耐心值
    early_stopping_monitor: EarlyStoppingMonitor = EarlyStoppingMonitor.VAL_LOSS  # 🆕 早停监控指标
    early_stopping_min_delta: float = 0.0001           # 🆕 早停最小改善值
    restore_best_weights: bool = True                  # 🆕 恢复最佳权重
    save_best_model: bool = True                       # 🆕 保存最佳模型
    
    # ============ 设备配置 ============
    device: Optional[str] = None                       # 设备(auto/cpu/cuda/mps)
    
    # ============ 日志配置 ============
    verbose: int = 1                                   # 训练日志详细程度
    log_dir: Optional[str] = None                      # 日志输出目录
    
    def __post_init__(self):
        """初始化后处理"""
        self._setup_paths()
        self._setup_device()
        self._validate_config()
    
    def _setup_paths(self):
        """设置路径"""
        # 确保输出路径存在
        self.output_path = Path(self.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志目录
        if self.log_dir is None:
            self.log_dir = self.output_path / "logs"
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        """自动设置设备"""
        if self.device is None or self.device == "auto":
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
    
    def _validate_config(self):
        """验证配置有效性"""
        # 验证路径
        if not Path(self.input_path).exists():
            raise ValueError(f"Input path does not exist: {self.input_path}")
        
        # 验证采样大小
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        
        # 验证PCA组件数
        if self.use_pca and self.pca_components <= 0:
            raise ValueError("pca_components must be positive")
        
        # 验证训练参数
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        
        # 验证早停参数
        if self.use_early_stopping:
            if self.early_stopping_patience <= 0:
                raise ValueError("early_stopping_patience must be positive")
            if self.early_stopping_min_delta < 0:
                raise ValueError("early_stopping_min_delta must be non-negative")
    
    def get_model_save_path(self, model_name: str) -> Path:
        """获取模型保存路径"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_path / f"model_{model_name}_{timestamp}.pth"
    
    def get_results_save_path(self, model_name: str) -> Path:
        """获取结果保存路径"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_path / f"results_{model_name}_{timestamp}.json"
    
    def get_feature_config_dict(self) -> Dict[str, Any]:
        """获取特征配置字典"""
        return {
            'feature_config': self.feature_config.value,
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'use_clip_image': self.use_clip_image,
            'use_clip_text': self.use_clip_text,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'label_normalization': self.label_normalization.value
        }
    
    def get_early_stopping_config(self) -> Dict[str, Any]:
        """获取早停配置字典"""
        if not self.use_early_stopping:
            return {}
        
        return {
            'monitor': self.early_stopping_monitor.value,
            'patience': self.early_stopping_patience,
            'min_delta': self.early_stopping_min_delta,
            'restore_best_weights': self.restore_best_weights,
            'verbose': 1 if self.verbose > 0 else 0
        }
    
    def get_training_config_dict(self) -> Dict[str, Any]:
        """获取训练配置字典"""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'use_fp16': self.use_fp16,
            'device': self.device,
            'verbose': self.verbose
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于保存配置"""
        return {
            # 数据配置
            'input_path': str(self.input_path),
            'output_path': str(self.output_path),
            'sample_size': self.sample_size,
            'filter_zeros': self.filter_zeros,
            'min_impression': self.min_impression,
            
            # 特征配置
            'feature_config': self.feature_config.value,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'label_normalization': self.label_normalization.value,
            'sparse_features': self.sparse_features,
            'dense_features': self.dense_features,
            'use_clip_image': self.use_clip_image,
            'use_clip_text': self.use_clip_text,
            
            # 训练配置
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'use_fp16': self.use_fp16,
            'device': self.device,
            
            # 早停配置
            'use_early_stopping': self.use_early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_monitor': self.early_stopping_monitor.value,
            'early_stopping_min_delta': self.early_stopping_min_delta,
            'restore_best_weights': self.restore_best_weights,
            'save_best_model': self.save_best_model
        }


@dataclass
class SingleTaskConfig(BaseTrainingConfig):
    """单任务训练配置
    
    支持所有原始run_local_training_model.py的配置选项
    """
    
    # ============ 单任务特有配置 ============
    model_name: str = "DeepFM"                         # 模型名称
    label_column: str = "ctr"                          # 标签列名
    
    # ============ 支持的模型列表 ============
    SUPPORTED_MODELS = [
        "PNN", "DeepFM", "WDL", "DCN", "xDeepFM", 
        "AutoInt", "FiBiNET", "CCPM", "FNN", "DNN"    # 🆕 补全原始支持的模型
    ]
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_single_task_config()
    
    def _validate_single_task_config(self):
        """验证单任务配置"""
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}. "
                           f"Supported models: {self.SUPPORTED_MODELS}")


@dataclass 
class MultiTaskConfig(BaseTrainingConfig):
    """多任务训练配置
    
    支持所有MTL模型的配置选项：PLE、MMOE、PNN-MMOE
    """
    
    # ============ 多任务特有配置 ============
    tasks: List[str] = field(default_factory=lambda: [
        "ctr", "like_rate", "fav_rate", "comment_rate", "share_rate",
        "follow_rate", "interaction_rate", "ces_rate", "impression", "sort_score"
    ])                                                 # 任务列表
    task_weights: Optional[Dict[str, float]] = None    # 🆕 任务权重
    
    # ============ 多任务模型配置 ============
    model_type: str = "PLE"                           # 多任务模型类型
    
    # ============ PLE专用配置 ============
    shared_expert_num: int = 2                        # 共享专家数量
    specific_expert_num: int = 2                      # 任务特定专家数量  
    num_levels: int = 2                               # PLE层数
    
    # ============ MMOE专用配置 ============  
    num_experts: int = 3                              # 🆕 MMOE专家数量
    
    # ============ PNN-MMOE专用配置 ============
    use_inner_product: bool = True                    # 🆕 使用PNN内积特征交互
    use_outter_product: bool = False                  # 🆕 使用PNN外积特征交互
    
    # ============ 通用网络架构配置 ============
    expert_dims: List[int] = field(default_factory=lambda: [128, 64])  # 专家网络维度
    gate_dims: List[int] = field(default_factory=lambda: [32])         # 门控网络维度
    tower_dims: List[int] = field(default_factory=lambda: [64, 32])    # 塔网络维度
    
    # ============ 正则化配置 ============
    dropout: float = 0.1                             # Dropout率
    l2_reg_embedding: float = 1e-5                   # 嵌入层L2正则化
    l2_reg_dnn: float = 0                            # DNN层L2正则化
    
    # ============ 支持的多任务模型 ============
    SUPPORTED_MODELS = ["PLE", "MMOE", "PNN_MMOE"]
    
    # ============ 预定义任务配置 ============
    TASK_DEFINITIONS = {
        'ctr': {
            'description': '点击率 (click_num/imp_num)', 
            'target_col': 'ctr',
            'calculation': lambda df: df['click_num'] / df['imp_num'].clip(lower=1)
        },
        'like_rate': {
            'description': '点赞率 (like_num/click_num)',
            'target_col': 'like_rate', 
            'calculation': lambda df: df['like_num'] / df['click_num'].clip(lower=1)
        },
        'fav_rate': {
            'description': '收藏率 (fav_num/click_num)',
            'target_col': 'fav_rate',
            'calculation': lambda df: df['fav_num'] / df['click_num'].clip(lower=1)
        },
        'comment_rate': {
            'description': '评论率 (cmt_num/click_num)', 
            'target_col': 'comment_rate',
            'calculation': lambda df: df['cmt_num'] / df['click_num'].clip(lower=1)
        },
        'share_rate': {
            'description': '分享率 (share_num/click_num)',
            'target_col': 'share_rate',
            'calculation': lambda df: df['share_num'] / df['click_num'].clip(lower=1)
        },
        'follow_rate': {
            'description': '加粉率 (follow_from_discovery_num/click_num)',
            'target_col': 'follow_rate',
            'calculation': lambda df: df['follow_from_discovery_num'] / df['click_num'].clip(lower=1)
        },
        'interaction_rate': {
            'description': '互动率 ((like+fav+cmt+share+follow)/click_num)',
            'target_col': 'interaction_rate',
            'calculation': lambda df: (df['like_num'] + df['fav_num'] + df['cmt_num'] + 
                                     df['share_num'] + df['follow_from_discovery_num']) / df['click_num'].clip(lower=1)
        },
        'ces_rate': {
            'description': 'CES评分率 ((1*like+1*fav+4*cmt+4*share+4*follow)/click_num)',
            'target_col': 'ces_rate', 
            'calculation': lambda df: (1*df['like_num'] + 1*df['fav_num'] + 4*df['cmt_num'] + 
                                     4*df['share_num'] + 4*df['follow_from_discovery_num']) / df['click_num'].clip(lower=1)
        },
        'impression': {
            'description': '曝光数预测 (log(imp_num))',
            'target_col': 'impression',
            'calculation': lambda df: np.log1p(df['imp_num'])
        },
        'sort_score': {
            'description': '排序分数预测 (sort_score2)',
            'target_col': 'sort_score',
            'calculation': lambda df: df['sort_score2'].fillna(0)
        }
    }
    
    def __post_init__(self):
        super().__post_init__()
        self._validate_multi_task_config()
    
    def _validate_multi_task_config(self):
        """验证多任务配置"""
        # 验证模型类型
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_type}. "
                           f"Supported models: {self.SUPPORTED_MODELS}")
        
        # 验证任务列表
        valid_tasks = set(self.TASK_DEFINITIONS.keys())
        invalid_tasks = set(self.tasks) - valid_tasks
        if invalid_tasks:
            raise ValueError(f"Invalid tasks: {invalid_tasks}. "
                           f"Valid tasks: {list(valid_tasks)}")
        
        # 验证任务权重
        if self.task_weights is not None:
            invalid_weight_tasks = set(self.task_weights.keys()) - set(self.tasks)
            if invalid_weight_tasks:
                raise ValueError(f"Task weights contain invalid tasks: {invalid_weight_tasks}")
        
        # 验证模型特定配置
        if self.model_type == "PLE":
            if self.shared_expert_num <= 0:
                raise ValueError("shared_expert_num must be positive")
            if self.specific_expert_num <= 0:
                raise ValueError("specific_expert_num must be positive")
            if self.num_levels <= 0:
                raise ValueError("num_levels must be positive")
        
        elif self.model_type == "MMOE":
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
        
        elif self.model_type == "PNN_MMOE":
            if self.num_experts <= 0:
                raise ValueError("num_experts must be positive")
            # PNN参数无需验证，布尔值总是有效的
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """获取任务信息"""
        if task_name not in self.TASK_DEFINITIONS:
            raise ValueError(f"Unknown task: {task_name}")
        return self.TASK_DEFINITIONS[task_name]
    
    def get_all_task_columns(self) -> List[str]:
        """获取所有任务的目标列名"""
        return [self.TASK_DEFINITIONS[task]['target_col'] for task in self.tasks]
    
    def get_task_weight(self, task_name: str) -> float:
        """获取任务权重"""
        if self.task_weights is None:
            return 1.0  # 默认权重
        return self.task_weights.get(task_name, 1.0)


def create_config_from_args(args, config_type: str = "single") -> BaseTrainingConfig:
    """从命令行参数创建配置对象
    
    Args:
        args: 命令行参数对象
        config_type: 配置类型 ("single" 或 "multi")
    
    Returns:
        配置对象
    """
    # 解析枚举类型
    feature_config = FeatureConfigType(getattr(args, 'feature_config', 'all'))
    label_normalization = LabelNormalizationType(getattr(args, 'label_normalization', 'standard'))
    early_stopping_monitor = EarlyStoppingMonitor(getattr(args, 'early_stopping_monitor', 'val_loss'))
    
    # 基础配置参数
    base_kwargs = {
        'input_path': getattr(args, 'input_path'),
        'output_path': getattr(args, 'output_path', 'models/training_output'),
        'sample_size': getattr(args, 'sample_size', None),
        'filter_zeros': getattr(args, 'filter_zeros', True),
        'min_impression': getattr(args, 'min_impression', 5000),
        'feature_config': feature_config,
        'use_pca': getattr(args, 'use_pca', False),
        'pca_components': getattr(args, 'pca_components', 256),
        'label_normalization': label_normalization,
        'sparse_features': getattr(args, 'sparse_features', None),
        'dense_features': getattr(args, 'dense_features', None),
        'use_clip_image': getattr(args, 'use_clip_image', True),
        'use_clip_text': getattr(args, 'use_clip_text', True),
        'epochs': getattr(args, 'epochs', 20),
        'batch_size': getattr(args, 'batch_size', 256),
        'learning_rate': getattr(args, 'learning_rate', 0.001),
        'validation_split': getattr(args, 'validation_split', 0.2),
        'use_fp16': getattr(args, 'use_fp16', False),
        'use_early_stopping': getattr(args, 'use_early_stopping', True),
        'early_stopping_patience': getattr(args, 'early_stopping_patience', 10),
        'early_stopping_monitor': early_stopping_monitor,
        'early_stopping_min_delta': getattr(args, 'early_stopping_min_delta', 0.0001),
        'restore_best_weights': getattr(args, 'restore_best_weights', True),
        'save_best_model': getattr(args, 'save_best_model', True),
        'device': getattr(args, 'device', None),
        'verbose': getattr(args, 'verbose', 1)
    }
    
    if config_type == "single":
        # 单任务特有参数
        single_kwargs = {
            'model_name': getattr(args, 'model_name', 'DeepFM'),
            'label_column': getattr(args, 'label_column', 'ctr')
        }
        return SingleTaskConfig(**base_kwargs, **single_kwargs)
    
    elif config_type == "multi":
        # 多任务特有参数
        tasks = getattr(args, 'tasks', None)
        if isinstance(tasks, str):
            tasks = tasks.split(',')
        elif tasks is None:
            tasks = list(MultiTaskConfig.TASK_DEFINITIONS.keys())
        
        multi_kwargs = {
            'tasks': tasks,
            'task_weights': getattr(args, 'task_weights', None),
            'model_type': getattr(args, 'model_type', 'PLE'),
            # PLE专用参数
            'shared_expert_num': getattr(args, 'shared_expert_num', 2),
            'specific_expert_num': getattr(args, 'specific_expert_num', 2),
            'num_levels': getattr(args, 'num_levels', 2),
            # MMOE和PNN-MMOE专用参数
            'num_experts': getattr(args, 'num_experts', 3),
            # PNN-MMOE专用参数
            'use_inner_product': getattr(args, 'use_inner_product', True),
            'use_outter_product': getattr(args, 'use_outter_product', False),
            # 通用网络架构参数
            'expert_dims': getattr(args, 'expert_dims', [128, 64]),
            'gate_dims': getattr(args, 'gate_dims', [32]),
            'tower_dims': getattr(args, 'tower_dims', [64, 32]),
            # 正则化参数
            'dropout': getattr(args, 'dropout', 0.1),
            'l2_reg_embedding': getattr(args, 'l2_reg_embedding', 1e-5),
            'l2_reg_dnn': getattr(args, 'l2_reg_dnn', 0)
        }
        return MultiTaskConfig(**base_kwargs, **multi_kwargs)
    
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


# 为了向前兼容导入numpy
import numpy as np