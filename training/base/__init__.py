#!/usr/bin/env python3
"""
Training Base Module

统一的训练基础框架，提供多任务和单任务trainer的公共组件：

多任务组件：
- BaseMTLTrainer: 统一MTL训练流程模板
- MultiTaskConfig: 多任务配置管理系统
- BaseMTLDataProcessor: MTL数据处理器
- BaseMTLFeatureProcessor: MTL特征处理器
- BaseMTLEvaluator: MTL评估器
- ModelFactory: 模型工厂

单任务组件：
- BaseSingleTaskTrainer: 统一单任务训练流程模板
- BaseSingleTaskDataProcessor: 单任务数据处理器
- BaseSingleTaskFeatureProcessor: 单任务特征处理器
- BaseSingleTaskEvaluator: 单任务评估器
- FeatureSelector: 特征选择器

目标：
1. 消除代码重复
2. 统一训练流程
3. 补全缺失特性
4. 提升可维护性
"""

# 配置管理
from .base_config import (
    BaseTrainingConfig,
    SingleTaskConfig,
    MultiTaskConfig,
    FeatureConfigType,
    LabelNormalizationType,
    EarlyStoppingMonitor,
    create_config_from_args
)

# 数据处理
from .data_processor import BaseMTLDataProcessor

# 特征处理
from .feature_processor import BaseMTLFeatureProcessor

# 评估器
from .evaluator import BaseMTLEvaluator

# 模型工厂
from .model_factory import (
    ModelFactory,
    SingleTaskModel,
    MultiTaskModel,
    create_model
)

# 训练器基类
from .base_trainer import BaseMTLTrainer

# PNN-MMOE模型
from .pnn_mmoe_model import PNN_MMOE

# 单任务组件
from .single_task_data_processor import BaseSingleTaskDataProcessor
from .single_task_feature_processor import BaseSingleTaskFeatureProcessor
from .single_task_evaluator import BaseSingleTaskEvaluator
from .single_task_trainer import BaseSingleTaskTrainer
from .feature_selector import FeatureSelector

__all__ = [
    # 配置类
    'BaseTrainingConfig',
    'SingleTaskConfig',
    'MultiTaskConfig',
    'FeatureConfigType',
    'LabelNormalizationType',
    'EarlyStoppingMonitor',
    'create_config_from_args',
    
    # 数据处理
    'BaseMTLDataProcessor',
    
    # 特征处理
    'BaseMTLFeatureProcessor',
    
    # 评估器
    'BaseMTLEvaluator',
    
    # 模型工厂
    'ModelFactory',
    'SingleTaskModel',
    'MultiTaskModel',
    'create_model',
    
    # 训练器基类
    'BaseMTLTrainer',
    
    # 自定义模型
    'PNN_MMOE',
    
    # 单任务组件
    'BaseSingleTaskDataProcessor',
    'BaseSingleTaskFeatureProcessor',
    'BaseSingleTaskEvaluator',
    'BaseSingleTaskTrainer',
    'FeatureSelector',
]