"""
多任务CTR预估模型训练器
包含MMOE、PLE、PNN+MMOE等多任务学习方法

基于BaseTrainer重构，保持与原始脚本逻辑一致。
"""

# PLE训练器
from .ple_trainer import (
    PLETrainer,
    TaskConfig,
    PLEFeatureProcessor,
    train_ple_model
)

# MMOE训练器
from .mmoe_trainer import (
    MMOETrainer,
    TaskDefinition,
    create_mmoe_config,
    train_mmoe_model
)

# PNN+MMOE训练器
from .pnn_mmoe_trainer import PNNMMOETrainer

__all__ = [
    # PLE
    'PLETrainer',
    'TaskConfig',
    'PLEFeatureProcessor',
    'train_ple_model',
    
    # MMOE
    'MMOETrainer',
    'TaskDefinition',
    'create_mmoe_config', 
    'train_mmoe_model',
    
    # PNN+MMOE
    'PNNMMOETrainer',
]