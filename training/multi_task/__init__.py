"""
多任务CTR预估模型训练器

基于统一base class架构的多任务学习训练器集合。
包含PLE、MMOE、PNN-MMOE三种多任务学习架构。
"""

from .ple_trainer import PLETrainer
from .mmoe_trainer import MMOETrainer
from .pnn_mmoe_trainer import PNNMMOETrainer

__all__ = [
    'PLETrainer',
    'MMOETrainer',
    'PNNMMOETrainer'
]
