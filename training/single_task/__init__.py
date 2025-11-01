"""
单任务CTR预估模型训练器

基于统一base class架构的轻量级单任务训练器。
"""

from .ctr_trainer import LocalCTRTrainer

__all__ = ['LocalCTRTrainer']
