"""
单任务CTR预估模型训练器

基于BaseTrainer重构，保持与run_local_training_model.py的逻辑一致。
"""

from .ctr_trainer import (
    CTRTrainer,
    FeatureSelector,
    create_ctr_config,
    train_single_model,
    train_all_models
)

__all__ = [
    'CTRTrainer',
    'FeatureSelector', 
    'create_ctr_config',
    'train_single_model',
    'train_all_models',
]