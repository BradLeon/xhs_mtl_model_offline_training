#!/usr/bin/env python3
"""
多任务学习训练基类

提供所有MTL训练器的统一训练框架：
- 8步统一训练流程
- 组合数据处理、特征处理、评估器
- 抽象模型创建接口
- 统一的训练和编译逻辑
- 完整的训练生命周期管理

消除MTL训练脚本中的核心逻辑重复
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from deepctr_torch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 导入基础组件
from .data_processor import BaseMTLDataProcessor
from .feature_processor import BaseMTLFeatureProcessor
from .evaluator import BaseMTLEvaluator
from .model_factory import ModelFactory

# 导入配置和预处理模块
from ..base.base_config import MultiTaskConfig
from src.models.loss_functions import create_multitask_loss_function, detect_tasks_with_missing_values
from src.models.preprocessing import MultiTaskLabelNormalizer

logger = logging.getLogger(__name__)


class BaseMTLTrainer(ABC):
    """多任务学习训练基类
    
    提供统一的8步训练流程和通用训练逻辑，子类只需实现模型特定的配置。
    
    统一训练流程：
    1. 加载数据
    2. 工程化目标变量
    3. 准备特征
    4. 准备目标变量
    5. 创建模型
    6. 编译模型
    7. 训练模型
    8. 保存结果
    """
    
    def __init__(self, config: MultiTaskConfig):
        """初始化MTL训练器
        
        Args:
            config: 多任务训练配置
        """
        self.config = config
        
        # 初始化组件
        self.data_processor = BaseMTLDataProcessor(
            input_path=config.input_path,
            sample_size=config.sample_size,
            min_impression_threshold=config.min_impression,
            tasks=config.tasks
        )
        
        self.feature_processor = BaseMTLFeatureProcessor(
            filter_zeros=config.filter_zeros,
            use_pca=config.use_pca,
            pca_components=config.pca_components
        )
        
        self.evaluator = BaseMTLEvaluator(
            task_names=config.tasks,
            task_column_mapping=self.data_processor.available_tasks,
            output_path=config.output_path,
            use_label_normalization=(config.label_normalization.value != 'none'),
            label_normalizer=None  # 将在训练时初始化
        )
        
        self.model_factory = ModelFactory(config)
        
        # 任务权重
        self.task_weights = config.task_weights or self._get_default_task_weights()
        
        # 训练状态
        self.model = None
        self.label_normalizer = None
        self.data = None
        
        logger.info(f"Initialized {self.__class__.__name__} for {len(config.tasks)} tasks")
        logger.info(f"Tasks: {config.tasks}")
        logger.info(f"Model type: {config.model_type}")
    
    def _get_default_task_weights(self) -> Dict[str, float]:
        """获取默认任务权重"""
        return {
            'ctr': 1.0,
            'like_rate': 2.0,
            'fav_rate': 3.0,
            'comment_rate': 5.0,
            'share_rate': 4.0,
            'follow_rate': 4.0,
            'interaction_rate': 2.0,
            'ces_rate': 1.5,
            'impression': 1.0,
            'sort_score': 2.0
        }
    
    @abstractmethod
    def create_model(self, feature_columns: List) -> torch.nn.Module:
        """创建模型（抽象方法，子类必须实现）
        
        Args:
            feature_columns: DeepCTR特征列定义
            
        Returns:
            初始化的模型
        """
        pass
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置（子类可覆盖）
        
        Returns:
            模型配置字典
        """
        return {
            'model_type': self.config.model_type,
            'tasks': self.config.tasks,
            'dropout': getattr(self.config, 'dropout', 0.1),
            'l2_reg_embedding': getattr(self.config, 'l2_reg_embedding', 1e-5),
            'l2_reg_dnn': getattr(self.config, 'l2_reg_dnn', 0)
        }
    
    def compile_model(self, learning_rate: float, targets: Dict[str, np.ndarray]) -> None:
        """编译模型（统一逻辑）
        
        Args:
            learning_rate: 学习率
            targets: 目标变量字典
        """
        logger.info("Compiling MTL model...")
        
        # 检测缺失值任务
        if targets:
            targets_with_columns = {
                task_name: targets[self.data_processor.get_column_name(task_name)] 
                for task_name in self.config.tasks 
                if self.data_processor.get_column_name(task_name) in targets
            }
            tasks_with_missing, tasks_missing_ratio = detect_tasks_with_missing_values(
                targets_with_columns, self.config.tasks
            )
            if tasks_with_missing:
                logger.info(f"Tasks with missing values (will use masked loss): {tasks_with_missing}")
        else:
            tasks_with_missing = []
        
        # 创建多任务损失函数
        multitask_loss = create_multitask_loss_function(
            task_names=self.config.tasks,
            task_weights=self.task_weights,
            tasks_with_missing=tasks_with_missing
        )
        
        # 创建优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 编译模型
        self.model.compile(
            optimizer=optimizer,
            loss=multitask_loss,
            metrics=['mse']
        )
        
        logger.info(f"✅ Model compiled with Adam optimizer (lr={learning_rate})")
        logger.info(f"   Task weights: {dict((task, self.task_weights.get(task, 1.0)) for task in self.config.tasks)}")
        logger.info(f"   Tasks with masked loss: {len(tasks_with_missing)}")
    
    def train_model(self, 
                   model_input: Dict[str, np.ndarray], 
                   targets: Dict[str, np.ndarray],
                   epochs: int, 
                   batch_size: int, 
                   learning_rate: float,
                   validation_split: float) -> Dict[str, Any]:
        """训练模型（统一逻辑）
        
        Args:
            model_input: 模型输入
            targets: 目标变量
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            validation_split: 验证集比例
            
        Returns:
            训练信息字典
        """
        logger.info("Starting MTL model training...")
        
        # 数据分割
        indices = np.arange(len(list(model_input.values())[0]))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
        
        # 分割输入特征
        train_model_input = {name: values[train_idx] for name, values in model_input.items()}
        val_model_input = {name: values[val_idx] for name, values in model_input.items()}
        
        # 准备目标变量
        train_targets = []
        val_targets = []
        
        for task_name in self.config.tasks:
            column_name = self.data_processor.get_column_name(task_name)
            if column_name in targets:
                train_targets.append(targets[column_name][train_idx])
                val_targets.append(targets[column_name][val_idx])
            else:
                logger.warning(f"Task {task_name} (column: {column_name}) not found, using zeros")
                train_targets.append(np.zeros(len(train_idx), dtype=np.float32))
                val_targets.append(np.zeros(len(val_idx), dtype=np.float32))
        
        # 转换为numpy数组
        train_targets_raw = np.column_stack(train_targets).astype(np.float32)
        val_targets_raw = np.column_stack(val_targets).astype(np.float32)
        
        # 标签归一化
        if self.config.label_normalization.value != 'none':
            logger.info(f"Normalizing target labels using {self.config.label_normalization.value} method...")
            
            self.label_normalizer = MultiTaskLabelNormalizer(
                normalization_method=self.config.label_normalization.value
            )
            self.evaluator.label_normalizer = self.label_normalizer
            
            # 归一化
            train_targets_dict = {task: train_targets_raw[:, i] for i, task in enumerate(self.config.tasks)}
            val_targets_dict = {task: val_targets_raw[:, i] for i, task in enumerate(self.config.tasks)}
            
            train_targets_dict_norm = self.label_normalizer.fit_transform(train_targets_dict, self.config.tasks)
            val_targets_dict_norm = self.label_normalizer.transform(val_targets_dict, self.config.tasks)
            
            train_targets = np.column_stack([train_targets_dict_norm[task] for task in self.config.tasks]).astype(np.float32)
            val_targets = np.column_stack([val_targets_dict_norm[task] for task in self.config.tasks]).astype(np.float32)
            
            logger.info("✅ Target normalization completed")
        else:
            train_targets = train_targets_raw
            val_targets = val_targets_raw
        
        logger.info(f"Training data: {len(train_idx)} samples, {train_targets.shape[1]} tasks")
        logger.info(f"Validation data: {len(val_idx)} samples, {val_targets.shape[1]} tasks")
        
        # 数据质量检查
        self._check_training_data_quality(train_targets, val_targets)
        
        # 设置回调
        callbacks = self._setup_callbacks()
        
        # 训练模型
        logger.info("🚀 Starting model training...")
        logger.info(f"Training config: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        logger.info(f"Early stopping: {'Enabled' if self.config.use_early_stopping else 'Disabled'}")
        
        try:
            history = self.model.fit(
                train_model_input,
                train_targets,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(val_model_input, val_targets),
                callbacks=callbacks if callbacks else None
            )
            
            logger.info("✅ Training completed successfully")
            
            # 分析训练历史
            self._analyze_training_history(history)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        # 评估模型
        final_results = self.evaluator.evaluate_model(self.model, val_model_input, targets, val_idx)
        
        training_info = {
            'epochs_completed': len(history.history.get('loss', [])) if hasattr(history, 'history') else epochs,
            'training_history': history.history if hasattr(history, 'history') else {},
            'early_stopped': self.config.use_early_stopping and len(history.history.get('loss', [])) < epochs if hasattr(history, 'history') else False,
            'evaluation_results': final_results,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        
        return training_info
    
    def _check_training_data_quality(self, train_targets: np.ndarray, val_targets: np.ndarray) -> None:
        """检查训练数据质量"""
        logger.info("="*50)
        logger.info("TRAINING DATA QUALITY CHECK")
        logger.info("="*50)
        
        for i, task_name in enumerate(self.config.tasks):
            train_task_data = train_targets[:, i]
            val_task_data = val_targets[:, i]
            
            logger.info(f"{task_name}:")
            logger.info(f"  Train - Mean: {train_task_data.mean():.6f}, Std: {train_task_data.std():.6f}")
            logger.info(f"  Train - Unique: {np.unique(train_task_data).shape[0]}")
            logger.info(f"  Val - Mean: {val_task_data.mean():.6f}, Std: {val_task_data.std():.6f}")
            
            # 检查常数目标
            if train_task_data.std() < 1e-8:
                logger.warning(f"  ⚠️  TRAIN TARGET IS CONSTANT for {task_name}!")
            if val_task_data.std() < 1e-8:
                logger.warning(f"  ⚠️  VAL TARGET IS CONSTANT for {task_name}!")
        
        logger.info("="*50)
    
    def _setup_callbacks(self) -> Optional[List]:
        """设置训练回调"""
        callbacks = []
        
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor='loss',
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                verbose=1,
                mode='min'
            )
            callbacks.append(early_stopping)
        
        return callbacks if callbacks else None
    
    def _analyze_training_history(self, history) -> None:
        """分析训练历史"""
        if hasattr(history, 'history') and history.history:
            logger.info("="*50)
            logger.info("TRAINING HISTORY ANALYSIS")
            logger.info("="*50)
            
            final_train_loss = history.history.get('loss', [0])[-1] if history.history.get('loss') else 0
            final_val_loss = history.history.get('val_loss', [0])[-1] if history.history.get('val_loss') else 0
            
            logger.info(f"Final training loss: {final_train_loss:.6f}")
            logger.info(f"Final validation loss: {final_val_loss:.6f}")
            
            if len(history.history.get('loss', [])) > 1:
                initial_loss = history.history['loss'][0]
                loss_reduction = (initial_loss - final_train_loss) / initial_loss * 100
                logger.info(f"Training loss reduction: {loss_reduction:.2f}%")
            
            # 检查过拟合
            if final_val_loss > final_train_loss * 1.5:
                logger.warning("⚠️  Possible overfitting detected (val_loss >> train_loss)")
            elif final_val_loss < final_train_loss * 0.8:
                logger.info("✅ Good generalization (val_loss < train_loss)")
            
            logger.info("="*50)
    
    def run(self, 
           epochs: int = None, 
           batch_size: int = None, 
           learning_rate: float = None,
           validation_split: float = None) -> Dict[str, Any]:
        """运行完整的MTL训练流程（8步统一流程）
        
        Args:
            epochs: 训练轮数（可选，使用配置默认值）
            batch_size: 批次大小（可选，使用配置默认值）
            learning_rate: 学习率（可选，使用配置默认值）
            validation_split: 验证集比例（可选，使用配置默认值）
            
        Returns:
            完整结果字典
        """
        start_time = time.time()
        
        # 使用配置默认值
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        learning_rate = learning_rate or self.config.learning_rate
        validation_split = validation_split or self.config.validation_split
        
        logger.info("="*80)
        logger.info(f"Starting {self.__class__.__name__} Multi-Task Learning Training")
        logger.info("="*80)
        
        # 1. 加载数据
        logger.info("Step 1/8: Loading data...")
        df = self.data_processor.load_data()
        self.data = df
        
        # 2. 工程化目标变量
        logger.info("Step 2/8: Engineering target variables...")
        df = self.data_processor.engineer_targets(df)
        
        # 3. 准备特征
        logger.info("Step 3/8: Preparing features...")
        model_input, feature_columns, feature_info = self.feature_processor.prepare_features(df)
        
        # 4. 准备目标变量
        logger.info("Step 4/8: Preparing targets...")
        task_columns = self.data_processor.get_task_columns()
        targets = self.feature_processor.prepare_targets(df, task_columns)
        
        # 5. 创建模型
        logger.info("Step 5/8: Creating model...")
        self.model = self.create_model(feature_columns)
        
        # 6. 编译模型
        logger.info("Step 6/8: Compiling model...")
        self.compile_model(learning_rate, targets)
        
        # 7. 训练模型
        logger.info("Step 7/8: Training model...")
        training_info = self.train_model(model_input, targets, epochs, batch_size, 
                                       learning_rate, validation_split)
        
        # 8. 保存结果
        logger.info("Step 8/8: Saving results...")
        results = self.evaluator.save_results(
            model_type=self.config.model_type,
            training_info=training_info,
            feature_info=feature_info,
            model_config=self.get_model_config(),
            training_config=self.config.get_training_config_dict(),
            input_path=self.config.input_path,
            model=self.model,
            preprocessors=self.feature_processor.get_preprocessors()
        )
        
        # 总时间
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info(f"✅ {self.__class__.__name__} TRAINING COMPLETED")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Tasks trained: {len(self.config.tasks)}")
        logger.info(f"Feature columns: {len(feature_columns)}")
        logger.info("="*80)
        
        # 打印结果摘要
        self.evaluator.print_summary(training_info['evaluation_results'])
        
        return results