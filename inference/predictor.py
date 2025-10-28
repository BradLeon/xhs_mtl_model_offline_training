#!/usr/bin/env python3
"""
多任务学习预测器

提供高级推理API，整合模型加载、特征处理和预测功能
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import torch

from .model_loader import ModelLoader
from .feature_processor import InferenceFeatureProcessor

logger = logging.getLogger(__name__)


class MTLPredictor:
    """多任务学习预测器
    
    提供简单易用的推理接口
    """
    
    def __init__(self, checkpoint_dir: str, device: str = 'auto', load_method: str = 'auto'):
        """初始化预测器
        
        Args:
            checkpoint_dir: checkpoint目录路径
            device: 设备类型 ('auto', 'cpu', 'cuda', 'mps')
            load_method: 模型加载方法 ('auto', 'complete' 或 'rebuild')
                - 'auto': 自动选择（优先尝试complete，失败则rebuild）
                - 'complete': 直接加载完整模型
                - 'rebuild': 从配置重建模型
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        logger.info("="*60)
        logger.info("Initializing MTL Predictor")
        logger.info(f"Checkpoint: {checkpoint_dir}")
        logger.info(f"Device: {device}")
        logger.info(f"Load method: {load_method}")
        logger.info("="*60)
        
        # 1. 初始化模型加载器
        self.model_loader = ModelLoader(checkpoint_dir, device)
        
        # 2. 加载模型
        logger.info("Loading model...")
        self.model = self.model_loader.load_model(method=load_method)
        self.model.eval()
        
        # 3. 加载预处理器
        logger.info("Loading preprocessors...")
        self.preprocessors = self.model_loader.load_preprocessors()
        
        # 4. 加载特征列定义
        feature_columns = self.model_loader._load_feature_columns()
        
        # 5. 初始化特征处理器
        self.feature_processor = InferenceFeatureProcessor(
            self.preprocessors, 
            feature_columns
        )
        
        # 6. 加载标签归一化器（如果存在）
        self.label_normalizer = self.model_loader.load_label_normalizer()
        
        # 7. 加载训练信息
        self.training_info = self.model_loader.load_training_info()
        self.tasks = self.training_info.get('tasks', [])
        self.task_column_mapping = self.training_info.get('task_column_mapping', {})
        
        logger.info("✅ MTL Predictor initialized successfully")
        logger.info(f"Model type: {self.training_info.get('model_type', 'unknown')}")
        logger.info(f"Tasks: {', '.join(self.tasks)}")
        
        # 预热模型
        self._warmup()
    
    def _warmup(self):
        """预热模型，减少首次推理延迟"""
        logger.info("Warming up model...")
        
        # 创建虚拟输入
        dummy_input = {}
        for feat_name in self.feature_processor.feature_names:
            dummy_input[feat_name] = np.zeros(1, dtype=np.float32)
        
        # 执行一次推理
        with torch.no_grad():
            _ = self.model.predict(dummy_input, batch_size=1)
        
        logger.info("✅ Model warmed up")
    
    def predict_single(self, note_data: Dict[str, Any]) -> Dict[str, float]:
        """预测单条笔记
        
        Args:
            note_data: 笔记数据字典
            
        Returns:
            预测结果字典，key为任务名，value为预测值
        """
        start_time = time.time()
        
        # 处理特征
        model_input = self.feature_processor.process_single(note_data)
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model.predict(model_input, batch_size=1)
        
        # 处理预测结果
        results = self._process_predictions(predictions)
        
        # 记录推理时间
        inference_time = time.time() - start_time
        results['inference_time_ms'] = inference_time * 1000
        
        logger.info(f"Single prediction completed in {inference_time*1000:.2f}ms")
        return results
    
    def predict_batch(self, notes_data: Union[List[Dict], pd.DataFrame], 
                     batch_size: int = 256) -> pd.DataFrame:
        """批量预测
        
        Args:
            notes_data: 笔记数据列表或DataFrame
            batch_size: 批处理大小
            
        Returns:
            预测结果DataFrame
        """
        start_time = time.time()
        
        # 转换为DataFrame
        if isinstance(notes_data, list):
            df = pd.DataFrame(notes_data)
        else:
            df = notes_data.copy()
        
        logger.info(f"Batch prediction for {len(df)} samples")
        
        # 处理特征
        model_input = self.feature_processor.process_batch(df)
        
        # 模型预测
        with torch.no_grad():
            predictions = self.model.predict(model_input, batch_size=batch_size)
        
        # 构建结果DataFrame
        results_df = pd.DataFrame()
        
        # 如果有标签归一化，需要反归一化
        if self.label_normalizer is not None:
            logger.info("Denormalizing predictions...")
            predictions = self.label_normalizer.inverse_transform(predictions, self.tasks)
        
        # 添加预测结果
        for i, task in enumerate(self.tasks):
            if predictions.ndim > 1:
                results_df[f'{task}_pred'] = predictions[:, i]
            else:
                results_df[f'{task}_pred'] = predictions
        
        # 添加原始ID（如果存在）
        if 'note_id' in df.columns:
            results_df['note_id'] = df['note_id'].values
        
        # 记录推理时间
        inference_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {inference_time:.2f}s")
        logger.info(f"Average time per sample: {inference_time/len(df)*1000:.2f}ms")
        
        return results_df
    
    def _process_predictions(self, predictions: np.ndarray) -> Dict[str, float]:
        """处理预测结果
        
        Args:
            predictions: 模型原始预测
            
        Returns:
            处理后的预测字典
        """
        results = {}
        
        # 如果有标签归一化，需要反归一化
        if self.label_normalizer is not None:
            predictions = self.label_normalizer.inverse_transform(predictions, self.tasks)
        
        # 构建结果字典
        for i, task in enumerate(self.tasks):
            if predictions.ndim > 1:
                value = float(predictions[0, i])
            else:
                value = float(predictions[0]) if i == 0 else 0.0
            
            # 确保预测值在合理范围内
            if 'rate' in task or task == 'ctr':
                value = np.clip(value, 0.0, 1.0)
            
            results[task] = value
        
        return results
    
    def predict_with_explanation(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """带解释的预测
        
        Args:
            note_data: 笔记数据
            
        Returns:
            包含预测结果和解释的字典
        """
        # 基础预测
        predictions = self.predict_single(note_data)
        
        # 构建解释
        explanation = {
            'predictions': predictions,
            'model_info': {
                'type': self.training_info.get('model_type', 'unknown'),
                'tasks': self.tasks,
                'trained_at': self.training_info.get('timestamp', 'unknown')
            },
            'feature_info': self._get_feature_explanation(note_data),
            'confidence': self._calculate_confidence(predictions)
        }
        
        return explanation
    
    def _get_feature_explanation(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取特征解释
        
        Args:
            note_data: 输入数据
            
        Returns:
            特征解释字典
        """
        feature_importance = self.feature_processor.get_feature_importance()
        
        explanation = {
            'total_features': len(feature_importance),
            'feature_types': {
                'sparse': len([f for f, t in feature_importance.items() if t == 'sparse']),
                'dense': len([f for f, t in feature_importance.items() if t == 'dense']),
                'clip': len([f for f, t in feature_importance.items() if t == 'clip'])
            },
            'missing_features': []
        }
        
        # 检查缺失特征
        for feat in feature_importance.keys():
            if feat not in note_data or note_data[feat] is None:
                explanation['missing_features'].append(feat)
        
        return explanation
    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> str:
        """计算预测置信度
        
        Args:
            predictions: 预测结果
            
        Returns:
            置信度等级
        """
        # 简单的置信度估计（可以根据实际需求改进）
        ctr = predictions.get('ctr', 0)
        
        if ctr > 0.1:
            return 'high'
        elif ctr > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'model_type': self.training_info.get('model_type', 'unknown'),
            'tasks': self.tasks,
            'num_features': len(self.feature_processor.feature_names),
            'feature_breakdown': {
                'sparse': len(self.feature_processor.sparse_features),
                'dense': len(self.feature_processor.dense_features),
                'clip': len(self.feature_processor.clip_features)
            },
            'has_label_normalizer': self.label_normalizer is not None,
            'training_info': {
                'timestamp': self.training_info.get('timestamp', 'unknown'),
                'input_path': self.training_info.get('input_path', 'unknown')
            }
        }
        
        # 添加模型参数数量
        if hasattr(self.model, 'parameters'):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            summary['model_params'] = {
                'total': total_params,
                'trainable': trainable_params
            }
        
        return summary