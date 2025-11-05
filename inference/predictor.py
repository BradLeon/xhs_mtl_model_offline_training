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

    def predict_single_with_diagnosis(self, note_data: Dict[str, Any]) -> Dict[str, Any]:
        """预测单条笔记并返回完整诊断信息（避免重复计算）

        Args:
            note_data: 笔记数据字典

        Returns:
            包含原始预测、反归一化预测和最终结果的完整字典
        """
        start_time = time.time()

        # 处理特征（只处理一次）
        model_input = self.feature_processor.process_single(note_data)

        # 模型预测（只预测一次）
        with torch.no_grad():
            raw_predictions = self.model.predict(model_input, batch_size=1)

        # 诊断预测值（包含原始值和反归一化过程）
        prediction_diagnosis = self.diagnose_predictions(
            predictions=raw_predictions,
            note_data=note_data,
            show_details=True
        )

        # 处理预测结果（最终结果）
        final_results = self._process_predictions(raw_predictions)

        # 记录推理时间
        inference_time = time.time() - start_time
        final_results['inference_time_ms'] = inference_time * 1000

        logger.info(f"Prediction with diagnosis completed in {inference_time*1000:.2f}ms")

        # 返回完整诊断信息
        return {
            'raw_predictions': raw_predictions,  # 原始预测（归一化空间）
            'prediction_diagnosis': prediction_diagnosis,  # 详细诊断信息
            'final_predictions': final_results  # 最终结果
        }

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
                task_predictions = predictions[:, i]

                # ✅ CRITICAL FIX: Convert impression from log-space to actual count
                if task == 'impression':
                    # Apply exp() transformation to convert from log-space
                    task_predictions = np.exp(task_predictions)
                    logger.info(f"Applied exp() transformation to impression_log -> impression count")

                results_df[f'{task}_pred'] = task_predictions
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

        # 💾 保存原始预测（归一化空间）用于诊断
        logger.info(f"📊 Raw predictions from model (normalized space): {predictions}")
        logger.info(f"   Shape: {predictions.shape}, Dtype: {predictions.dtype}")

        # 如果有标签归一化，需要反归一化
        denormalized_predictions = predictions
        if self.label_normalizer is not None:
            logger.info(f"🔄 Applying label denormalization...")
            denormalized_predictions = self.label_normalizer.inverse_transform(predictions, self.tasks)
            logger.info(f"✅ Denormalized predictions (original space): {denormalized_predictions}")

            # 显示每个任务的反归一化前后对比
            for i, task in enumerate(self.tasks):
                if i < len(predictions[0]):
                    logger.info(f"   {task}: {predictions[0][i]:.4f} → {denormalized_predictions[0][i]:.4f}")
        else:
            logger.warning("⚠️  No label_normalizer available - predictions remain in normalized space!")

        # 构建结果字典
        for i, task in enumerate(self.tasks):
            if denormalized_predictions.ndim > 1:
                value = float(denormalized_predictions[0, i])
            else:
                value = float(denormalized_predictions[0]) if i == 0 else 0.0

            # ✅ 确保预测值在合理范围内 (clip rate/ctr tasks to [0, 1])
            if 'rate' in task or task == 'ctr':
                original_value = value
                value = np.clip(value, 0.0, 1.0)
                if abs(original_value - value) > 0.01:
                    logger.warning(f"⚠️  Task {task} prediction {original_value:.4f} clipped to {value:.4f}")

            # ✅ CRITICAL FIX: Convert impression from log-space to actual count
            if task == 'impression':
                # The denormalized value is in log-space (impression_log)
                # Need to apply exp() to get actual impression count
                impression_log = value
                value = np.exp(impression_log) if impression_log > 0 else 1000.0
                logger.info(f"🔄 Impression: {impression_log:.4f} (log) → {value:.0f} (actual count)")

            results[task] = value

        logger.info(f"\n📋 Final predictions:")
        for task, value in results.items():
            if task == 'impression':
                logger.info(f"   {task}: {value:.0f}")
            else:
                logger.info(f"   {task}: {value:.4f}")

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

    # ==================== 诊断工具方法 ====================

    def diagnose_normalizer(self) -> Dict[str, Any]:
        """诊断label normalizer的统计信息

        Returns:
            normalizer诊断信息字典
        """
        logger.info("="*80)
        logger.info("LABEL NORMALIZER DIAGNOSIS")
        logger.info("="*80)

        if self.label_normalizer is None:
            logger.warning("⚠️  No label normalizer found!")
            return {'status': 'no_normalizer'}

        diagnosis = {
            'normalization_method': self.label_normalizer.normalization_method,
            'fitted_tasks': list(self.label_normalizer.fitted_tasks),
            'task_statistics': {}
        }

        # 获取每个任务的统计信息
        for task in self.tasks:
            if task in self.label_normalizer.normalizers:
                scaler = self.label_normalizer.normalizers[task]

                task_stats = {
                    'scaler_type': type(scaler).__name__
                }

                # StandardScaler
                if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    task_stats['mean'] = float(scaler.mean_[0]) if scaler.mean_.ndim > 0 else float(scaler.mean_)
                    task_stats['std'] = float(scaler.scale_[0]) if scaler.scale_.ndim > 0 else float(scaler.scale_)
                    logger.info(f"\n📊 {task}:")
                    logger.info(f"  Scaler: {task_stats['scaler_type']}")
                    logger.info(f"  Mean: {task_stats['mean']:.6f}")
                    logger.info(f"  Std: {task_stats['std']:.6f}")

                # MinMaxScaler
                elif hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                    task_stats['min'] = float(scaler.data_min_[0]) if scaler.data_min_.ndim > 0 else float(scaler.data_min_)
                    task_stats['max'] = float(scaler.data_max_[0]) if scaler.data_max_.ndim > 0 else float(scaler.data_max_)
                    logger.info(f"\n📊 {task}:")
                    logger.info(f"  Scaler: {task_stats['scaler_type']}")
                    logger.info(f"  Min: {task_stats['min']:.6f}")
                    logger.info(f"  Max: {task_stats['max']:.6f}")

                diagnosis['task_statistics'][task] = task_stats
            else:
                logger.warning(f"⚠️  Task '{task}' not found in normalizer!")
                diagnosis['task_statistics'][task] = {'status': 'missing'}

        logger.info("="*80)
        return diagnosis

    def verify_task_order(self) -> Dict[str, Any]:
        """验证tasks顺序一致性

        Returns:
            顺序验证结果字典
        """
        logger.info("="*80)
        logger.info("TASK ORDER VERIFICATION")
        logger.info("="*80)

        verification = {
            'loaded_tasks': self.tasks,
            'task_column_mapping': self.task_column_mapping,
            'normalizer_tasks': list(self.label_normalizer.fitted_tasks) if self.label_normalizer else [],
            'consistency_check': {}
        }

        logger.info(f"\n📋 Loaded tasks order ({len(self.tasks)}):")
        for i, task in enumerate(self.tasks):
            column_name = self.task_column_mapping.get(task, task)
            logger.info(f"  [{i}] {task} → {column_name}")

        # 检查normalizer中的任务
        if self.label_normalizer:
            normalizer_tasks_set = set(self.label_normalizer.fitted_tasks)
            loaded_tasks_set = set(self.tasks)

            missing_in_normalizer = loaded_tasks_set - normalizer_tasks_set
            extra_in_normalizer = normalizer_tasks_set - loaded_tasks_set

            verification['consistency_check'] = {
                'all_tasks_in_normalizer': len(missing_in_normalizer) == 0,
                'missing_in_normalizer': list(missing_in_normalizer),
                'extra_in_normalizer': list(extra_in_normalizer)
            }

            if missing_in_normalizer:
                logger.warning(f"⚠️  Tasks missing in normalizer: {missing_in_normalizer}")
            if extra_in_normalizer:
                logger.warning(f"⚠️  Extra tasks in normalizer: {extra_in_normalizer}")

            if len(missing_in_normalizer) == 0 and len(extra_in_normalizer) == 0:
                logger.info("✅ All tasks are consistent between loaded tasks and normalizer")

        logger.info("="*80)
        return verification

    def diagnose_predictions(self, predictions: np.ndarray,
                           note_data: Optional[Dict] = None,
                           show_details: bool = True) -> Dict[str, Any]:
        """诊断预测值分布和处理流程

        Args:
            predictions: 原始预测值（归一化或未归一化）
            note_data: 输入笔记数据（可选，用于显示上下文）
            show_details: 是否显示详细信息

        Returns:
            诊断信息字典
        """
        logger.info("="*80)
        logger.info("PREDICTION DIAGNOSIS")
        logger.info("="*80)

        diagnosis = {
            'predictions_shape': predictions.shape,
            'tasks_count': len(self.tasks),
            'has_normalizer': self.label_normalizer is not None,
            'task_predictions': {}
        }

        # 显示输入数据上下文
        if note_data and show_details:
            logger.info(f"\n📝 Input note data:")
            logger.info(f"  note_id: {note_data.get('note_id', 'N/A')}")
            logger.info(f"  title: {note_data.get('title', 'N/A')[:50]}...")

        # 原始预测值
        logger.info(f"\n🔢 Raw predictions (from model):")
        logger.info(f"  Shape: {predictions.shape}")
        if predictions.ndim > 1:
            logger.info(f"  Values: {predictions[0, :]}")
        else:
            logger.info(f"  Values: {predictions}")

        # 保存原始预测用于对比
        raw_predictions = predictions.copy()

        # 反归一化（如果有normalizer）
        denormalized_predictions = predictions
        if self.label_normalizer is not None:
            logger.info(f"\n🔄 Applying denormalization...")
            denormalized_predictions = self.label_normalizer.inverse_transform(predictions, self.tasks)
            logger.info(f"  Denormalized shape: {denormalized_predictions.shape}")
            if denormalized_predictions.ndim > 1:
                logger.info(f"  Denormalized values: {denormalized_predictions[0, :]}")
            else:
                logger.info(f"  Denormalized values: {denormalized_predictions}")

        # 分析每个任务的预测
        logger.info(f"\n📊 Per-task prediction analysis:")
        logger.info("-" * 80)

        for i, task in enumerate(self.tasks):
            if predictions.ndim > 1:
                raw_val = float(raw_predictions[0, i])
                denorm_val = float(denormalized_predictions[0, i])
            else:
                raw_val = float(raw_predictions[0]) if i == 0 else 0.0
                denorm_val = float(denormalized_predictions[0]) if i == 0 else 0.0

            # 应用clip（如果适用）
            clipped_val = denorm_val
            if 'rate' in task or task == 'ctr':
                clipped_val = np.clip(denorm_val, 0.0, 1.0)

            # Impression特殊处理：检查是否需要exp变换
            imp_num_val = None
            if task == 'impression':
                imp_num_val = np.exp(denorm_val)

            task_info = {
                'raw_prediction': raw_val,
                'denormalized': denorm_val,
                'clipped': clipped_val,
                'clipped_applied': clipped_val != denorm_val
            }

            if imp_num_val is not None:
                task_info['exp_transformed'] = imp_num_val

            diagnosis['task_predictions'][task] = task_info

            # 日志输出
            logger.info(f"\n[{i}] {task}:")
            logger.info(f"  Raw (normalized):  {raw_val:>12.6f}")
            logger.info(f"  Denormalized:      {denorm_val:>12.6f}")

            if task_info['clipped_applied']:
                logger.info(f"  After clip:        {clipped_val:>12.6f} ⚠️ (clipped!)")
            else:
                logger.info(f"  After clip:        {clipped_val:>12.6f}")

            if imp_num_val is not None:
                logger.info(f"  Exp(denorm):       {imp_num_val:>12.0f} (imp_num estimate)")

            # 警告异常值
            if denorm_val < 0:
                logger.warning(f"  ⚠️  NEGATIVE VALUE DETECTED!")
            if 'rate' in task or task == 'ctr':
                if denorm_val > 1.0:
                    logger.warning(f"  ⚠️  Value > 1.0 for rate/ctr task!")

        logger.info("="*80)
        return diagnosis