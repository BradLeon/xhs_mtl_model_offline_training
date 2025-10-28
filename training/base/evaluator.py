#!/usr/bin/env python3
"""
多任务学习评估器基类

提供所有MTL训练器的通用评估功能：
- 统一模型性能评估
- 回归和排序指标计算
- 预测分布分析
- 结果保存和报告生成

消除MTL训练脚本中的评估代码重复
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class BaseMTLEvaluator:
    """多任务学习评估器基类
    
    封装所有MTL训练器通用的评估逻辑，包括：
    - 模型性能评估
    - 指标计算和分析
    - 预测分布分析
    - 结果保存和报告
    """
    
    def __init__(self, 
                 task_names: List[str],
                 task_column_mapping: Dict[str, str],
                 output_path: str,
                 use_label_normalization: bool = False,
                 label_normalizer = None):
        """初始化评估器
        
        Args:
            task_names: 任务名称列表
            task_column_mapping: 任务名到列名的映射
            output_path: 结果输出路径
            use_label_normalization: 是否使用标签归一化
            label_normalizer: 标签归一化器
        """
        self.task_names = task_names
        self.task_column_mapping = task_column_mapping
        self.output_path = Path(output_path)
        self.use_label_normalization = use_label_normalization
        self.label_normalizer = label_normalizer
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MTL evaluator for {len(task_names)} tasks")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Label normalization: {'Enabled' if use_label_normalization else 'Disabled'}")
    
    def evaluate_model(self, 
                      model,
                      model_input: Dict[str, np.ndarray], 
                      targets: Dict[str, np.ndarray], 
                      val_indices: np.ndarray) -> Dict[str, Dict[str, float]]:
        """评估MTL模型性能
        
        Args:
            model: 训练好的模型
            model_input: 验证集输入
            targets: 目标变量字典
            val_indices: 验证集索引
            
        Returns:
            每个任务的评估结果字典
        """
        logger.info("Evaluating MTL model performance...")
        
        results = {}
        
        try:
            # 获取模型预测
            predictions = model.predict(model_input, batch_size=256)
            logger.info(f"Predictions shape: {predictions.shape}")
            
            # 反归一化预测值
            if self.use_label_normalization and self.label_normalizer is not None:
                logger.info("Denormalizing predictions back to original scale...")
                predictions = self.label_normalizer.inverse_transform(predictions, self.task_names)
                logger.info("✅ Prediction denormalization completed")
            
            # 分析预测值分布
            self._analyze_prediction_distribution(predictions)
            
            # 计算每个任务的评估指标
            for i, task_name in enumerate(self.task_names):
                column_name = self.task_column_mapping.get(task_name, task_name)
                if column_name in targets:
                    task_results = self._evaluate_single_task(
                        task_name=task_name,
                        predictions=predictions[:, i] if predictions.ndim > 1 else predictions,
                        targets=targets[column_name][val_indices]
                    )
                    results[task_name] = task_results
                else:
                    logger.warning(f"Task {task_name} not found in targets")
                    results[task_name] = {'error': f'Target column {column_name} not found'}
                    
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # 返回错误结果
            for task_name in self.task_names:
                results[task_name] = {
                    'mse': 0.0,
                    'valid_samples': 0,
                    'error': str(e)
                }
        
        return results
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray) -> None:
        """分析预测值分布
        
        Args:
            predictions: 模型预测值
        """
        logger.info("="*60)
        logger.info("PREDICTION DISTRIBUTION ANALYSIS")
        logger.info("="*60)
        
        for i, task_name in enumerate(self.task_names):
            task_predictions = predictions[:, i] if predictions.ndim > 1 else predictions
            
            logger.info(f"{task_name} predictions:")
            logger.info(f"  Mean: {task_predictions.mean():.6f}")
            logger.info(f"  Std: {task_predictions.std():.6f}")
            logger.info(f"  Min: {task_predictions.min():.6f}")
            logger.info(f"  Max: {task_predictions.max():.6f}")
            logger.info(f"  Unique values: {np.unique(task_predictions).shape[0]}")
            
            # 检查预测值质量
            if task_predictions.std() < 1e-8:
                logger.warning(f"  ⚠️  CONSTANT PREDICTIONS for {task_name}!")
            elif np.unique(task_predictions).shape[0] < 5:
                logger.warning(f"  ⚠️  LIMITED PREDICTION DIVERSITY for {task_name}")
        
        logger.info("="*60)
    
    def _evaluate_single_task(self, 
                             task_name: str, 
                             predictions: np.ndarray, 
                             targets: np.ndarray) -> Dict[str, float]:
        """评估单个任务的性能
        
        Args:
            task_name: 任务名称
            predictions: 预测值
            targets: 真实值
            
        Returns:
            单任务评估指标字典
        """
        try:
            # 回归指标
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            rmse = np.sqrt(mse)
            
            # R²评分
            try:
                r2 = r2_score(targets, predictions)
            except:
                r2 = 0.0
            
            # 排序指标
            try:
                spearman_corr, spearman_p = spearmanr(targets, predictions)
                kendall_corr, kendall_p = kendalltau(targets, predictions)
            except:
                spearman_corr = spearman_p = 0.0
                kendall_corr = kendall_p = 0.0
            
            # 构建结果字典
            task_results = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'spearman_correlation': float(spearman_corr),
                'spearman_pvalue': float(spearman_p),
                'kendall_tau': float(kendall_corr),
                'kendall_pvalue': float(kendall_p),
                'valid_samples': int(len(targets))
            }
            
            logger.info(f"Task {task_name}: MSE={mse:.6f}, R²={r2:.4f}, "
                       f"Spearman={spearman_corr:.4f}, Kendall={kendall_corr:.4f}, Samples={len(targets)}")
            
            return task_results
            
        except Exception as e:
            logger.error(f"Failed to evaluate task {task_name}: {e}")
            return {
                'mse': 0.0,
                'valid_samples': 0,
                'error': str(e)
            }
    
    def save_results(self, 
                    model_type: str,
                    training_info: Dict[str, Any], 
                    feature_info: Dict[str, Any],
                    model_config: Dict[str, Any],
                    training_config: Dict[str, Any],
                    input_path: str,
                    model = None,
                    preprocessors: Optional[Dict[str, Any]] = None,
                    feature_columns: Optional[List] = None) -> Dict[str, Any]:
        """保存训练结果（增强版：包含完整checkpoint）
        
        Args:
            model_type: 模型类型
            training_info: 训练信息
            feature_info: 特征信息
            model_config: 模型配置
            training_config: 训练配置
            input_path: 输入路径
            model: 模型实例（可选）
            preprocessors: 预处理器（可选）
            feature_columns: DeepCTR特征列定义（可选）
            
        Returns:
            完整的结果字典，包含checkpoint路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建checkpoint目录
        checkpoint_dir = self.output_path / f"checkpoint_{model_type.lower()}_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating checkpoint directory: {checkpoint_dir}")
        
        # 构建结果字典
        results = {
            'model_type': model_type,
            'timestamp': timestamp,
            'tasks': self.task_names,
            'task_column_mapping': self.task_column_mapping,
            'model_config': model_config,
            'training_config': training_config,
            'feature_info': feature_info,
            'training_info': training_info,
            'input_path': input_path,
            'checkpoint_dir': str(checkpoint_dir)
        }
        
        # 1. 保存训练信息JSON
        training_info_file = checkpoint_dir / "training_info.json"
        with open(training_info_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✅ Training info saved to {training_info_file}")
        
        # 2. 保存模型权重
        if model is not None:
            model_file = checkpoint_dir / "model.pth"
            torch.save(model.state_dict(), model_file)
            logger.info(f"✅ Model weights saved to {model_file}")
            
            # 尝试保存完整模型（用于快速加载）
            # 注意：如果模型包含不可序列化的组件（如自定义损失函数），此操作可能失败
            complete_model_file = checkpoint_dir / "complete_model.pth"
            try:
                torch.save(model, complete_model_file)
                logger.info(f"✅ Complete model saved to {complete_model_file}")
            except (AttributeError, pickle.PicklingError) as e:
                logger.warning(f"⚠️  Could not save complete model due to unpicklable components: {e}")
                logger.info("   This is expected for models with custom loss functions.")
                logger.info("   The model can still be loaded using the 'rebuild' method.")
                # 删除可能创建的损坏文件
                if complete_model_file.exists():
                    complete_model_file.unlink()
        
        # 3. 保存模型配置（独立文件，便于重建模型）
        model_config_file = checkpoint_dir / "model_config.json"
        enhanced_model_config = {
            **model_config,
            'model_class': model.__class__.__name__ if model else None,
            'device': str(model.device) if model and hasattr(model, 'device') else 'cpu',
            'tasks': self.task_names,
            'task_column_mapping': self.task_column_mapping
        }
        with open(model_config_file, 'w') as f:
            json.dump(enhanced_model_config, f, indent=2)
        logger.info(f"✅ Model config saved to {model_config_file}")
        
        # 4. 保存特征列定义（关键：用于模型重建）
        if feature_columns:
            feature_columns_file = checkpoint_dir / "feature_columns.json"
            feature_columns_dict = self._serialize_feature_columns(feature_columns)
            with open(feature_columns_file, 'w') as f:
                json.dump(feature_columns_dict, f, indent=2)
            logger.info(f"✅ Feature columns saved to {feature_columns_file}")
        elif 'feature_columns' in feature_info:
            # 尝试从feature_info中获取
            feature_columns_file = checkpoint_dir / "feature_columns.json"
            feature_columns_dict = self._serialize_feature_columns(feature_info['feature_columns'])
            with open(feature_columns_file, 'w') as f:
                json.dump(feature_columns_dict, f, indent=2)
            logger.info(f"✅ Feature columns saved from feature_info")
        
        # 5. 保存预处理器
        if preprocessors is not None:
            preprocessor_file = checkpoint_dir / "preprocessors.pkl"
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(preprocessors, f)
            logger.info(f"✅ Preprocessors saved to {preprocessor_file}")
        
        # 6. 保存标签归一化器（如果存在）
        if self.label_normalizer is not None:
            normalizer_file = checkpoint_dir / "label_normalizer.pkl"
            with open(normalizer_file, 'wb') as f:
                pickle.dump(self.label_normalizer, f)
            logger.info(f"✅ Label normalizer saved to {normalizer_file}")
        
        # 7. 创建checkpoint元数据文件（便于版本控制）
        metadata = {
            'version': '1.0',
            'created_at': timestamp,
            'model_type': model_type,
            'tasks': self.task_names,
            'files': {
                'model_weights': 'model.pth',
                'complete_model': 'complete_model.pth',
                'model_config': 'model_config.json',
                'feature_columns': 'feature_columns.json',
                'preprocessors': 'preprocessors.pkl',
                'label_normalizer': 'label_normalizer.pkl' if self.label_normalizer else None,
                'training_info': 'training_info.json'
            },
            'requirements': {
                'deepctr_torch': '0.2.9',  # 根据实际版本调整
                'torch': torch.__version__
            }
        }
        metadata_file = checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Checkpoint metadata saved to {metadata_file}")
        
        # 打印checkpoint摘要
        logger.info("="*60)
        logger.info(f"📦 CHECKPOINT SAVED SUCCESSFULLY")
        logger.info(f"📁 Directory: {checkpoint_dir}")
        logger.info(f"📊 Model type: {model_type}")
        logger.info(f"🎯 Tasks: {', '.join(self.task_names)}")
        logger.info("="*60)
        
        # 更新results字典
        results['checkpoint_files'] = metadata['files']
        
        return results
    
    def _serialize_feature_columns(self, feature_columns: List) -> List[Dict]:
        """将DeepCTR特征列对象序列化为字典
        
        Args:
            feature_columns: DeepCTR特征列列表
            
        Returns:
            可序列化的字典列表
        """
        serialized = []
        for col in feature_columns:
            if hasattr(col, 'name'):
                col_dict = {
                    'name': col.name,
                    'type': col.__class__.__name__  # SparseFeat or DenseFeat
                }
                
                # SparseFeat特有属性
                if hasattr(col, 'vocabulary_size'):
                    col_dict['vocabulary_size'] = col.vocabulary_size
                    col_dict['embedding_dim'] = col.embedding_dim
                    col_dict['dtype'] = str(col.dtype)
                
                # DenseFeat特有属性
                if hasattr(col, 'dimension'):
                    col_dict['dimension'] = col.dimension
                    
                serialized.append(col_dict)
        
        return serialized
    
    def print_summary(self, evaluation_results: Dict[str, Dict[str, float]]) -> None:
        """打印评估结果摘要
        
        Args:
            evaluation_results: 评估结果字典
        """
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        for task_name, metrics in evaluation_results.items():
            if 'mse' in metrics:
                print(f"{task_name}:")
                print(f"  MSE: {metrics['mse']:.6f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  Spearman: {metrics['spearman_correlation']:.4f}")
                print(f"  Kendall: {metrics['kendall_tau']:.4f}")
                print(f"  Valid samples: {metrics['valid_samples']}")
            elif 'error' in metrics:
                print(f"{task_name}: ERROR - {metrics['error']}")
        
        print("="*60)
    
    def analyze_training_quality(self, training_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析训练质量
        
        Args:
            training_info: 训练信息
            
        Returns:
            训练质量分析报告
        """
        analysis = {
            'training_completed': False,
            'early_stopped': False,
            'convergence_issues': [],
            'overfitting_detected': False,
            'underfitting_detected': False
        }
        
        # 检查训练是否完成
        if 'epochs_completed' in training_info:
            analysis['training_completed'] = training_info['epochs_completed'] > 0
        
        # 检查是否早停
        if 'early_stopped' in training_info:
            analysis['early_stopped'] = training_info['early_stopped']
        
        # 分析训练历史
        if 'training_history' in training_info and training_info['training_history']:
            history = training_info['training_history']
            
            # 检查损失收敛
            if 'loss' in history and len(history['loss']) > 5:
                recent_losses = history['loss'][-5:]
                if max(recent_losses) - min(recent_losses) > 0.01:
                    analysis['convergence_issues'].append('Loss not converged')
            
            # 检查过拟合
            if 'loss' in history and 'val_loss' in history:
                final_train_loss = history['loss'][-1] if history['loss'] else 0
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0
                
                if final_val_loss > final_train_loss * 1.5:
                    analysis['overfitting_detected'] = True
                elif final_val_loss > 0.1 and final_train_loss > 0.1:  # 高损失
                    analysis['underfitting_detected'] = True
        
        return analysis
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Dict[str, float]],
                                 training_info: Dict[str, Any]) -> str:
        """生成评估报告
        
        Args:
            evaluation_results: 评估结果
            training_info: 训练信息
            
        Returns:
            评估报告字符串
        """
        report_lines = [
            "="*80,
            "MTL MODEL EVALUATION REPORT",
            "="*80,
            "",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Tasks evaluated: {len(self.task_names)}",
            "",
            "PERFORMANCE SUMMARY:",
            "-" * 40
        ]
        
        # 添加每个任务的性能
        for task_name, metrics in evaluation_results.items():
            if 'mse' in metrics:
                report_lines.extend([
                    f"{task_name}:",
                    f"  MSE: {metrics['mse']:.6f}",
                    f"  R²: {metrics['r2']:.4f}",
                    f"  Spearman: {metrics['spearman_correlation']:.4f}",
                    f"  Kendall: {metrics['kendall_tau']:.4f}",
                    f"  Samples: {metrics['valid_samples']}",
                    ""
                ])
        
        # 添加训练质量分析
        quality_analysis = self.analyze_training_quality(training_info)
        report_lines.extend([
            "TRAINING QUALITY ANALYSIS:",
            "-" * 40,
            f"Training completed: {quality_analysis['training_completed']}",
            f"Early stopped: {quality_analysis['early_stopped']}",
            f"Overfitting detected: {quality_analysis['overfitting_detected']}",
            f"Underfitting detected: {quality_analysis['underfitting_detected']}",
        ])
        
        if quality_analysis['convergence_issues']:
            report_lines.append(f"Convergence issues: {', '.join(quality_analysis['convergence_issues'])}")
        
        report_lines.append("="*80)
        
        return "\n".join(report_lines)