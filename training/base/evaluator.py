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
from scipy.stats import spearmanr, kendalltau, skew, kurtosis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - scatter plots will be disabled")

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
                 label_normalizer = None,
                 enable_visualization: bool = True,
                 scatter_sample_size: int = 1000,
                 plot_dpi: int = 150):
        """初始化评估器

        Args:
            task_names: 任务名称列表
            task_column_mapping: 任务名到列名的映射
            output_path: 结果输出路径
            use_label_normalization: 是否使用标签归一化
            label_normalizer: 标签归一化器
            enable_visualization: 是否启用可视化（散点图）
            scatter_sample_size: 散点图采样数量（默认1000）
            plot_dpi: 图表DPI质量（默认150）
        """
        self.task_names = task_names
        self.task_column_mapping = task_column_mapping
        self.output_path = Path(output_path)
        self.use_label_normalization = use_label_normalization
        self.label_normalizer = label_normalizer

        # 可视化配置
        self.enable_visualization = enable_visualization and MATPLOTLIB_AVAILABLE
        self.scatter_sample_size = scatter_sample_size
        self.plot_dpi = plot_dpi

        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized MTL evaluator for {len(task_names)} tasks")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Label normalization: {'Enabled' if use_label_normalization else 'Disabled'}")
        logger.info(f"Visualization: {'Enabled' if self.enable_visualization else 'Disabled'}")
        if self.enable_visualization:
            logger.info(f"  Scatter sample size: {scatter_sample_size}")
            logger.info(f"  Plot DPI: {plot_dpi}")
    
    def evaluate_model(self,
                      model,
                      model_input: Dict[str, np.ndarray],
                      targets: Dict[str, np.ndarray],
                      val_indices: np.ndarray,
                      checkpoint_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """评估MTL模型性能（增强版：包含详细统计和可视化）

        Args:
            model: 训练好的模型
            model_input: 验证集输入
            targets: 目标变量字典
            val_indices: 验证集索引
            checkpoint_dir: checkpoint目录路径（用于保存可视化图表）

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

            # 分析预测值分布（增强版：包含Target对比）
            self._analyze_prediction_distribution(predictions, targets, val_indices)

            # 创建验证集散点图（如果启用）
            if self.enable_visualization and checkpoint_dir is not None:
                try:
                    self._create_validation_scatter_plots(
                        predictions=predictions,
                        targets=targets,
                        val_indices=val_indices,
                        checkpoint_dir=checkpoint_dir,
                        sample_size=self.scatter_sample_size
                    )
                except Exception as e:
                    logger.warning(f"Failed to create scatter plots: {e}")
                    logger.warning("Continuing without visualization...")
                    import traceback
                    logger.debug(traceback.format_exc())

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
            import traceback
            logger.error(traceback.format_exc())
            # 返回错误结果
            for task_name in self.task_names:
                results[task_name] = {
                    'mse': 0.0,
                    'valid_samples': 0,
                    'error': str(e)
                }

        return results
    
    def _analyze_prediction_distribution(self,
                                        predictions: np.ndarray,
                                        targets: Dict[str, np.ndarray] = None,
                                        val_indices: np.ndarray = None) -> None:
        """分析预测值分布（增强版：包含偏度、峰度和Target对比）

        Args:
            predictions: 模型预测值
            targets: 目标变量字典（可选）
            val_indices: 验证集索引（可选）
        """
        logger.info("="*65)
        logger.info("DETAILED PREDICTION & TARGET DISTRIBUTION ANALYSIS")
        logger.info("="*65)

        for i, task_name in enumerate(self.task_names):
            task_predictions = predictions[:, i] if predictions.ndim > 1 else predictions

            # 获取对应的target值
            column_name = self.task_column_mapping.get(task_name, task_name)
            task_targets = None
            if targets is not None and column_name in targets and val_indices is not None:
                task_targets = targets[column_name][val_indices]

            logger.info(f"\nTASK: {task_name}")
            logger.info("-" * 65)

            # === Predictions 统计 ===
            pred_mean = task_predictions.mean()
            pred_std = task_predictions.std()
            pred_min = task_predictions.min()
            pred_max = task_predictions.max()
            pred_skew = skew(task_predictions)
            pred_kurt = kurtosis(task_predictions)

            logger.info("Predictions Statistics:")
            logger.info(f"  Mean: {pred_mean:.6f} | Std: {pred_std:.6f} | Min: {pred_min:.6f} | Max: {pred_max:.6f}")

            # 解释偏度
            skew_desc = "symmetric"
            if pred_skew > 0.5:
                skew_desc = "right-skewed (tail to right)"
            elif pred_skew < -0.5:
                skew_desc = "left-skewed (tail to left)"

            # 解释峰度
            kurt_desc = "mesokurtic (normal-like)"
            if pred_kurt > 1:
                kurt_desc = "leptokurtic (heavy tails)"
            elif pred_kurt < -1:
                kurt_desc = "platykurtic (light tails)"

            logger.info(f"  Skewness: {pred_skew:.4f} ({skew_desc})")
            logger.info(f"  Kurtosis: {pred_kurt:.4f} ({kurt_desc})")
            logger.info(f"  Unique values: {np.unique(task_predictions).shape[0]}")

            # === Targets 统计（如果可用）===
            if task_targets is not None:
                target_mean = task_targets.mean()
                target_std = task_targets.std()
                target_min = task_targets.min()
                target_max = task_targets.max()
                target_skew = skew(task_targets)
                target_kurt = kurtosis(task_targets)

                logger.info("\nTargets Statistics:")
                logger.info(f"  Mean: {target_mean:.6f} | Std: {target_std:.6f} | Min: {target_min:.6f} | Max: {target_max:.6f}")

                # 解释偏度
                target_skew_desc = "symmetric"
                if target_skew > 0.5:
                    target_skew_desc = "right-skewed (tail to right)"
                elif target_skew < -0.5:
                    target_skew_desc = "left-skewed (tail to left)"

                # 解释峰度
                target_kurt_desc = "mesokurtic (normal-like)"
                if target_kurt > 1:
                    target_kurt_desc = "leptokurtic (heavy tails)"
                elif target_kurt < -1:
                    target_kurt_desc = "platykurtic (light tails)"

                logger.info(f"  Skewness: {target_skew:.4f} ({target_skew_desc})")
                logger.info(f"  Kurtosis: {target_kurt:.4f} ({target_kurt_desc})")

                # === Prediction vs Target 对比 ===
                logger.info("\nPrediction vs Target Comparison:")

                # Mean差异
                mean_diff = pred_mean - target_mean
                mean_diff_pct = (mean_diff / target_mean * 100) if target_mean != 0 else 0
                mean_comp = "pred > target" if mean_diff > 0 else "pred < target" if mean_diff < 0 else "equal"
                logger.info(f"  Mean Difference: {abs(mean_diff):.6f} ({mean_comp} by {abs(mean_diff_pct):.2f}%)")

                # Std差异
                std_diff = pred_std - target_std
                std_diff_pct = (std_diff / target_std * 100) if target_std != 0 else 0
                std_comp = "pred more variable" if std_diff > 0 else "pred less variable" if std_diff < 0 else "same variability"
                logger.info(f"  Std Difference: {abs(std_diff):.6f} ({std_comp} by {abs(std_diff_pct):.2f}%)")

                # 分布形状对比
                skew_similar = abs(pred_skew - target_skew) < 0.5
                kurt_similar = abs(pred_kurt - target_kurt) < 1.0
                if skew_similar and kurt_similar:
                    logger.info(f"  Distribution Shape: Similar (both {skew_desc})")
                else:
                    logger.info(f"  Distribution Shape: Different")
                    if not skew_similar:
                        logger.info(f"    - Skewness differs: pred={pred_skew:.2f} vs target={target_skew:.2f}")
                    if not kurt_similar:
                        logger.info(f"    - Kurtosis differs: pred={pred_kurt:.2f} vs target={target_kurt:.2f}")

            # === 预测值质量检查 ===
            if pred_std < 1e-8:
                logger.warning(f"  ⚠️  CONSTANT PREDICTIONS for {task_name}!")
            elif np.unique(task_predictions).shape[0] < 5:
                logger.warning(f"  ⚠️  LIMITED PREDICTION DIVERSITY for {task_name}")

        logger.info("\n" + "="*65)

    def _create_validation_scatter_plots(self,
                                         predictions: np.ndarray,
                                         targets: Dict[str, np.ndarray],
                                         val_indices: np.ndarray,
                                         checkpoint_dir: str,
                                         sample_size: int = 1000) -> None:
        """创建验证集散点图（Predictions vs Targets）

        Args:
            predictions: 模型预测值
            targets: 目标变量字典
            val_indices: 验证集索引
            checkpoint_dir: checkpoint目录路径
            sample_size: 采样数量（默认1000）
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available - skipping scatter plot generation")
            return

        logger.info("\n" + "="*65)
        logger.info("CREATING VALIDATION SCATTER PLOTS")
        logger.info("="*65)

        # 创建plots子目录
        plots_dir = Path(checkpoint_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # 确定采样策略
        n_samples = predictions.shape[0]
        if n_samples <= sample_size:
            # 样本数量不超过采样大小，使用全部样本
            sample_indices = np.arange(n_samples)
            logger.info(f"Using all {n_samples} validation samples for visualization")
        else:
            # 随机采样
            np.random.seed(42)  # 设置随机种子确保可重复
            sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
            logger.info(f"Randomly sampled {sample_size} out of {n_samples} validation samples")

        # 为每个task创建散点图
        for i, task_name in enumerate(self.task_names):
            try:
                # 获取task的预测值
                task_predictions = predictions[:, i] if predictions.ndim > 1 else predictions
                task_predictions_sampled = task_predictions[sample_indices]

                # 获取task的目标值
                column_name = self.task_column_mapping.get(task_name, task_name)
                if column_name not in targets:
                    logger.warning(f"Target column {column_name} not found for task {task_name}, skipping scatter plot")
                    continue

                task_targets = targets[column_name][val_indices]
                task_targets_sampled = task_targets[sample_indices]

                # 计算R²分数用于显示
                try:
                    r2 = r2_score(task_targets_sampled, task_predictions_sampled)
                except:
                    r2 = 0.0

                # 创建图表
                fig, ax = plt.subplots(figsize=(10, 8))

                # 绘制散点
                ax.scatter(task_targets_sampled, task_predictions_sampled,
                          alpha=0.5, s=30, c='#1f77b4', edgecolors='none')

                # 添加y=x参考线（完美预测线）
                min_val = min(task_targets_sampled.min(), task_predictions_sampled.min())
                max_val = max(task_targets_sampled.max(), task_predictions_sampled.max())
                ax.plot([min_val, max_val], [min_val, max_val],
                       'r--', lw=2, label='Perfect Prediction (y=x)', alpha=0.7)

                # 设置标签和标题
                ax.set_xlabel('Target Value', fontsize=12, fontweight='bold')
                ax.set_ylabel('Predicted Value', fontsize=12, fontweight='bold')
                ax.set_title(f'Task: {task_name} - Validation Set\n(n={len(sample_indices)} samples, R²={r2:.4f})',
                           fontsize=14, fontweight='bold', pad=15)

                # 添加网格
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

                # 添加图例
                ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

                # 设置坐标轴范围相同（保证对角线是45度）
                ax.set_aspect('equal', adjustable='box')

                # 优化布局
                plt.tight_layout()

                # 保存图表
                plot_filename = f"validation_scatter_{task_name}.png"
                plot_path = plots_dir / plot_filename
                plt.savefig(plot_path, dpi=self.plot_dpi, bbox_inches='tight')
                plt.close(fig)

                logger.info(f"✅ Saved scatter plot: {plot_path} ({len(sample_indices)} samples)")

            except Exception as e:
                logger.error(f"Failed to create scatter plot for task {task_name}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue

        logger.info("="*65 + "\n")

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