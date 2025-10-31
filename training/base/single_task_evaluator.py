#!/usr/bin/env python3
"""
单任务学习评估器基类

提供所有单任务训练器的通用评估功能：
- 统一模型性能评估
- 回归和排序指标计算
- 预测分布分析
- 结果保存和报告生成
- 多模型对比分析

消除单任务训练脚本中的评估代码重复
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


class BaseSingleTaskEvaluator:
    """单任务学习评估器基类
    
    封装所有单任务训练器通用的评估逻辑，包括：
    - 模型性能评估
    - 指标计算和分析
    - 预测分布分析
    - 结果保存和报告
    - 多模型对比
    """
    
    def __init__(self, 
                 output_dir: str,
                 model_type: str = 'unknown'):
        """初始化评估器
        
        Args:
            output_dir: 结果输出目录
            model_type: 模型类型名称
        """
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized single-task evaluator")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model type: {model_type}")
    
    def evaluate_model(self, 
                      model,
                      X_test: Dict[str, np.ndarray], 
                      y_test: np.ndarray) -> Dict[str, float]:
        """评估单任务模型性能
        
        Args:
            model: 训练好的模型
            X_test: 测试集输入
            y_test: 测试集标签
            
        Returns:
            评估结果字典
        """
        logger.info(f"📊 Evaluating {self.model_type} model...")
        
        try:
            # 处理'type'字段重命名问题（与训练时保持一致）
            if 'type' in X_test and 'item_type' not in X_test:
                logger.debug("Renaming 'type' to 'item_type' in test data for consistency")
                X_test['item_type'] = X_test.pop('type')
            
            # 获取模型预测
            predictions = self._get_model_predictions(model, X_test)
            
            # 分析预测值分布
            self._analyze_prediction_distribution(predictions, y_test)
            
            # 计算评估指标
            results = self._calculate_metrics(y_test, predictions)
            
            # 记录评估结果
            self._log_evaluation_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {
                'model': self.model_type,
                'mse': 0.0,
                'test_samples': len(y_test),
                'error': str(e)
            }
    
    def _get_model_predictions(self, model, X_test: Dict[str, np.ndarray]) -> np.ndarray:
        """获取模型预测值"""
        # 获取模型的特征名称
        if hasattr(model, 'feature_index'):
            feature_names = list(model.feature_index.keys())
        else:
            feature_names = list(X_test.keys())
        
        logger.debug(f"Model expects features: {feature_names}")
        
        # 准备测试数据
        test_model_input = {}
        for name in feature_names:
            if name in X_test:
                test_model_input[name] = X_test[name].astype(np.float32)
            else:
                logger.warning(f"Feature {name} not found in test data")
        
        # 进行预测
        predictions = model.predict(test_model_input, batch_size=256)
        return predictions.flatten()
    
    def _analyze_prediction_distribution(self, predictions: np.ndarray, y_true: np.ndarray) -> None:
        """分析预测值分布"""
        logger.info("=" * 60)
        logger.info("PREDICTION DISTRIBUTION ANALYSIS")
        logger.info("=" * 60)
        
        logger.info(f"Predictions:")
        logger.info(f"  Mean: {predictions.mean():.6f}")
        logger.info(f"  Std: {predictions.std():.6f}")
        logger.info(f"  Min: {predictions.min():.6f}")
        logger.info(f"  Max: {predictions.max():.6f}")
        logger.info(f"  Unique values: {np.unique(predictions).shape[0]}")
        
        logger.info(f"Ground Truth:")
        logger.info(f"  Mean: {y_true.mean():.6f}")
        logger.info(f"  Std: {y_true.std():.6f}")
        logger.info(f"  Min: {y_true.min():.6f}")
        logger.info(f"  Max: {y_true.max():.6f}")
        
        # 检查预测值质量
        if predictions.std() < 1e-8:
            logger.warning(f"  ⚠️  CONSTANT PREDICTIONS detected!")
        elif np.unique(predictions).shape[0] < 5:
            logger.warning(f"  ⚠️  LIMITED PREDICTION DIVERSITY detected")
        
        logger.info("=" * 60)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        try:
            # 回归指标
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # R²评分
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = 0.0
            
            # 排序指标
            try:
                spearman_corr, spearman_p = spearmanr(y_true, y_pred)
                kendall_corr, kendall_p = kendalltau(y_true, y_pred)
            except:
                spearman_corr = spearman_p = 0.0
                kendall_corr = kendall_p = 0.0
            
            # 构建结果字典
            results = {
                'model': self.model_type,
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'spearman_correlation': float(spearman_corr),
                'spearman_pvalue': float(spearman_p),
                'kendall_tau': float(kendall_corr),
                'kendall_pvalue': float(kendall_p),
                'test_samples': int(len(y_true))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {
                'model': self.model_type,
                'mse': 0.0,
                'test_samples': int(len(y_true)),
                'error': str(e)
            }
    
    def _log_evaluation_results(self, results: Dict[str, float]) -> None:
        """记录评估结果"""
        if 'error' in results:
            logger.error(f"❌ {self.model_type} evaluation failed: {results['error']}")
            return
        
        logger.info(f"✅ {self.model_type} evaluation results:")
        logger.info(f"   - MSE: {results['mse']:.6f}")
        logger.info(f"   - MAE: {results['mae']:.6f}")
        logger.info(f"   - R²: {results['r2']:.4f}")
        logger.info(f"   - Spearman: {results['spearman_correlation']:.4f} (p={results['spearman_pvalue']:.4f})")
        logger.info(f"   - Kendall Tau: {results['kendall_tau']:.4f} (p={results['kendall_pvalue']:.4f})")
        logger.info(f"   - Test samples: {results['test_samples']}")
    
    def save_results(self, 
                    results: Dict[str, Any],
                    model = None,
                    preprocessors: Optional[Dict[str, Any]] = None,
                    additional_info: Optional[Dict[str, Any]] = None,
                    feature_columns: Optional[List] = None,
                    model_config: Optional[Dict[str, Any]] = None,
                    training_config: Optional[Dict[str, Any]] = None,
                    input_path: Optional[str] = None) -> str:
        """保存训练结果（增强版：完整checkpoint结构）
        
        Args:
            results: 评估结果字典
            model: 模型实例（可选）
            preprocessors: 预处理器（可选）
            additional_info: 额外信息（可选）
            feature_columns: DeepCTR特征列定义（可选）
            model_config: 模型配置（可选）
            training_config: 训练配置（可选）
            input_path: 输入路径（可选）
            
        Returns:
            checkpoint目录路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建checkpoint目录
        checkpoint_dir = self.output_dir / f"checkpoint_{self.model_type.lower()}_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating single-task checkpoint directory: {checkpoint_dir}")
        
        # 构建完整结果字典
        full_results = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'task_type': 'single_task',  # 标识为单任务
            'evaluation_results': results,
            'model_config': model_config or {},
            'training_config': training_config or {},
            'additional_info': additional_info or {},
            'input_path': input_path or 'unknown',
            'checkpoint_dir': str(checkpoint_dir)
        }
        
        # 1. 保存训练信息JSON
        training_info_file = checkpoint_dir / "training_info.json"
        try:
            with open(training_info_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            logger.info(f"✅ Training info saved to {training_info_file}")
        except TypeError as e:
            # 如果仍有序列化问题，尝试清理数据
            logger.warning(f"JSON serialization failed: {e}, attempting to clean data...")
            cleaned_results = self._make_json_serializable(full_results)
            with open(training_info_file, 'w') as f:
                json.dump(cleaned_results, f, indent=2)
            logger.info(f"✅ Training info saved to {training_info_file} (with data cleaning)")
        
        # 2. 保存模型权重
        if model is not None:
            model_file = checkpoint_dir / "model.pth"
            torch.save(model.state_dict(), model_file)
            logger.info(f"✅ Model weights saved to {model_file}")
            
            # 尝试保存完整模型（用于快速加载）
            complete_model_file = checkpoint_dir / "complete_model.pth"
            try:
                torch.save(model, complete_model_file)
                logger.info(f"✅ Complete model saved to {complete_model_file}")
            except (AttributeError, pickle.PicklingError) as e:
                logger.warning(f"⚠️  Could not save complete model: {e}")
                logger.info("   This is normal for models with custom components.")
                # 删除可能创建的损坏文件
                if complete_model_file.exists():
                    complete_model_file.unlink()
        
        # 3. 保存模型配置（独立文件，便于重建模型）
        if model_config:
            model_config_file = checkpoint_dir / "model_config.json"
            enhanced_model_config = {
                **model_config,
                'model_class': model.__class__.__name__ if model else None,
                'device': str(model.device) if model and hasattr(model, 'device') else 'cpu',
                'task_type': 'single_task',
                'label_column': getattr(self, 'label_column', 'ctr')
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
        
        # 5. 保存预处理器
        if preprocessors is not None:
            preprocessor_file = checkpoint_dir / "preprocessors.pkl"
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(preprocessors, f)
            logger.info(f"✅ Preprocessors saved to {preprocessor_file}")
        
        # 6. 创建checkpoint元数据文件（便于版本控制）
        metadata = {
            'version': '1.0',
            'created_at': timestamp,
            'model_type': self.model_type,
            'task_type': 'single_task',
            'files': {
                'model_weights': 'model.pth',
                'complete_model': 'complete_model.pth' if (checkpoint_dir / 'complete_model.pth').exists() else None,
                'model_config': 'model_config.json' if model_config else None,
                'feature_columns': 'feature_columns.json' if feature_columns else None,
                'preprocessors': 'preprocessors.pkl' if preprocessors else None,
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
        logger.info(f"📦 SINGLE-TASK CHECKPOINT SAVED SUCCESSFULLY")
        logger.info(f"📁 Directory: {checkpoint_dir}")
        logger.info(f"📊 Model type: {self.model_type}")
        logger.info(f"🎯 Task type: Single-task")
        logger.info("="*60)
        
        return str(checkpoint_dir)
    
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
    
    def _make_json_serializable(self, obj):
        """递归地将对象转换为JSON可序列化格式
        
        Args:
            obj: 要转换的对象
            
        Returns:
            JSON可序列化的对象
        """
        if obj is None:
            return None
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._make_json_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            # 对象有属性字典，尝试序列化其属性
            return {str(key): self._make_json_serializable(value) for key, value in obj.__dict__.items()}
        else:
            # 其他类型转换为字符串
            return str(obj)
    
    def compare_models(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """对比多个模型的结果
        
        Args:
            all_results: 所有模型的结果字典
            
        Returns:
            对比分析结果
        """
        logger.info("\n📊 GENERATING MODEL COMPARISON...")
        
        # 筛选成功的模型
        successful_models = {}
        failed_models = {}
        
        for model_name, results in all_results.items():
            if 'error' not in results:
                successful_models[model_name] = results
            else:
                failed_models[model_name] = results
        
        if not successful_models:
            logger.warning("⚠️ No successful models to compare")
            return {'error': 'No successful models'}
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(successful_models, failed_models)
        
        # 保存对比报告
        self._save_comparison_report(comparison_report, all_results)
        
        return comparison_report
    
    def _generate_comparison_report(self, successful_models: Dict, failed_models: Dict) -> Dict[str, Any]:
        """生成对比报告"""
        import pandas as pd
        
        # 创建对比数据
        comparison_data = []
        for model_name, results in successful_models.items():
            comparison_data.append({
                'Model': model_name,
                'MSE': f"{results['mse']:.6f}",
                'MAE': f"{results['mae']:.6f}",
                'R²': f"{results['r2']:.4f}",
                'Spearman': f"{results['spearman_correlation']:.4f}",
                'Kendall': f"{results['kendall_tau']:.4f}",
                'Time(s)': f"{results.get('training_time_seconds', 0):.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 模型推荐
        recommendations = self._generate_recommendations(successful_models)
        
        # 控制台输出
        self._print_comparison_table(df)
        self._print_recommendations(recommendations)
        
        return {
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'comparison_table': df.to_dict('records'),
            'recommendations': recommendations,
            'failed_model_details': failed_models
        }
    
    def _generate_recommendations(self, successful_models: Dict) -> Dict[str, str]:
        """生成模型推荐"""
        if not successful_models:
            return {}
        
        models_list = [(name, results) for name, results in successful_models.items()]
        
        recommendations = {}
        
        # 按不同指标推荐
        metrics = [
            ('mse', '最低MSE (最佳回归精度)', False),
            ('r2', '最高R² (最佳拟合度)', True),
            ('spearman_correlation', '最高Spearman (最佳排序能力)', True),
            ('training_time_seconds', '最快训练速度', False)
        ]
        
        for metric, description, reverse in metrics:
            try:
                sorted_models = sorted(models_list, 
                                     key=lambda x: abs(x[1].get(metric, 0)), reverse=reverse)
                best_model = sorted_models[0]
                value = best_model[1].get(metric, 0)
                recommendations[description] = f"{best_model[0]} ({value:.4f})"
            except:
                recommendations[description] = "N/A"
        
        return recommendations
    
    def _print_comparison_table(self, df) -> None:
        """打印对比表格"""
        logger.info("\n📋 MODEL COMPARISON RESULTS:")
        logger.info("-" * 100)
        
        if not df.empty:
            # 打印表格
            for i, row in df.iterrows():
                if i == 0:
                    # 打印表头
                    headers = " | ".join([f"{col:>12}" for col in df.columns])
                    logger.info(headers)
                    logger.info("-" * len(headers))
                
                # 打印数据行
                data_row = " | ".join([f"{str(row[col]):>12}" for col in df.columns])
                logger.info(data_row)
    
    def _print_recommendations(self, recommendations: Dict[str, str]) -> None:
        """打印推荐信息"""
        if recommendations:
            logger.info("\n🏆 MODEL RECOMMENDATIONS:")
            for description, recommendation in recommendations.items():
                logger.info(f"   • {description}: {recommendation}")
        
        logger.info(f"\n💡 RECOMMENDATION FOR CTR REGRESSION:")
        logger.info(f"   推荐使用 DeepFM 或 PNN，它们在CTR回归任务中表现较好")
        logger.info(f"   如果注重排序能力，关注 Spearman 相关系数")
        logger.info(f"   如果注重预测精度，关注 MSE 和 R² 指标")
    
    def _save_comparison_report(self, comparison_report: Dict, all_results: Dict) -> None:
        """保存对比报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CSV格式
        if 'comparison_table' in comparison_report:
            import pandas as pd
            df = pd.DataFrame(comparison_report['comparison_table'])
            csv_file = self.output_dir / f"comparison_report_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"📄 CSV report saved: {csv_file}")
        
        # JSON格式（详细结果）
        json_file = self.output_dir / f"comparison_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'comparison_report': comparison_report,
                'all_results': all_results,
                'timestamp': timestamp
            }, f, indent=2)
        logger.info(f"📄 JSON report saved: {json_file}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """打印结果摘要"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 60)
        
        if 'error' in results:
            print(f"❌ Evaluation failed: {results['error']}")
        else:
            print(f"Model: {results.get('model', 'Unknown')}")
            print(f"MSE: {results.get('mse', 0):.6f}")
            print(f"R²: {results.get('r2', 0):.4f}")
            print(f"Spearman: {results.get('spearman_correlation', 0):.4f}")
            print(f"Kendall: {results.get('kendall_tau', 0):.4f}")
            print(f"Test samples: {results.get('test_samples', 0)}")
        
        print("=" * 60)