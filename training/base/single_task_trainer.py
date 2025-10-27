#!/usr/bin/env python3
"""
单任务学习训练基类

提供所有单任务训练器的统一训练框架：
- 8步统一训练流程
- 组合数据处理、特征处理、评估器
- 抽象模型创建接口
- 统一的训练和编译逻辑
- 完整的训练生命周期管理

消除单任务训练脚本中的核心逻辑重复
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# 导入基础组件
from .single_task_data_processor import BaseSingleTaskDataProcessor
from .single_task_feature_processor import BaseSingleTaskFeatureProcessor
from .single_task_evaluator import BaseSingleTaskEvaluator

logger = logging.getLogger(__name__)


class BaseSingleTaskTrainer(ABC):
    """单任务学习训练基类
    
    提供统一的8步训练流程和通用训练逻辑，子类只需实现模型特定的配置。
    
    统一训练流程：
    1. 加载数据
    2. 准备特征
    3. 分割数据
    4. 创建模型
    5. 编译模型
    6. 训练模型
    7. 评估模型
    8. 保存结果
    """
    
    # 支持的模型列表（子类可覆盖）
    SUPPORTED_MODELS = ['PNN', 'DeepFM', 'WDL', 'DCN', 'xDeepFM', 'FiBiNET', 'AutoInt', 'CCPM']
    
    def __init__(self,
                 input_path: str,
                 model_type: str = 'PNN',
                 feature_config: dict = None,
                 sample_size: Optional[int] = None,
                 output_dir: str = "models/single_task",
                 label_column: str = 'ctr',
                 # 数据处理参数
                 filter_zeros: bool = False,
                 min_impression_threshold: int = 5000,
                 # 特征处理参数
                 use_pca: bool = False,
                 pca_components: int = 256,
                 use_fp16: bool = False,
                 # 训练参数
                 use_early_stopping: bool = True,
                 early_stopping_patience: int = 10,
                 early_stopping_monitor: str = 'loss',
                 early_stopping_min_delta: float = 0.0001,
                 restore_best_weights: bool = True,
                 save_best_model: bool = False):
        """初始化单任务训练器
        
        Args:
            input_path: 输入文件或目录路径
            model_type: 模型类型
            feature_config: 特征配置字典
            sample_size: 随机采样文件数量
            output_dir: 输出目录
            label_column: 标签列名
            filter_zeros: 是否过滤全零CLIP特征样本
            min_impression_threshold: 最小曝光量阈值
            use_pca: 是否对CLIP特征使用PCA降维
            pca_components: PCA组件数量
            use_fp16: 是否使用float16数据类型
            use_early_stopping: 是否启用早停机制
            early_stopping_patience: 早停耐心值
            early_stopping_monitor: 早停监控指标
            early_stopping_min_delta: 早停最小改善幅度
            restore_best_weights: 是否恢复最佳权重
            save_best_model: 是否保存最佳模型
        """
        self.input_path = input_path
        self.model_type = model_type.upper()
        self.feature_config = feature_config or {}
        self.output_dir = output_dir
        self.label_column = label_column
        
        # 早停机制相关参数
        self.use_early_stopping = use_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_min_delta = early_stopping_min_delta
        self.restore_best_weights = restore_best_weights
        self.save_best_model = save_best_model
        
        # 初始化组件
        self.data_processor = BaseSingleTaskDataProcessor(
            input_path=input_path,
            sample_size=sample_size,
            min_impression_threshold=min_impression_threshold,
            filter_zeros=filter_zeros,
            label_column=label_column
        )
        
        self.feature_processor = BaseSingleTaskFeatureProcessor(
            feature_config=feature_config,
            use_pca=use_pca,
            pca_components=pca_components,
            use_fp16=use_fp16
        )
        
        self.evaluator = BaseSingleTaskEvaluator(
            output_dir=output_dir,
            model_type=model_type
        )
        
        # 训练状态
        self.model = None
        self.data = None
        self.device = self._detect_device()
        
        logger.info(f"🚀 Initialized {self.__class__.__name__}")
        logger.info(f"📁 Input path: {input_path}")
        logger.info(f"📊 Model type: {model_type}")
        logger.info(f"🖥️ Device: {self.device}")
        if self.use_early_stopping:
            logger.info(f"⏹️ Early stopping enabled: patience={self.early_stopping_patience}, monitor={self.early_stopping_monitor}")
        else:
            logger.info(f"⏹️ Early stopping disabled")
    
    def _detect_device(self) -> str:
        """自动检测可用设备"""
        if torch.backends.mps.is_available():
            device = 'mps'
            logger.info("✨ Using MPS acceleration on Apple Silicon")
        elif torch.cuda.is_available():
            device = 'cuda'
            logger.info("🚀 Using CUDA acceleration")
        else:
            device = 'cpu'
            logger.info("💻 Using CPU")
        return device
    
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
        """获取模型配置（子类可覆盖）"""
        return {
            'model_type': self.model_type,
            'device': self.device,
            'dnn_hidden_units': [128, 64, 32],
            'dnn_dropout': 0.3
        }
    
    def compile_model(self, learning_rate: float = 0.001) -> None:
        """编译模型（统一逻辑）"""
        logger.info("🔧 Compiling model...")
        
        # 编译模型 - 确保包含验证指标
        self.model.compile("adam", "mse", metrics=['mae', 'mse'])
        
        logger.info(f"✅ Model compiled with Adam optimizer (lr={learning_rate})")
    
    def train_model(self, 
                   X_train: Dict[str, np.ndarray], 
                   y_train: np.ndarray,
                   X_val: Dict[str, np.ndarray], 
                   y_val: np.ndarray,
                   epochs: int = 10,
                   batch_size: int = 256) -> Dict[str, Any]:
        """训练模型（统一逻辑）"""
        logger.info("🚀 Starting model training...")
        
        # 确保标签是float32（MPS不支持float64）
        y_train_float32 = y_train.astype(np.float32)
        y_val_float32 = y_val.astype(np.float32)
        
        # 创建callbacks列表
        callbacks = self._setup_callbacks()
        
        # 训练模型
        logger.info(f"Training config: {epochs} epochs, batch_size={batch_size}")
        logger.info(f"Callbacks: {len(callbacks)} callbacks enabled")
        
        try:
            history = self.model.fit(
                X_train, y_train_float32, 
                batch_size=batch_size, 
                epochs=epochs,
                validation_data=(X_val, y_val_float32), 
                verbose=1,
                callbacks=callbacks if callbacks else None
            )
            
            logger.info("✅ Training completed successfully")
            
            # 将History对象转换为可JSON序列化的格式
            serializable_history = {}
            if hasattr(history, 'history') and history.history:
                # 提取所有可用的训练指标
                for key, values in history.history.items():
                    # 确保values是列表且可序列化
                    if isinstance(values, list):
                        serializable_history[key] = [float(v) if isinstance(v, (int, float)) else str(v) for v in values]
                    else:
                        serializable_history[key] = str(values)
            
            return {
                'history': serializable_history,
                'epochs_completed': len(history.history.get('loss', [])) if hasattr(history, 'history') else epochs,
                'early_stopped': self.use_early_stopping and len(history.history.get('loss', [])) < epochs if hasattr(history, 'history') else False
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _setup_callbacks(self) -> List:
        """设置训练回调"""
        callbacks = []
        
        if self.use_early_stopping:
            logger.info(f"🔧 Setting up EarlyStopping callback:")
            logger.info(f"   Monitor: {self.early_stopping_monitor}")
            logger.info(f"   Patience: {self.early_stopping_patience}")
            logger.info(f"   Min delta: {self.early_stopping_min_delta}")
            
            # 确保监控指标存在
            monitor_metric = self.early_stopping_monitor
            if monitor_metric.startswith('val_') and monitor_metric != 'val_loss':
                logger.warning(f"⚠️ Metric '{monitor_metric}' may not be available, using 'loss' instead")
                monitor_metric = 'loss'
            
            early_stopping = EarlyStopping(
                monitor=monitor_metric,
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                verbose=1,
                restore_best_weights=False,  # 禁用以避免PyTorch兼容性问题
                mode='min'
            )
            
            if self.restore_best_weights:
                logger.warning("⚠️ restore_best_weights is not supported with deepctr-torch, disabled")
            callbacks.append(early_stopping)
        
        # 添加模型检查点（可选）
        if self.save_best_model:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            checkpoint_path = f"{self.output_dir}/best_{self.model_type}_{timestamp}.pth"
            
            logger.info(f"🔧 Setting up ModelCheckpoint: {checkpoint_path}")
            
            monitor_metric = self.early_stopping_monitor if self.use_early_stopping else 'loss'
            
            model_checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=monitor_metric,
                verbose=1,
                save_best_only=True,
                mode='min'
            )
            callbacks.append(model_checkpoint)
        
        return callbacks
    
    def split_data(self, 
                   feature_dict: Dict[str, np.ndarray], 
                   labels: np.ndarray,
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Tuple:
        """分割数据集"""
        logger.info(f"🔀 Splitting data (test={test_size:.1%}, val={val_size:.1%})...")
        
        # 第一次分割：训练+验证 vs 测试
        indices = np.arange(len(labels))
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42
        )
        
        # 第二次分割：训练 vs 验证
        val_size_adjusted = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size_adjusted, random_state=42
        )
        
        # 创建数据集
        X_train = {k: v[train_idx] for k, v in feature_dict.items()}
        X_val = {k: v[val_idx] for k, v in feature_dict.items()}
        X_test = {k: v[test_idx] for k, v in feature_dict.items()}
        
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        y_test = labels[test_idx]
        
        logger.info(f"✅ Data split completed:")
        logger.info(f"   - Train: {len(y_train):,} samples")
        logger.info(f"   - Val: {len(y_val):,} samples")
        logger.info(f"   - Test: {len(y_test):,} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def run(self, 
           epochs: int = 10,
           batch_size: int = 256,
           learning_rate: float = 0.001,
           test_size: float = 0.2,
           val_size: float = 0.1) -> Dict[str, Any]:
        """运行完整的单任务训练流程（8步统一流程）
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            完整结果字典
        """
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"Starting {self.__class__.__name__} Single-Task Training")
        logger.info("=" * 80)
        
        try:
            # 1. 加载数据
            logger.info("Step 1/8: Loading data...")
            df = self.data_processor.load_data()
            self.data = df
            
            # 2. 准备特征
            logger.info("Step 2/8: Preparing features...")
            feature_dict, labels = self.feature_processor.prepare_features(df)
            
            # 3. 分割数据
            logger.info("Step 3/8: Splitting data...")
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(
                feature_dict, labels, test_size, val_size
            )
            
            # 4. 创建模型
            logger.info("Step 4/8: Creating model...")
            feature_columns = self.feature_processor.build_deepctr_features()
            self.model = self.create_model(feature_columns)
            
            # 5. 编译模型
            logger.info("Step 5/8: Compiling model...")
            self.compile_model(learning_rate)
            
            # 6. 准备模型输入
            train_model_input = self.feature_processor.prepare_model_input(X_train, feature_columns)
            val_model_input = self.feature_processor.prepare_model_input(X_val, feature_columns)
            test_model_input = self.feature_processor.prepare_model_input(X_test, feature_columns)
            
            # 7. 训练模型
            logger.info("Step 6/8: Training model...")
            training_info = self.train_model(
                train_model_input, y_train, val_model_input, y_val, epochs, batch_size
            )
            
            # 8. 评估模型
            logger.info("Step 7/8: Evaluating model...")
            evaluation_results = self.evaluator.evaluate_model(self.model, test_model_input, y_test)
            
            # 9. 保存结果
            logger.info("Step 8/8: Saving results...")
            training_time = time.time() - start_time
            
            additional_info = {
                'training_time_seconds': training_time,
                'feature_config': self.feature_config,
                'input_path': str(self.input_path),
                'model_config': self.get_model_config(),
                'training_info': training_info,
                'data_info': self.data_processor.get_data_info()
            }
            
            result_file = self.evaluator.save_results(
                results=evaluation_results,
                model=self.model,
                preprocessors=self.feature_processor.get_preprocessors(),
                additional_info=additional_info
            )
            
            # 总时间
            logger.info("=" * 80)
            logger.info(f"✅ {self.__class__.__name__} TRAINING COMPLETED")
            logger.info(f"Total time: {training_time/60:.2f} minutes")
            logger.info(f"Result saved: {result_file}")
            logger.info("=" * 80)
            
            # 打印结果摘要
            self.evaluator.print_summary(evaluation_results)
            
            return {
                'evaluation_results': evaluation_results,
                'training_info': training_info,
                'additional_info': additional_info,
                'result_file': result_file
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_all_models(self, 
                      epochs: int = 10,
                      batch_size: int = 256,
                      learning_rate: float = 0.001) -> Dict[str, Dict]:
        """训练所有支持的模型并生成对比报告
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            所有模型的结果字典
        """
        logger.info("=" * 80)
        logger.info("🚀 RUNNING ALL MODELS COMPARISON")
        logger.info("=" * 80)
        
        all_results = {}
        
        # 只加载一次数据
        logger.info("📚 Loading data once for all models...")
        df = self.data_processor.load_data()
        feature_dict, labels = self.feature_processor.prepare_features(df)
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(feature_dict, labels)
        
        feature_columns = self.feature_processor.build_deepctr_features()
        train_model_input = self.feature_processor.prepare_model_input(X_train, feature_columns)
        val_model_input = self.feature_processor.prepare_model_input(X_val, feature_columns)
        test_model_input = self.feature_processor.prepare_model_input(X_test, feature_columns)
        
        logger.info(f"📊 Will train {len(self.SUPPORTED_MODELS)} models: {', '.join(self.SUPPORTED_MODELS)}")
        
        for i, model_name in enumerate(self.SUPPORTED_MODELS, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 [{i}/{len(self.SUPPORTED_MODELS)}] Training {model_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # 重置模型相关状态
                self.model = None
                original_model_type = self.model_type
                self.model_type = model_name
                self.evaluator.model_type = model_name
                
                # 创建并训练模型
                self.model = self.create_model(feature_columns)
                self.compile_model(learning_rate)
                training_info = self.train_model(
                    train_model_input, y_train, val_model_input, y_val, epochs, batch_size
                )
                
                # 评估模型
                results = self.evaluator.evaluate_model(self.model, test_model_input, y_test)
                results['training_time_seconds'] = time.time() - start_time
                results['feature_config'] = self.feature_config
                results['input_path'] = str(self.input_path)
                
                all_results[model_name] = results
                
                logger.info(f"✅ {model_name} completed in {results['training_time_seconds']:.2f}s")
                logger.info(f"   MSE: {results['mse']:.6f}, R²: {results['r2']:.4f}")
                logger.info(f"   Spearman: {results['spearman_correlation']:.4f}")
                
                # 恢复原始模型类型
                self.model_type = original_model_type
                self.evaluator.model_type = original_model_type
                
            except Exception as e:
                logger.error(f"❌ {model_name} training failed: {e}")
                all_results[model_name] = {
                    'model': model_name,
                    'error': str(e),
                    'training_time_seconds': time.time() - start_time
                }
        
        # 生成对比报告
        comparison_results = self.evaluator.compare_models(all_results)
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ ALL MODELS COMPARISON COMPLETED")
        logger.info(f"{'='*80}")
        
        return all_results