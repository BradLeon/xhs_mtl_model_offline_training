#!/usr/bin/env python3
"""
模型加载器

从checkpoint加载训练好的模型，支持两种加载方式：
1. 快速加载：直接加载complete_model.pth
2. 重建加载：从配置重建模型结构并加载权重
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from deepctr_torch.inputs import SparseFeat, DenseFeat

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器
    
    支持从checkpoint完整恢复模型和预处理器
    """
    
    def __init__(self, checkpoint_dir: str, device: str = 'auto'):
        """初始化模型加载器
        
        Args:
            checkpoint_dir: checkpoint目录路径
            device: 设备类型 ('auto', 'cpu', 'cuda', 'mps')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # 验证checkpoint目录
        if not self.checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # 设置设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"ModelLoader initialized with checkpoint: {checkpoint_dir}")
        logger.info(f"Device: {self.device}")
        
        # 加载元数据
        self._load_metadata()
        
    def _load_metadata(self):
        """加载checkpoint元数据"""
        metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded checkpoint metadata: version {self.metadata.get('version', 'unknown')}")
        else:
            logger.warning("No metadata file found in checkpoint")
            self.metadata = {}
    
    def load_model(self, method: str = 'auto') -> nn.Module:
        """加载模型
        
        Args:
            method: 加载方法
                - 'auto': 自动选择最佳方法（默认）
                - 'complete': 快速加载完整模型
                - 'rebuild': 从配置重建模型（兼容性更好）
                
        Returns:
            加载的模型
        """
        if method == 'auto':
            # 自动模式：优先尝试complete，失败则rebuild
            complete_model_path = self.checkpoint_dir / "complete_model.pth"
            if complete_model_path.exists():
                try:
                    return self._load_complete_model()
                except Exception:
                    return self._rebuild_and_load_model()
            else:
                return self._rebuild_and_load_model()
        elif method == 'complete':
            return self._load_complete_model()
        elif method == 'rebuild':
            return self._rebuild_and_load_model()
        else:
            raise ValueError(f"Unknown loading method: {method}")
    
    def _load_complete_model(self) -> nn.Module:
        """直接加载完整模型（最简单快速）"""
        model_file = self.checkpoint_dir / "complete_model.pth"
        
        if not model_file.exists():
            logger.info("Complete model not found (this is normal for models with custom loss functions)")
            logger.info("Falling back to rebuild method...")
            return self._rebuild_and_load_model()
        
        try:
            logger.info(f"Loading complete model from {model_file}")
            model = torch.load(model_file, map_location=self.device)
            model.eval()
            logger.info(f"✅ Model loaded successfully: {model.__class__.__name__}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load complete model: {e}")
            logger.info("Falling back to rebuild method...")
            return self._rebuild_and_load_model()
    
    def _rebuild_and_load_model(self) -> nn.Module:
        """从配置重建模型并加载权重（更灵活）"""
        # 1. 加载模型配置
        model_config = self._load_model_config()
        
        # 2. 加载特征列定义
        feature_columns = self._load_feature_columns()
        
        # 3. 重建模型
        model = self._create_model(model_config, feature_columns)
        
        # 4. 加载权重
        weights_file = self.checkpoint_dir / "model.pth"
        if weights_file.exists():
            logger.info(f"Loading model weights from {weights_file}")
            state_dict = torch.load(weights_file, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info("✅ Model weights loaded successfully")
        else:
            logger.warning("Model weights not found!")
        
        model.eval()
        return model
    
    def _load_model_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_file = self.checkpoint_dir / "model_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded model config: {config.get('model_type', 'unknown')}")
        return config
    
    def _load_feature_columns(self) -> List:
        """加载并重建特征列定义"""
        columns_file = self.checkpoint_dir / "feature_columns.json"
        if not columns_file.exists():
            raise FileNotFoundError(f"Feature columns not found: {columns_file}")
        
        with open(columns_file, 'r') as f:
            columns_dict = json.load(f)
        
        # 重建特征列对象
        feature_columns = []
        for col_dict in columns_dict:
            if col_dict['type'] == 'SparseFeat':
                feature_columns.append(
                    SparseFeat(
                        name=col_dict['name'],
                        vocabulary_size=col_dict['vocabulary_size'],
                        embedding_dim=col_dict.get('embedding_dim', 8),
                        dtype=col_dict.get('dtype', 'int32')
                    )
                )
            elif col_dict['type'] == 'DenseFeat':
                feature_columns.append(
                    DenseFeat(
                        name=col_dict['name'],
                        dimension=col_dict.get('dimension', 1)
                    )
                )
        
        logger.info(f"Reconstructed {len(feature_columns)} feature columns")
        return feature_columns
    
    def _create_model(self, config: Dict[str, Any], feature_columns: List) -> nn.Module:
        """根据配置创建模型
        
        Args:
            config: 模型配置
            feature_columns: 特征列定义
            
        Returns:
            创建的模型实例
        """
        model_type = config.get('model_type', '').upper()
        tasks = config.get('tasks', [])
        
        # 检查任务类型：单任务 vs 多任务
        task_type = config.get('task_type', 'multi_task')  # 从配置读取任务类型
        
        if task_type == 'single_task':
            # === 单任务模型 ===
            model = self._create_single_task_model(model_type, config, feature_columns)
        else:
            # === 多任务模型 ===
            model = self._create_multi_task_model(model_type, config, feature_columns, tasks)
        
        logger.info(f"Created {model_type} model with {len(tasks) if tasks else 1} tasks")
        return model
    
    def _create_single_task_model(self, model_type: str, config: Dict[str, Any], feature_columns: List) -> nn.Module:
        """创建单任务模型
        
        Args:
            model_type: 模型类型
            config: 模型配置
            feature_columns: 特征列定义
            
        Returns:
            创建的单任务模型
        """
        # 导入DeepCTR模型
        from deepctr_torch.models import DeepFM, PNN, WDL, DCN, xDeepFM, FiBiNET, AutoInt, CCPM
        
        # 获取设备
        device = torch.device(self.device)
        
        # 根据模型类型创建模型
        if model_type in ['DEEPFM', 'DeepFM']:
            model = DeepFM(feature_columns, feature_columns, task='regression', device=device)
        elif model_type == 'PNN':
            model = PNN(feature_columns, task='regression', device=device)
        elif model_type == 'WDL':
            model = WDL(feature_columns, feature_columns, task='regression', device=device)
        elif model_type == 'DCN':
            model = DCN(feature_columns, task='regression', device=device)
        elif model_type in ['XDEEPFM', 'xDeepFM']:
            model = xDeepFM(feature_columns, feature_columns, task='regression', device=device)
        elif model_type in ['FIBINET', 'FiBiNET']:
            model = FiBiNET(feature_columns, feature_columns, task='regression', device=device)
        elif model_type in ['AUTOINT', 'AutoInt']:
            model = AutoInt(feature_columns, task='regression', device=device)
        elif model_type == 'CCPM':
            model = CCPM(feature_columns, task='regression', device=device)
        else:
            logger.warning(f"Unknown single-task model type: {model_type}, falling back to DeepFM")
            model = DeepFM(feature_columns, feature_columns, task='regression', device=device)
        
        logger.info(f"Created single-task {model_type} model")
        return model
    
    def _create_multi_task_model(self, model_type: str, config: Dict[str, Any], feature_columns: List, tasks: List[str]) -> nn.Module:
        """创建多任务模型
        
        Args:
            model_type: 模型类型
            config: 模型配置
            feature_columns: 特征列定义
            tasks: 任务列表
            
        Returns:
            创建的多任务模型
        """
        # 根据模型类型创建相应的模型
        if model_type == 'PNN_MMOE':
            # 尝试导入PNN_MMOE模型
            try:
                from training.base.pnn_mmoe_model import PNN_MMOE
            except ImportError:
                # 如果相对导入失败，尝试添加项目路径
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                from training.base.pnn_mmoe_model import PNN_MMOE
            
            # 提取PNN-MMOE特定配置
            pnn_mmoe_config = config.get('pnn_mmoe_config', {})
            pnn_config = pnn_mmoe_config.get('pnn', {})
            mmoe_config = pnn_mmoe_config.get('mmoe', {})
            
            # 准备必需的参数
            num_tasks = len(tasks)
            # PNN_MMOE期望所有任务都是回归任务，使用'binary'类型
            task_types = ['binary'] * num_tasks
            
            model = PNN_MMOE(
                dnn_feature_columns=feature_columns,  # 修正：使用dnn_feature_columns而不是feature_columns
                num_tasks=num_tasks,                   # 添加：必需参数
                task_types=task_types,                 # 添加：必需参数
                task_names=tasks,                      # 保留：任务名称
                use_inner_product=pnn_config.get('use_inner_product', True),
                use_outter_product=pnn_config.get('use_outter_product', False),
                num_experts=mmoe_config.get('num_experts', 3),
                expert_dnn_hidden_units=tuple(mmoe_config.get('expert_dims', [128, 64])),
                gate_dnn_hidden_units=tuple(mmoe_config.get('gate_dims', [32])),
                tower_dnn_hidden_units=tuple(mmoe_config.get('tower_dims', [64, 32])),
                dnn_dropout=config.get('dropout', 0.1),
                l2_reg_embedding=config.get('l2_reg_embedding', 1e-5),
                l2_reg_dnn=config.get('l2_reg_dnn', 0),
                device=self.device
            )
        elif model_type == 'MMOE':
            from deepctr_torch.models import MMOE
            
            # MMOE配置
            model = MMOE(
                dnn_feature_columns=feature_columns,
                task_names=tasks,
                num_experts=config.get('num_experts', 3),
                expert_dnn_hidden_units=tuple(config.get('expert_dims', [128, 64])),
                gate_dnn_hidden_units=tuple(config.get('gate_dims', [32])),
                tower_dnn_hidden_units=tuple(config.get('tower_dims', [64, 32])),
                dnn_dropout=config.get('dropout', 0.1),
                l2_reg_embedding=config.get('l2_reg_embedding', 1e-5),
                l2_reg_dnn=config.get('l2_reg_dnn', 0),
                device=self.device
            )
        elif model_type == 'PLE':
            from deepctr_torch.models import PLE
            
            # PLE配置
            model = PLE(
                dnn_feature_columns=feature_columns,
                task_names=tasks,
                shared_expert_num=config.get('shared_expert_num', 2),
                specific_expert_num=config.get('specific_expert_num', 1),
                num_levels=config.get('num_levels', 2),
                expert_dnn_hidden_units=tuple(config.get('expert_dims', [128, 64])),
                gate_dnn_hidden_units=tuple(config.get('gate_dims', [32])),
                tower_dnn_hidden_units=tuple(config.get('tower_dims', [64, 32])),
                dnn_dropout=config.get('dropout', 0.1),
                l2_reg_embedding=config.get('l2_reg_embedding', 1e-5),
                l2_reg_dnn=config.get('l2_reg_dnn', 0),
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported multi-task model type: {model_type}")
        
        logger.info(f"Created multi-task {model_type} model with {len(tasks)} tasks")
        return model
    
    def load_preprocessors(self) -> Dict[str, Any]:
        """加载预处理器
        
        Returns:
            预处理器字典，包含：
            - label_encoders: 标签编码器
            - scalers: 标准化器
            - pca_transformers: PCA转换器
            - feature_settings: 特征设置
        """
        preprocessor_file = self.checkpoint_dir / "preprocessors.pkl"
        if not preprocessor_file.exists():
            logger.warning("Preprocessors not found in checkpoint")
            return {}
        
        with open(preprocessor_file, 'rb') as f:
            preprocessors = pickle.load(f)
        
        logger.info(f"Loaded preprocessors: {list(preprocessors.keys())}")
        return preprocessors
    
    def load_label_normalizer(self):
        """加载标签归一化器
        
        Returns:
            标签归一化器实例，如果不存在则返回None
        """
        normalizer_file = self.checkpoint_dir / "label_normalizer.pkl"
        if not normalizer_file.exists():
            logger.info("No label normalizer found in checkpoint")
            return None
        
        with open(normalizer_file, 'rb') as f:
            normalizer = pickle.load(f)
        
        logger.info("Loaded label normalizer")
        return normalizer
    
    def load_training_info(self) -> Dict[str, Any]:
        """加载训练信息
        
        Returns:
            训练信息字典
        """
        training_info_file = self.checkpoint_dir / "training_info.json"
        if not training_info_file.exists():
            logger.warning("Training info not found in checkpoint")
            return {}
        
        with open(training_info_file, 'r') as f:
            training_info = json.load(f)
        
        logger.info(f"Loaded training info from {training_info.get('timestamp', 'unknown')}")
        return training_info
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取checkpoint摘要信息
        
        Returns:
            checkpoint摘要字典
        """
        summary = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'metadata': self.metadata,
            'files': {}
        }
        
        # 检查各个文件是否存在
        expected_files = [
            'model.pth',
            'complete_model.pth',
            'model_config.json',
            'feature_columns.json',
            'preprocessors.pkl',
            'label_normalizer.pkl',
            'training_info.json',
            'checkpoint_metadata.json'
        ]
        
        for file_name in expected_files:
            file_path = self.checkpoint_dir / file_name
            summary['files'][file_name] = {
                'exists': file_path.exists(),
                'size': file_path.stat().st_size if file_path.exists() else 0
            }
        
        # 加载模型配置摘要
        try:
            config = self._load_model_config()
            summary['model_type'] = config.get('model_type', 'unknown')
            summary['tasks'] = config.get('tasks', [])
            summary['device'] = config.get('device', 'unknown')
        except:
            pass
        
        return summary