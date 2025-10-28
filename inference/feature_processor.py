#!/usr/bin/env python3
"""
推理时特征处理器

处理推理时的特征工程，包括：
- 稀疏特征编码
- 密集特征标准化
- CLIP特征PCA转换
- 特征验证和缺失值处理
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InferenceFeatureProcessor:
    """推理时特征处理器
    
    使用训练时保存的预处理器对新数据进行特征工程
    """
    
    def __init__(self, preprocessors: Dict[str, Any], feature_columns: List):
        """初始化推理特征处理器
        
        Args:
            preprocessors: 训练时保存的预处理器字典
            feature_columns: DeepCTR特征列定义
        """
        self.preprocessors = preprocessors
        self.feature_columns = feature_columns
        
        # 提取各种预处理器
        self.label_encoders = preprocessors.get('label_encoders', {})
        self.scalers = preprocessors.get('scalers', {})
        self.pca_transformers = preprocessors.get('pca_transformers', {})
        self.feature_settings = preprocessors.get('feature_settings', {})
        
        # 构建特征名称列表
        self.feature_names = [col.name for col in feature_columns]
        
        # 分类特征
        self.sparse_features = []
        self.dense_features = []
        self.clip_features = []
        
        self._classify_features()
        
        logger.info(f"Initialized inference feature processor")
        logger.info(f"Total features: {len(self.feature_names)}")
        logger.info(f"Sparse: {len(self.sparse_features)}, Dense: {len(self.dense_features)}, "
                   f"CLIP: {len(self.clip_features)}")
    
    def _classify_features(self):
        """分类特征"""
        for col in self.feature_columns:
            col_name = col.name
            col_type = col.__class__.__name__
            
            if col_type == 'SparseFeat':
                self.sparse_features.append(col_name)
            elif col_type == 'DenseFeat':
                # 判断是否为CLIP特征
                if any(pattern in col_name for pattern in ['_feat_', '_embedding_']):
                    self.clip_features.append(col_name)
                else:
                    self.dense_features.append(col_name)
    
    def process_single(self, note_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """处理单条笔记数据
        
        Args:
            note_data: 笔记数据字典，包含所有原始特征
            
        Returns:
            模型输入字典，key为特征名，value为numpy数组
        """
        # 转换为DataFrame便于处理
        df = pd.DataFrame([note_data])
        return self.process_batch(df)
    
    def process_batch(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """批量处理数据
        
        Args:
            df: 包含多条记录的DataFrame
            
        Returns:
            模型输入字典
        """
        logger.info(f"Processing batch of {len(df)} samples")
        
        # 1. 处理稀疏特征
        for feat in self.sparse_features:
            if feat in df.columns:
                if feat in self.label_encoders:
                    # 使用训练时的编码器
                    encoder = self.label_encoders[feat]
                    # 处理未见过的类别
                    df[feat] = df[feat].apply(
                        lambda x: self._safe_encode(x, encoder, feat)
                    )
                else:
                    logger.warning(f"No encoder found for sparse feature {feat}")
                    # 使用默认值0
                    df[feat] = 0
            else:
                logger.warning(f"Sparse feature {feat} not found in input, using default value 0")
                df[feat] = 0
        
        # 2. 处理密集特征
        for feat in self.dense_features:
            if feat in df.columns:
                if feat in self.scalers:
                    # 使用训练时的标准化器
                    scaler = self.scalers[feat]
                    df[feat] = scaler.transform(df[[feat]].values).flatten()
                else:
                    logger.warning(f"No scaler found for dense feature {feat}")
            else:
                logger.warning(f"Dense feature {feat} not found in input, using default value 0")
                df[feat] = 0.0
        
        # 3. 处理CLIP特征
        if self.clip_features:
            df = self._process_clip_features(df)
        
        # 4. 构建模型输入
        model_input = {}
        for feat_name in self.feature_names:
            if feat_name in df.columns:
                model_input[feat_name] = df[feat_name].values.astype(np.float32)
            else:
                # 对于缺失的特征，使用零值
                logger.warning(f"Feature {feat_name} not found, using zeros")
                model_input[feat_name] = np.zeros(len(df), dtype=np.float32)
        
        logger.info(f"Prepared {len(model_input)} features for inference")
        return model_input
    
    def _safe_encode(self, value: Any, encoder, feature_name: str) -> int:
        """安全编码，处理未见过的类别
        
        Args:
            value: 要编码的值
            encoder: LabelEncoder实例
            feature_name: 特征名称
            
        Returns:
            编码后的整数值
        """
        try:
            # 转换为字符串（与训练时一致）
            str_value = str(value)
            if str_value in encoder.classes_:
                return encoder.transform([str_value])[0]
            else:
                logger.warning(f"Unknown category '{value}' for feature {feature_name}, using default")
                # 使用最频繁的类别或0
                return 0
        except Exception as e:
            logger.error(f"Error encoding {feature_name}: {e}")
            return 0
    
    def _process_clip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理CLIP特征
        
        Args:
            df: 输入DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 识别CLIP特征组
        clip_groups = {}
        for feat in self.clip_features:
            # 提取特征前缀（如cover_image_feat_, title_feat_等）
            prefix = '_'.join(feat.split('_')[:-1]) + '_'
            if prefix not in clip_groups:
                clip_groups[prefix] = []
            clip_groups[prefix].append(feat)
        
        # 对每组CLIP特征应用PCA（如果训练时使用了）
        if self.feature_settings.get('use_pca', False) and self.pca_transformers:
            for prefix, feat_list in clip_groups.items():
                # 查找对应的PCA转换器
                pca_key = None
                for key in self.pca_transformers.keys():
                    if prefix.replace('_feat_', '') in key or key in prefix:
                        pca_key = key
                        break
                
                if pca_key and pca_key in self.pca_transformers:
                    pca = self.pca_transformers[pca_key]
                    
                    # 收集该组的所有特征
                    original_features = []
                    for feat in feat_list:
                        if feat in df.columns:
                            original_features.append(df[feat].values)
                        else:
                            # 使用零值填充缺失特征
                            original_features.append(np.zeros(len(df)))
                    
                    if original_features:
                        # 转换为矩阵并应用PCA
                        feature_matrix = np.column_stack(original_features)
                        transformed = pca.transform(feature_matrix)
                        
                        # 更新DataFrame
                        for i, feat in enumerate(feat_list[:transformed.shape[1]]):
                            df[feat] = transformed[:, i]
                        
                        logger.info(f"Applied PCA to {prefix} features")
        
        # 填充缺失的CLIP特征
        for feat in self.clip_features:
            if feat not in df.columns:
                df[feat] = 0.0
        
        return df
    
    def validate_input(self, data: Union[Dict, pd.DataFrame]) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            是否有效
        """
        # 检查必需的稀疏特征
        required_sparse = self.sparse_features[:5]  # 检查前5个重要特征
        
        if isinstance(data, dict):
            missing = [f for f in required_sparse if f not in data]
        else:
            missing = [f for f in required_sparse if f not in data.columns]
        
        if missing:
            logger.warning(f"Missing important features: {missing}")
        
        return True  # 允许缺失特征，使用默认值
    
    def get_feature_importance(self) -> Dict[str, str]:
        """获取特征重要性信息
        
        Returns:
            特征类型字典
        """
        importance = {}
        
        for feat in self.sparse_features:
            importance[feat] = 'sparse'
        
        for feat in self.dense_features:
            importance[feat] = 'dense'
        
        for feat in self.clip_features:
            importance[feat] = 'clip'
        
        return importance