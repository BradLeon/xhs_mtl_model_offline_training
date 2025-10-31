#!/usr/bin/env python3
"""
单任务学习特征处理基类

提供所有单任务训练器的通用特征处理功能：
- 统一特征分类和选择逻辑
- 统一稀疏/密集特征处理
- CLIP特征PCA降维
- 特征标准化和编码
- DeepCTR格式转换

消除单任务训练脚本中的特征处理代码重复
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

logger = logging.getLogger(__name__)


class BaseSingleTaskFeatureProcessor:
    """单任务学习特征处理基类
    
    封装所有单任务训练器通用的特征处理逻辑，包括：
    - 特征分类和选择
    - 稀疏特征编码
    - 密集特征标准化
    - CLIP特征PCA降维
    - DeepCTR格式转换
    """
    
    def __init__(self,
                 feature_config: dict = None,
                 use_pca: bool = False,
                 pca_components: int = 256,
                 use_fp16: bool = False,
                 unified_embedding_dim: int = 8):
        """初始化特征处理器
        
        Args:
            feature_config: 特征配置字典
            use_pca: 是否对CLIP特征使用PCA降维
            pca_components: PCA组件数量
            use_fp16: 是否使用float16数据类型
            unified_embedding_dim: 统一的embedding维度（PNN兼容性）
        """
        self.feature_config = feature_config or {}
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_fp16 = use_fp16
        self.unified_embedding_dim = unified_embedding_dim
        
        # 存储预处理器
        self.label_encoders = {}
        self.scalers = {}
        self.pca_transformers = {}
        
        # 特征信息
        self.feature_columns = {}
        self.feature_names = []
        
        logger.info(f"Initialized single-task feature processor")
        logger.info(f"  Use PCA: {use_pca} (components: {pca_components})")
        logger.info(f"  Use FP16: {use_fp16}")
        logger.info(f"  Unified embedding dim: {unified_embedding_dim}")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """准备特征和标签
        
        Args:
            df: 输入DataFrame
            
        Returns:
            (特征字典, 标签数组)
        """
        logger.info("🔧 Preparing features for single-task training...")
        
        # 1. 分类特征
        sparse_features, dense_features, clip_features = self._classify_features(df)
        
        logger.info(f"📊 Feature breakdown:")
        logger.info(f"   - Sparse features ({len(sparse_features)}): {sparse_features}")
        logger.info(f"   - Dense features ({len(dense_features)}): {dense_features[:5]}...")
        logger.info(f"   - CLIP features ({len(clip_features)}): {len(clip_features)} dimensions")
        
        # 2. 处理特征
        feature_dict = {}
        all_dense_features = dense_features + clip_features
        self.feature_columns = {'sparse': sparse_features, 'dense': all_dense_features}
        
        # 3. 处理稀疏特征
        self._process_sparse_features(df, sparse_features, feature_dict)
        
        # 4. 处理密集特征
        self._process_dense_features(df, all_dense_features, feature_dict)
        
        # 5. PCA降维（如果启用）
        if self.use_pca and clip_features:
            self._apply_pca_to_clip_features(clip_features, feature_dict)
        
        # 6. 应用FP16优化（如果启用）
        if self.use_fp16:
            self._apply_fp16_optimization(feature_dict)
        
        # 7. 准备标签
        labels = self._prepare_labels(df)
        
        logger.info(f"✅ Features prepared: {len(feature_dict)} features")
        logger.info(f"📌 Label range: [{labels.min():.4f}, {labels.max():.4f}]")
        
        return feature_dict, labels
    
    def _classify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """分类特征列
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple包含 (稀疏特征列表, 密集特征列表, CLIP特征列表)
        """
        if not self.feature_config:
            # 使用默认分类逻辑
            return self._default_feature_classification(df)
        
        sparse_features = []
        dense_features = []
        clip_features = []
        
        # 根据配置选择稀疏特征
        for feat in self.feature_config.get('sparse', []):
            if feat in df.columns:
                sparse_features.append(feat)
        
        # 根据配置选择密集特征
        for feat in self.feature_config.get('dense', []):
            if feat in df.columns:
                dense_features.append(feat)
        
        # 根据配置选择CLIP特征
        if self.feature_config.get('clip_image', False):
            image_feats = [col for col in df.columns if col.startswith('image_feat_')]
            clip_features.extend(image_feats)
        
        if self.feature_config.get('clip_text', False):
            text_feats = [col for col in df.columns if col.startswith('text_feat_')]
            clip_features.extend(text_feats)
        
        logger.info(f"📋 Using configured feature selection:")
        logger.info(f"   Sparse: {len(sparse_features)} features")
        logger.info(f"   Dense: {len(dense_features)} features")
        logger.info(f"   CLIP: {len(clip_features)} features")
        
        return sparse_features, dense_features, clip_features
    
    def _default_feature_classification(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """默认特征分类逻辑"""
        all_features = list(df.columns)
        
        # 排除目标变量和ID列
        exclude_patterns = [
            'ctr', 'click', 'imp_num', 'impression', 'click_num',
            'note_id', 'user_id'
        ]
        
        feature_candidates = []
        for col in all_features:
            if not any(pattern in col for pattern in exclude_patterns):
                feature_candidates.append(col)
        
        # 按类型分组特征
        sparse_features = []
        dense_features = []
        clip_features = []
        
        for feat in feature_candidates:
            if any(pattern in feat for pattern in ['image_feat_', 'text_feat_']):
                clip_features.append(feat)
            elif any(pattern in feat for pattern in ['nickname', 'title_clean', 'type', 'taxonomy', 'intention']):
                sparse_features.append(feat)
            elif feat in ['title_length']:
                dense_features.append(feat)
        
        # 处理type字段冲突
        if 'type' in sparse_features:
            idx = sparse_features.index('type')
            sparse_features[idx] = 'item_type'
            df['item_type'] = df['type']
            if 'type' in df.columns:
                df = df.drop(columns=['type'])
        
        return sparse_features, dense_features, clip_features
    
    def _process_sparse_features(self, df: pd.DataFrame, sparse_features: List[str], feature_dict: Dict[str, np.ndarray]) -> None:
        """处理稀疏特征"""
        for feat in sparse_features:
            if feat not in df.columns:
                logger.warning(f"Sparse feature {feat} not found in dataframe")
                continue
            
            # 处理缺失值并转换为字符串
            feat_values = df[feat].fillna('missing').astype(str)
            
            # 标签编码
            if feat not in self.label_encoders:
                self.label_encoders[feat] = LabelEncoder()
                feature_dict[feat] = self.label_encoders[feat].fit_transform(feat_values)
            else:
                # 处理未见过的类别
                encoded = []
                for val in feat_values:
                    if val in self.label_encoders[feat].classes_:
                        encoded.append(self.label_encoders[feat].transform([val])[0])
                    else:
                        encoded.append(0)  # 未知类别映射为0
                feature_dict[feat] = np.array(encoded)
        
        logger.info(f"✅ Processed {len(sparse_features)} sparse features with LabelEncoder")
    
    def _process_dense_features(self, df: pd.DataFrame, dense_features: List[str], feature_dict: Dict[str, np.ndarray]) -> None:
        """处理密集特征"""
        for feat in dense_features:
            if feat not in df.columns:
                logger.warning(f"Dense feature {feat} not found in dataframe")
                continue
            
            # 确保数值类型并填充缺失值
            feat_values = pd.to_numeric(df[feat], errors='coerce').fillna(0).values.reshape(-1, 1)
            
            # 标准化
            if feat not in self.scalers:
                self.scalers[feat] = StandardScaler()
                feature_dict[feat] = self.scalers[feat].fit_transform(feat_values).flatten()
            else:
                feature_dict[feat] = self.scalers[feat].transform(feat_values).flatten()
        
        logger.info(f"✅ Processed {len(dense_features)} dense features with StandardScaler")
    
    def _apply_pca_to_clip_features(self, clip_features: List[str], feature_dict: Dict[str, np.ndarray]) -> None:
        """对CLIP特征应用PCA降维"""
        logger.info(f"🔄 Applying PCA to CLIP features ({len(clip_features)} → {self.pca_components*2} dimensions)")
        
        # 分离image和text特征
        image_features = [f for f in clip_features if 'image_feat' in f]
        text_features = [f for f in clip_features if 'text_feat' in f]
        
        # 对image特征进行PCA
        if image_features:
            self._apply_pca_to_feature_group(image_features, feature_dict, 'image')
        
        # 对text特征进行PCA
        if text_features:
            self._apply_pca_to_feature_group(text_features, feature_dict, 'text')
        
        # 更新feature_columns
        pca_features = []
        if image_features:
            pca_features.extend([f'pca_image_{i}' for i in range(self.pca_components)])
        if text_features:
            pca_features.extend([f'pca_text_{i}' for i in range(self.pca_components)])
        
        # 移除原始CLIP特征，添加PCA特征
        non_clip_dense = [f for f in self.feature_columns['dense'] if f not in clip_features]
        self.feature_columns['dense'] = non_clip_dense + pca_features
    
    def _apply_pca_to_feature_group(self, features: List[str], feature_dict: Dict[str, np.ndarray], group_name: str) -> None:
        """对特征组应用PCA"""
        if not features:
            return
        
        # 构建特征矩阵
        feature_matrix = np.column_stack([feature_dict[f] for f in features])
        
        # 应用PCA
        pca_key = f'pca_{group_name}'
        if pca_key not in self.pca_transformers:
            self.pca_transformers[pca_key] = PCA(n_components=min(self.pca_components, feature_matrix.shape[1]))
            reduced_features = self.pca_transformers[pca_key].fit_transform(feature_matrix)
            explained_variance = self.pca_transformers[pca_key].explained_variance_ratio_.sum()
            logger.info(f"   {group_name.title()} PCA: {len(features)} → {self.pca_components} dims, "
                       f"variance explained: {explained_variance:.2%}")
        else:
            reduced_features = self.pca_transformers[pca_key].transform(feature_matrix)
        
        # 删除原始特征，添加PCA特征
        for feat in features:
            if feat in feature_dict:
                del feature_dict[feat]
        
        for i in range(reduced_features.shape[1]):
            feature_dict[f'pca_{group_name}_{i}'] = reduced_features[:, i]
    
    def _apply_fp16_optimization(self, feature_dict: Dict[str, np.ndarray]) -> None:
        """应用FP16数据类型优化"""
        logger.info("📉 Converting to float16 for memory optimization")
        
        for feat in self.feature_columns['dense']:
            if feat in feature_dict:
                feature_dict[feat] = feature_dict[feat].astype(np.float16)
    
    def _prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """准备标签数据"""
        # 默认使用ctr作为标签
        label_column = 'ctr'
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataframe")
        
        labels = df[label_column].values
        return labels
    
    def build_deepctr_features(self) -> List:
        """构建DeepCTR特征列定义
        
        Returns:
            DeepCTR特征列定义列表
        """
        feature_columns = []
        
        # 稀疏特征
        for feat_name in self.feature_columns['sparse']:
            if feat_name in self.label_encoders:
                vocab_size = len(self.label_encoders[feat_name].classes_)
                
                # 处理特殊字段名（type是Python保留字）
                actual_feat_name = 'item_type' if feat_name == 'type' else feat_name
                
                feature_columns.append(
                    SparseFeat(
                        actual_feat_name, 
                        vocabulary_size=max(vocab_size, 2), 
                        embedding_dim=self.unified_embedding_dim
                    )
                )
        
        # 密集特征
        for feat_name in self.feature_columns['dense']:
            feature_columns.append(DenseFeat(feat_name, 1))
        
        logger.info(f"📋 Built {len(feature_columns)} DeepCTR feature columns")
        logger.info(f"   Sparse: {len(self.feature_columns['sparse'])} features")
        logger.info(f"   Dense: {len(self.feature_columns['dense'])} features")
        
        return feature_columns
    
    def prepare_model_input(self, feature_dict: Dict[str, np.ndarray], feature_columns: List) -> Dict[str, np.ndarray]:
        """准备DeepCTR模型输入格式
        
        Args:
            feature_dict: 特征字典
            feature_columns: DeepCTR特征列定义
            
        Returns:
            模型输入字典
        """
        feature_names = get_feature_names(feature_columns)
        model_input = {}
        
        for name in feature_names:
            # 处理字段重命名
            source_name = 'type' if name == 'item_type' else name
            
            if source_name in feature_dict:
                # 转换为float32以兼容MPS
                model_input[name] = feature_dict[source_name].astype(np.float32)
            elif name in feature_dict:
                model_input[name] = feature_dict[name].astype(np.float32)
            else:
                logger.warning(f"Feature {name} not found in feature dictionary")
        
        return model_input
    
    def get_preprocessors(self) -> Dict[str, Any]:
        """获取所有预处理器（增强版：包含feature_settings）
        
        Returns:
            预处理器字典，包含feature_settings信息用于推理重建
        """
        # 构建feature_settings信息
        feature_settings = {
            'use_pca': self.use_pca,
            'pca_components': self.pca_components,
            'use_fp16': self.use_fp16,
            'unified_embedding_dim': self.unified_embedding_dim,
            'feature_columns': self.feature_columns,
            'feature_config': self.feature_config
        }
        
        # 添加预处理器详细信息
        feature_settings['preprocessor_info'] = {
            'label_encoders': {
                name: {
                    'classes': encoder.classes_.tolist(),
                    'n_classes': len(encoder.classes_)
                } for name, encoder in self.label_encoders.items()
            },
            'scalers': {
                name: {
                    'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                } for name, scaler in self.scalers.items()
            },
            'pca_transformers': {
                name: {
                    'n_components': pca.n_components_,
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                } for name, pca in self.pca_transformers.items()
            }
        }
        
        return {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'pca_transformers': self.pca_transformers,
            'feature_config': self.feature_config,
            'unified_embedding_dim': self.unified_embedding_dim,
            'feature_settings': feature_settings  # 添加：feature_settings信息
        }
    
    def validate_features(self, feature_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """验证特征质量
        
        Args:
            feature_dict: 特征字典
            
        Returns:
            特征质量报告
        """
        report = {
            'total_features': len(feature_dict),
            'sparse_features': len(self.feature_columns['sparse']),
            'dense_features': len(self.feature_columns['dense']),
            'constant_features': [],
            'issues': []
        }
        
        # 检查常数特征
        for feat_name, feat_data in feature_dict.items():
            if np.unique(feat_data).shape[0] <= 1:
                report['constant_features'].append(feat_name)
        
        # 检查常见问题
        if report['total_features'] == 0:
            report['issues'].append('No features found')
        
        if len(report['constant_features']) > 0:
            report['issues'].append(f'{len(report["constant_features"])} constant features detected')
        
        return report