#!/usr/bin/env python3
"""
多任务学习特征处理基类

提供所有MTL训练器的通用特征处理功能：
- 统一特征分类和准备逻辑
- 统一稀疏/密集特征处理
- CLIP特征PCA降维
- 特征标准化和编码
- DeepCTR格式转换

消除MTL训练脚本中的特征处理代码重复
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat

logger = logging.getLogger(__name__)


class BaseMTLFeatureProcessor:
    """多任务学习特征处理基类
    
    封装所有MTL训练器通用的特征处理逻辑，包括：
    - 特征分类和识别
    - 稀疏特征编码
    - 密集特征标准化
    - CLIP特征PCA降维
    - DeepCTR格式转换
    """
    
    def __init__(self,
                 filter_zeros: bool = True,
                 use_pca: bool = False,
                 pca_components: int = 256):
        """初始化特征处理器
        
        Args:
            filter_zeros: 是否过滤全零CLIP特征样本
            use_pca: 是否对CLIP特征使用PCA降维
            pca_components: PCA组件数量
        """
        self.filter_zeros = filter_zeros
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        # 存储预处理器
        self.label_encoders = {}
        self.scalers = {}
        self.pca_transformers = {}
        
        logger.info(f"Initialized MTL feature processor")
        logger.info(f"  Filter zeros: {filter_zeros}")
        logger.info(f"  Use PCA: {use_pca} (components: {pca_components})")
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List, Dict]:
        """准备deepctr-torch格式的特征数据
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple包含:
            - model_input: DeepCTR格式的模型输入字典
            - feature_columns: 特征列定义列表
            - feature_info: 特征信息字典
        """
        logger.info("Preparing features for deepctr-torch MTL models...")
        
        # 1. 识别和分类特征
        sparse_features, dense_features, clip_features = self._classify_features(df)
        
        logger.info(f"Feature breakdown: {len(sparse_features)} sparse, "
                   f"{len(dense_features)} dense, {len(clip_features)} CLIP")
        
        # 2. 处理稀疏特征
        sparse_feature_columns = self._process_sparse_features(df, sparse_features)
        
        # 3. 过滤零特征样本
        if self.filter_zeros and clip_features:
            df = self._filter_zero_features(df, clip_features)
        
        # 4. 处理CLIP特征的PCA降维
        final_clip_features = self._process_clip_features(df, clip_features)
        
        # 5. 处理密集特征
        all_dense_features = dense_features + final_clip_features
        dense_feature_columns = self._process_dense_features(df, all_dense_features)
        
        # 6. 合并特征列并准备模型输入
        feature_columns = sparse_feature_columns + dense_feature_columns
        model_input = self._prepare_model_input(df, feature_columns)
        
        # 7. 构建特征信息
        feature_info = {
            'sparse_features': [feat.name for feat in sparse_feature_columns],
            'dense_features': [feat.name for feat in dense_feature_columns],
            'total_features': len(feature_columns),
            'feature_columns': feature_columns
        }
        
        logger.info(f"Prepared deepctr-torch features: {len(feature_columns)} total (float32)")
        return model_input, feature_columns, feature_info
    
    def _classify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """分类特征列
        
        Args:
            df: 输入DataFrame
            
        Returns:
            Tuple包含 (稀疏特征列表, 密集特征列表, CLIP特征列表)
        """
        all_features = list(df.columns)
        
        # 排除目标变量和ID列
        exclude_patterns = [
            # 目标变量
            'ctr', 'like_rate', 'fav_rate', 'comment_rate', 'share_rate', 'follow_rate', 
            'interaction_rate', 'ces_rate', 'impression_log', 'sort_score2',
            # 中间计算列
            'total_interactions',
            # ID列
            'note_id', 'user_id'
        ]
        
        feature_candidates = []
        for col in all_features:
            if any(pattern in col for pattern in exclude_patterns):
                continue
            feature_candidates.append(col)
        
        # 按类型分组特征
        sparse_features = []
        dense_features = []
        clip_features = []
        
        for feat in feature_candidates:
            if any(pattern in feat for pattern in ['cover_image_feat_', 'inner_image_feat_', 'title_feat_', 
                                                   'content_feat_', 'tag_feat_', 'text_feat_', 'image_feat_']):
                clip_features.append(feat)
            elif any(pattern in feat for pattern in ['taxonomy1', 'taxonomy2', 'taxonomy3', 
                                                     'intention_lv1', 'note_marketing_integrated_level',
                                                     'tag_info', 'cover_image_ocr_text', 
                                                     'inner_images_ocr_text']):
                sparse_features.append(feat)
            elif feat == 'type':  # 处理type字段避免冲突
                df['note_type'] = df['type']
                df = df.drop(columns=['type'])
                sparse_features.append('note_type')
            elif feat == 'intention_lv2is_mcn':  # 原parquet文件字段名写入错误，在此处修正
                df['intention_lv2'] = df['intention_lv2is_mcn']
                df = df.drop(columns=['intention_lv2is_mcn'])
                sparse_features.append('intention_lv2')
            
            elif feat in ['title_length', 'content_length', 'num_images']:
                dense_features.append(feat)
        
        logger.info("="*50)
        logger.info("FEATURE CLASSIFICATION ANALYSIS")
        logger.info("="*50)
        logger.info(f"Sparse features ({len(sparse_features)}): {sparse_features[:10]}...")
        logger.info(f"Dense features ({len(dense_features)}): {dense_features[:10]}...")
        logger.info(f"CLIP features ({len(clip_features)}): {clip_features[:5]}...")
        logger.info("="*50)
        
        return sparse_features, dense_features, clip_features
    
    def _process_sparse_features(self, df: pd.DataFrame, sparse_features: List[str]) -> List[SparseFeat]:
        """处理稀疏特征
        
        Args:
            df: DataFrame
            sparse_features: 稀疏特征列表
            
        Returns:
            SparseFeat对象列表
        """
        sparse_feature_columns = []
        
        for feat in sparse_features:
            if feat in df.columns:
                # 转换为字符串并填充缺失值
                df[feat] = df[feat].astype(str).fillna('unknown')
                
                # 标签编码
                le = LabelEncoder()
                df[feat] = le.fit_transform(df[feat])
                self.label_encoders[feat] = le
                
                # 计算embedding维度 - 对于PNN兼容，所有稀疏特征使用统一维度
                vocab_size = len(le.classes_)
                embedding_dim = 8  # 固定embedding维度，确保PNN兼容性
                
                sparse_feature_columns.append(
                    SparseFeat(feat, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
                )
        
        logger.info(f"Processed {len(sparse_feature_columns)} sparse features with LabelEncoder")
        return sparse_feature_columns
    
    def _filter_zero_features(self, df: pd.DataFrame, clip_features: List[str]) -> pd.DataFrame:
        """过滤全零CLIP特征的样本
        
        Args:
            df: DataFrame
            clip_features: CLIP特征列表
            
        Returns:
            过滤后的DataFrame
        """
        if not clip_features:
            return df
        
        logger.info("Filtering samples with all-zero CLIP features...")
        original_size = len(df)
        
        clip_df = df[clip_features]
        non_zero_mask = (clip_df != 0).any(axis=1)
        df = df[non_zero_mask]
        
        logger.info(f"After filtering zero features: {len(df)}/{original_size} rows retained")
        return df
    
    def _process_clip_features(self, df: pd.DataFrame, clip_features: List[str]) -> List[str]:
        """处理CLIP特征（包括可选的PCA降维）
        
        Args:
            df: DataFrame
            clip_features: CLIP特征列表
            
        Returns:
            处理后的CLIP特征列表
        """
        if not clip_features:
            return []
        
        final_clip_features = clip_features
        
        # PCA降维
        if self.use_pca:
            logger.info(f"Applying PCA to {len(clip_features)} CLIP features")
            clip_matrix = df[clip_features].values
            
            pca = PCA(n_components=min(self.pca_components, clip_matrix.shape[1]))
            clip_pca = pca.fit_transform(clip_matrix)
            
            # 创建PCA特征列
            final_clip_features = []
            for i in range(clip_pca.shape[1]):
                pca_feat_name = f'clip_pca_{i}'
                df[pca_feat_name] = clip_pca[:, i].astype(np.float32)
                final_clip_features.append(pca_feat_name)
            
            # 删除原始CLIP特征
            df = df.drop(columns=clip_features)
            
            self.pca_transformers['clip'] = pca
            logger.info(f"PCA reduced CLIP features from {len(clip_features)} to {len(final_clip_features)}")
        
        return final_clip_features
    
    def _process_dense_features(self, df: pd.DataFrame, dense_features: List[str]) -> List[DenseFeat]:
        """处理密集特征
        
        Args:
            df: DataFrame
            dense_features: 密集特征列表
            
        Returns:
            DenseFeat对象列表
        """
        if not dense_features:
            return []
        
        # 确保数值类型并填充缺失值
        for feat in dense_features:
            if feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors='coerce').fillna(0).astype(np.float32)
        
        # 标准化密集特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[dense_features])
        df[dense_features] = scaled_features.astype(np.float32)
        self.scalers['dense'] = scaler
        
        logger.info(f"StandardScaler scaled {len(dense_features)} dense features (float32)")
        logger.info(f"Scaled features stats - Mean: {scaled_features.mean():.6f}, Std: {scaled_features.std():.6f}")
        logger.info(f"Scaled features range - Min: {scaled_features.min():.6f}, Max: {scaled_features.max():.6f}")
        
        # 创建DenseFeat对象
        dense_feature_columns = []
        for feat in dense_features:
            dense_feature_columns.append(DenseFeat(feat, 1))
        
        return dense_feature_columns
    
    def _prepare_model_input(self, df: pd.DataFrame, feature_columns: List) -> Dict[str, np.ndarray]:
        """准备DeepCTR模型输入格式
        
        Args:
            df: DataFrame
            feature_columns: 特征列定义
            
        Returns:
            模型输入字典
        """
        model_input = {}
        
        for feat in feature_columns:
            if feat.name in df.columns:
                # 转换为float32以兼容MPS
                model_input[feat.name] = df[feat.name].values.astype(np.float32)
            else:
                logger.warning(f"Feature {feat.name} not found in dataframe")
        
        return model_input
    
    def prepare_targets(self, df: pd.DataFrame, task_columns: List[str]) -> Dict[str, np.ndarray]:
        """准备目标变量
        
        Args:
            df: DataFrame
            task_columns: 任务列名列表
            
        Returns:
            目标变量字典
        """
        task_targets = {}
        
        for target_col in task_columns:
            if target_col in df.columns:
                # 强制转换为float32以兼容MPS
                targets = df[target_col].fillna(0).astype(np.float32)
                task_targets[target_col] = targets.values
                logger.info(f"Prepared target {target_col}: {len(targets)} samples, "
                           f"range [{targets.min():.4f}, {targets.max():.4f}] (float32)")
            else:
                logger.warning(f"Target column {target_col} not found")
        
        return task_targets
    
    def get_preprocessors(self) -> Dict[str, Any]:
        """获取所有预处理器（增强版）
        
        Returns:
            预处理器字典，包含完整的特征处理信息
        """
        return {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'pca_transformers': self.pca_transformers,
            'feature_settings': {
                'filter_zeros': self.filter_zeros,
                'use_pca': self.use_pca,
                'pca_components': self.pca_components
            }
        }
    
    def validate_features(self, df: pd.DataFrame, feature_columns: List) -> Dict[str, Any]:
        """验证特征质量
        
        Args:
            df: DataFrame
            feature_columns: 特征列定义
            
        Returns:
            特征质量报告
        """
        report = {
            'total_features': len(feature_columns),
            'sparse_features': 0,
            'dense_features': 0,
            'missing_features': [],
            'constant_features': []
        }
        
        for feat in feature_columns:
            # 统计特征类型
            if isinstance(feat, SparseFeat):
                report['sparse_features'] += 1
            elif isinstance(feat, DenseFeat):
                report['dense_features'] += 1
            
            # 检查缺失特征
            if feat.name not in df.columns:
                report['missing_features'].append(feat.name)
            else:
                # 检查常数特征
                data = df[feat.name]
                if data.nunique() <= 1:
                    report['constant_features'].append(feat.name)
        
        return report