#!/usr/bin/env python3
"""
CLIP特征评估脚本 - 统一版本
评估CLIP图像和文本特征提取的效果

评估三个层面：
1. 视觉理解层：模型是否"看懂"了图像的关联性
2. 表征质量层：特征向量本身的质量
3. 任务适配层：特征对CTR预测的有用性

内置问题分析：自动分析评估结果并提供改进方案

使用方法：
    python evaluate_clip_features.py --input /path/to/parquet/dir --output evaluation_report.html
    python evaluate_clip_features.py --input /path/to/parquet/dir --analyze-only  # 仅分析现有结果
"""

import os
import sys
import argparse
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CLIPFeatureEvaluator:
    """CLIP特征评估器 - 集成评估和问题分析功能"""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_data(self, data_path: str, max_samples: Optional[int] = None, 
                  min_impression_threshold: int = 5000, filter_zeros: bool = True) -> pd.DataFrame:
        """加载数据并应用过滤"""
        logger.info(f"Loading data from {data_path}")
        
        # 支持目录或单个文件
        path = Path(data_path)
        if path.is_dir():
            # 读取目录下所有parquet文件
            parquet_files = list(path.glob("**/*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {data_path}")
            
            dfs = []
            # 随机取5个文件
            parquet_files = random.sample(parquet_files, min(10, len(parquet_files))) # 限制读取文件数量
            for file_path in parquet_files:
            #for file_path in parquet_files[-5:]:  # 限制读取文件数量
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                    logger.info(f"  Loaded {len(df)} rows from {file_path.name}")
                except Exception as e:
                    logger.warning(f"  Failed to load {file_path}: {e}")
            
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(data_path)
        
        original_size = len(df)
        logger.info(f"Loaded {original_size} total samples before filtering")
        
        # 提取特征列
        self.image_feat_cols = [col for col in df.columns if col.startswith('image_feat_')]
        self.text_feat_cols = [col for col in df.columns if col.startswith('text_feat_')]
        
        logger.info(f"Found {len(self.image_feat_cols)} image feature dimensions")
        logger.info(f"Found {len(self.text_feat_cols)} text feature dimensions")
        
        # 应用过滤
        filtered_by_impression = 0
        filtered_by_zeros = 0
        
        # 1. 过滤低曝光量的行
        if 'imp_num' in df.columns and min_impression_threshold > 0:
            mask_impression = df['imp_num'] >= min_impression_threshold
            filtered_by_impression = (~mask_impression).sum()
            df = df[mask_impression]
            logger.info(f"Filtered {filtered_by_impression} rows with imp_num < {min_impression_threshold}")
        
        # 2. 过滤全零特征的行（可选）
        if filter_zeros and self.image_feat_cols:
            # 检查图像特征是否全零
            image_features = df[self.image_feat_cols].values
            non_zero_mask = np.any(image_features != 0, axis=1)
            
            # 如果有文本特征，也检查文本特征
            if self.text_feat_cols:
                text_features = df[self.text_feat_cols].values
                text_non_zero_mask = np.any(text_features != 0, axis=1)
                # 保留至少有一个非零特征的行
                non_zero_mask = non_zero_mask | text_non_zero_mask
            
            filtered_by_zeros = (~non_zero_mask).sum()
            df = df[non_zero_mask]
            logger.info(f"Filtered {filtered_by_zeros} rows with all-zero CLIP features")
        
        # 统计过滤结果
        final_size = len(df)
        total_filtered = original_size - final_size
        filter_percentage = (total_filtered / original_size * 100) if original_size > 0 else 0
        
        logger.info("=" * 60)
        logger.info("Filtering Statistics:")
        logger.info(f"  Original samples: {original_size:,}")
        logger.info(f"  Filtered by impression threshold: {filtered_by_impression:,}")
        logger.info(f"  Filtered by zero features: {filtered_by_zeros:,}")
        logger.info(f"  Total filtered: {total_filtered:,} ({filter_percentage:.1f}%)")
        logger.info(f"  Final samples: {final_size:,}")
        logger.info("=" * 60)
        
        # 保存过滤统计信息
        self.filter_stats = {
            'original_size': original_size,
            'filtered_by_impression': filtered_by_impression,
            'filtered_by_zeros': filtered_by_zeros,
            'total_filtered': total_filtered,
            'filter_percentage': filter_percentage,
            'final_size': final_size,
            'min_impression_threshold': min_impression_threshold,
            'filter_zeros': filter_zeros
        }
        
        # 如果过滤后样本太少，发出警告
        if final_size < 100:
            logger.warning(f"WARNING: Only {final_size} samples remain after filtering!")
            logger.warning("This may affect evaluation reliability.")
        
        # 限制样本数量（在过滤之后）
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            logger.info(f"Sampled {max_samples} rows from filtered {len(df)} samples")
        
        # 计算CTR（如果需要）
        if 'ctr' not in df.columns and all(col in df.columns for col in ['imp_num', 'click_num']):
            df['ctr'] = df.apply(
                lambda row: row['click_num'] / row['imp_num'] if row['imp_num'] > 0 else 0,
                axis=1
            )
            logger.info("Calculated CTR from imp_num and click_num")
        
        # 如果有imp_num，打印曝光量分布统计
        if 'imp_num' in df.columns:
            imp_stats = df['imp_num'].describe()
            logger.info("\nImpression Distribution (after filtering):")
            logger.info(f"  Mean: {imp_stats['mean']:,.0f}")
            logger.info(f"  Median: {imp_stats['50%']:,.0f}")
            logger.info(f"  Min: {imp_stats['min']:,.0f}")
            logger.info(f"  Max: {imp_stats['max']:,.0f}")
        
        return df
    
    def evaluate_visual_understanding(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估视觉理解层"""
        logger.info("=" * 60)
        logger.info("Evaluating Visual Understanding Layer")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. 聚类质量评估
        logger.info("\n1. Clustering Quality Evaluation")
        cluster_results = self._evaluate_clustering(df)
        results['clustering'] = cluster_results
        
        # 2. 图文语义一致性评估
        logger.info("\n2. Image-Text Semantic Consistency Evaluation")
        consistency_results = self._evaluate_semantic_consistency(df)
        results['semantic_consistency'] = consistency_results
        
        return results
    
    def _evaluate_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估聚类质量"""
        # 提取图像特征
        image_features = df[self.image_feat_cols].values
        
        # 过滤全零向量
        non_zero_mask = np.any(image_features != 0, axis=1)
        image_features_valid = image_features[non_zero_mask]
        
        if len(image_features_valid) < 100:
            logger.warning(f"Not enough valid samples for clustering: {len(image_features_valid)}")
            return {}
        
        # 检查特征多样性
        feature_std = np.std(image_features_valid, axis=0)
        effective_dims = np.sum(feature_std > 1e-6)
        logger.info(f"  Effective feature dimensions: {effective_dims}/{image_features_valid.shape[1]}")
        
        if effective_dims < 10:
            logger.warning("Very few effective dimensions, clustering may not be meaningful")
        
        # 标准化特征
        scaler = StandardScaler()
        image_features_scaled = scaler.fit_transform(image_features_valid)
        
        results = {}
        
        # 测试不同的聚类数
        k_values = [10, 20, 30, 50]
        valid_results = []  # 存储有效的聚类结果
        
        for k in k_values:
            if k > len(image_features_valid) // 2:
                continue
                
            logger.info(f"  Testing K={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(image_features_scaled)
            
            # 检查聚类结果
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logger.warning(f"    K={k}: All samples assigned to single cluster, skipping")
                continue
                
            try:
                # 计算聚类指标
                sil_score = silhouette_score(image_features_scaled, labels)
                db_score = davies_bouldin_score(image_features_scaled, labels)
                
                valid_results.append({
                    'k': k,
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score,
                    'labels': labels
                })
                
                logger.info(f"    Silhouette Score: {sil_score:.4f}")
                logger.info(f"    Davies-Bouldin Score: {db_score:.4f}")
                
            except Exception as e:
                logger.warning(f"    K={k}: Failed to compute clustering metrics: {e}")
                continue
        
        # 检查是否有有效的聚类结果
        if not valid_results:
            logger.warning("No valid clustering results obtained. Skipping clustering analysis.")
            return {
                'error': 'No valid clustering results',
                'reason': 'All samples may be too similar or data quality issues'
            }
        
        # 选择最佳K值
        best_result = max(valid_results, key=lambda x: x['silhouette_score'])
        best_k = best_result['k']
        final_labels = best_result['labels']
        
        logger.info(f"\n  Best K={best_k} with Silhouette Score={best_result['silhouette_score']:.4f}")
        
        # 如果有类别信息，计算聚类纯度
        category_cols = ['taxonomy1', 'taxonomy2', 'intention_lv1']
        available_cats = [col for col in category_cols if col in df.columns]
        
        if available_cats:
            # 只对有效样本计算纯度
            df_valid = df[non_zero_mask].copy()
            df_valid['cluster'] = final_labels
            
            purity_scores = {}
            for cat_col in available_cats:
                if cat_col in df_valid.columns and df_valid[cat_col].notna().sum() > 0:
                    # 计算每个聚类的纯度
                    cluster_purities = []
                    for cluster_id in range(best_k):
                        cluster_data = df_valid[df_valid['cluster'] == cluster_id]
                        if len(cluster_data) > 0:
                            # 找出该聚类中最常见的类别
                            most_common = cluster_data[cat_col].value_counts().iloc[0]
                            purity = most_common / len(cluster_data)
                            cluster_purities.append(purity)
                    
                    avg_purity = np.mean(cluster_purities) if cluster_purities else 0
                    purity_scores[cat_col] = avg_purity
                    logger.info(f"  Clustering Purity for {cat_col}: {avg_purity:.4f}")
            
            results['category_purity'] = purity_scores
        
        # 构建结果
        results.update({
            'best_k': best_k,
            'silhouette_score': best_result['silhouette_score'],
            'davies_bouldin_score': best_result['davies_bouldin_score'],
            'k_values': [r['k'] for r in valid_results],
            'silhouette_scores': [r['silhouette_score'] for r in valid_results],
            'db_scores': [r['davies_bouldin_score'] for r in valid_results]
        })
        
        return results
    
    def _evaluate_semantic_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估图文语义一致性"""
        # 提取特征
        image_features = df[self.image_feat_cols].values
        text_features = df[self.text_feat_cols].values
        
        # 计算同一note的图文相似度（正样本）
        positive_similarities = []
        for i in range(len(df)):
            img_feat = image_features[i]
            txt_feat = text_features[i]
            
            # 跳过全零向量
            if np.all(img_feat == 0) or np.all(txt_feat == 0):
                continue
            
            # 计算余弦相似度
            sim = 1 - cosine(img_feat, txt_feat)
            positive_similarities.append(sim)
        
        # 随机配对计算负样本相似度
        negative_similarities = []
        n_negative = min(1000, len(df))
        
        for _ in range(n_negative):
            i = np.random.randint(0, len(df))
            j = np.random.randint(0, len(df))
            
            if i == j:
                continue
            
            img_feat = image_features[i]
            txt_feat = text_features[j]
            
            if np.all(img_feat == 0) or np.all(txt_feat == 0):
                continue
            
            sim = 1 - cosine(img_feat, txt_feat)
            negative_similarities.append(sim)
        
        # 统计分析
        pos_mean = np.mean(positive_similarities) if positive_similarities else 0
        pos_std = np.std(positive_similarities) if positive_similarities else 0
        neg_mean = np.mean(negative_similarities) if negative_similarities else 0
        neg_std = np.std(negative_similarities) if negative_similarities else 0
        
        logger.info(f"  Positive (same note) similarity: {pos_mean:.4f} ± {pos_std:.4f}")
        logger.info(f"  Negative (random) similarity: {neg_mean:.4f} ± {neg_std:.4f}")
        
        # T检验
        if positive_similarities and negative_similarities:
            t_stat, p_value = stats.ttest_ind(positive_similarities, negative_similarities)
            logger.info(f"  T-test: t={t_stat:.4f}, p={p_value:.6f}")
            
            # 效应大小 (Cohen's d)
            pooled_std = np.sqrt((np.var(positive_similarities) + np.var(negative_similarities)) / 2)
            cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
            logger.info(f"  Cohen's d (effect size): {cohens_d:.4f}")
        else:
            t_stat, p_value, cohens_d = 0, 1, 0
        
        return {
            'positive_similarity_mean': pos_mean,
            'positive_similarity_std': pos_std,
            'negative_similarity_mean': neg_mean,
            'negative_similarity_std': neg_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'positive_samples': len(positive_similarities),
            'negative_samples': len(negative_similarities)
        }
    
    def evaluate_feature_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估表征质量层"""
        logger.info("=" * 60)
        logger.info("Evaluating Feature Quality Layer")
        logger.info("=" * 60)
        
        results = {}
        
        # 1. 方差分析
        logger.info("\n1. Variance Analysis")
        variance_results = self._analyze_variance(df)
        results['variance'] = variance_results
        
        # 2. 维度相关性分析
        logger.info("\n2. Dimension Correlation Analysis")
        correlation_results = self._analyze_dimension_correlation(df)
        results['correlation'] = correlation_results
        
        # 3. PCA分析
        logger.info("\n3. PCA Analysis")
        pca_results = self._analyze_pca(df)
        results['pca'] = pca_results
        
        return results
    
    def _analyze_variance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析特征方差"""
        results = {}
        
        for feat_type, feat_cols in [('image', self.image_feat_cols), ('text', self.text_feat_cols)]:
            features = df[feat_cols].values
            
            # 计算每个维度的方差
            variances = np.var(features, axis=0)
            
            # 统计不同方差级别的维度数
            zero_var = np.sum(variances < 1e-6)
            low_var = np.sum((variances >= 1e-6) & (variances < 0.01))
            med_var = np.sum((variances >= 0.01) & (variances < 0.1))
            high_var = np.sum(variances >= 0.1)
            
            total_dims = len(variances)
            
            logger.info(f"  {feat_type.capitalize()} features:")
            logger.info(f"    Zero variance dims: {zero_var}/{total_dims} ({zero_var/total_dims*100:.1f}%)")
            logger.info(f"    Low variance dims: {low_var}/{total_dims} ({low_var/total_dims*100:.1f}%)")
            logger.info(f"    Medium variance dims: {med_var}/{total_dims} ({med_var/total_dims*100:.1f}%)")
            logger.info(f"    High variance dims: {high_var}/{total_dims} ({high_var/total_dims*100:.1f}%)")
            
            results[feat_type] = {
                'variances': variances.tolist(),
                'zero_var_dims': zero_var,
                'low_var_dims': low_var,
                'med_var_dims': med_var,
                'high_var_dims': high_var,
                'total_dims': total_dims,
                'mean_variance': float(np.mean(variances)),
                'median_variance': float(np.median(variances))
            }
        
        return results
    
    def _analyze_dimension_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析维度间相关性"""
        results = {}
        
        for feat_type, feat_cols in [('image', self.image_feat_cols), ('text', self.text_feat_cols)]:
            features = df[feat_cols].values
            
            # 计算相关性矩阵（采样以加速）
            n_samples = min(1000, len(features))
            sampled_features = features[np.random.choice(len(features), n_samples, replace=False)]
            
            corr_matrix = np.corrcoef(sampled_features.T)
            
            # 获取上三角矩阵（排除对角线）
            upper_triangle = np.triu(corr_matrix, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            # 统计高相关性的维度对
            high_corr_threshold = 0.9
            high_corr_pairs = np.sum(np.abs(correlations) > high_corr_threshold)
            total_pairs = len(correlations)
            
            logger.info(f"  {feat_type.capitalize()} features:")
            logger.info(f"    Mean correlation: {np.mean(np.abs(correlations)):.4f}")
            logger.info(f"    Max correlation: {np.max(np.abs(correlations)):.4f}")
            logger.info(f"    High correlation pairs (>0.9): {high_corr_pairs}/{total_pairs} ({high_corr_pairs/total_pairs*100:.2f}%)")
            
            # 找出最相关的维度对
            high_corr_indices = np.where(np.abs(upper_triangle) > high_corr_threshold)
            top_corr_pairs = []
            for i, j in zip(high_corr_indices[0][:10], high_corr_indices[1][:10]):
                top_corr_pairs.append({
                    'dim1': i,
                    'dim2': j,
                    'correlation': float(corr_matrix[i, j])
                })
            
            results[feat_type] = {
                'mean_correlation': float(np.mean(np.abs(correlations))),
                'max_correlation': float(np.max(np.abs(correlations))),
                'high_corr_pairs': high_corr_pairs,
                'total_pairs': total_pairs,
                'top_corr_pairs': top_corr_pairs
            }
        
        return results
    
    def _analyze_pca(self, df: pd.DataFrame) -> Dict[str, Any]:
        """PCA分析"""
        results = {}
        
        for feat_type, feat_cols in [('image', self.image_feat_cols), ('text', self.text_feat_cols)]:
            features = df[feat_cols].values
            
            # 过滤全零向量
            non_zero_mask = np.any(features != 0, axis=1)
            features_valid = features[non_zero_mask]
            
            if len(features_valid) < 10:
                continue
            
            # PCA分析
            pca = PCA()
            pca.fit(features_valid)
            
            # 计算保留不同方差所需的维度数
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            dims_95 = np.argmax(cumsum_var >= 0.95) + 1
            dims_99 = np.argmax(cumsum_var >= 0.99) + 1
            dims_999 = np.argmax(cumsum_var >= 0.999) + 1
            
            logger.info(f"  {feat_type.capitalize()} features:")
            logger.info(f"    Dims for 95% variance: {dims_95}/{len(feat_cols)} ({dims_95/len(feat_cols)*100:.1f}%)")
            logger.info(f"    Dims for 99% variance: {dims_99}/{len(feat_cols)} ({dims_99/len(feat_cols)*100:.1f}%)")
            logger.info(f"    Dims for 99.9% variance: {dims_999}/{len(feat_cols)} ({dims_999/len(feat_cols)*100:.1f}%)")
            
            results[feat_type] = {
                'dims_95': dims_95,
                'dims_99': dims_99,
                'dims_999': dims_999,
                'total_dims': len(feat_cols),
                'explained_variance_ratio': pca.explained_variance_ratio_[:50].tolist(),  # 前50个主成分
                'cumulative_variance': cumsum_var[:50].tolist()
            }
        
        return results
    
    def evaluate_task_adaptation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估任务适配层"""
        logger.info("=" * 60)
        logger.info("Evaluating Task Adaptation Layer")
        logger.info("=" * 60)
        
        # 检查是否有CTR数据
        if 'ctr' not in df.columns:
            logger.warning("No CTR data available, skipping task adaptation evaluation")
            return {}
        
        # 过滤有效的CTR数据
        valid_mask = (df['ctr'] >= 0) & (df['ctr'] <= 1) & df['ctr'].notna()
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) < 100:
            logger.warning(f"Not enough valid CTR data: {len(df_valid)}")
            return {}
        
        results = {}
        
        # 1. 相关性分析
        logger.info("\n1. Feature-CTR Correlation Analysis")
        correlation_results = self._analyze_ctr_correlation(df_valid)
        results['correlation'] = correlation_results
        
        # 2. 预测能力评估
        logger.info("\n2. CTR Prediction Evaluation")
        prediction_results = self._evaluate_ctr_prediction(df_valid)
        results['prediction'] = prediction_results
        
        # 3. 特征重要性分析
        logger.info("\n3. Feature Importance Analysis")
        importance_results = self._analyze_feature_importance(df_valid)
        results['importance'] = importance_results
        
        # 4. 消融实验
        logger.info("\n4. Ablation Study")
        ablation_results = self._ablation_study(df_valid)
        results['ablation'] = ablation_results
        
        return results
    
    def _analyze_ctr_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析特征与CTR的相关性"""
        results = {}
        ctr_values = df['ctr'].values
        
        for feat_type, feat_cols in [('image', self.image_feat_cols), ('text', self.text_feat_cols)]:
            features = df[feat_cols].values
            
            # 计算每个维度与CTR的相关性
            correlations = []
            for i in range(features.shape[1]):
                corr, _ = stats.pearsonr(features[:, i], ctr_values)
                correlations.append(corr)
            
            correlations = np.array(correlations)
            
            # 统计
            max_corr = np.max(np.abs(correlations))
            mean_corr = np.mean(np.abs(correlations))
            significant_dims = np.sum(np.abs(correlations) > 0.1)
            
            logger.info(f"  {feat_type.capitalize()} features:")
            logger.info(f"    Max correlation: {max_corr:.4f}")
            logger.info(f"    Mean correlation: {mean_corr:.4f}")
            logger.info(f"    Significant dims (|r|>0.1): {significant_dims}/{len(correlations)}")
            
            # 找出最相关的维度
            top_dims_idx = np.argsort(np.abs(correlations))[-10:][::-1]
            top_dims = [(idx, correlations[idx]) for idx in top_dims_idx]
            
            results[feat_type] = {
                'max_correlation': float(max_corr),
                'mean_correlation': float(mean_corr),
                'significant_dims': int(significant_dims),
                'top_dimensions': top_dims
            }
        
        return results
    
    def _evaluate_ctr_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估CTR预测能力"""
        results = {}
        
        # 准备数据
        image_features = df[self.image_feat_cols].values
        text_features = df[self.text_feat_cols].values
        all_features = np.concatenate([image_features, text_features], axis=1)
        
        # CTR目标
        y = df['ctr'].values
        
        # 转换为二分类问题（高低CTR）
        median_ctr = np.median(y)
        y_binary = (y > median_ctr).astype(int)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, y_binary, test_size=0.2, random_state=42
        )
        
        # 逻辑回归模型
        lr_model = LogisticRegression(max_iter=100, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred_proba)
        
        logger.info(f"  Logistic Regression AUC: {lr_auc:.4f}")
        
        # 回归任务
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            all_features, y, test_size=0.2, random_state=42
        )
        
        # 随机森林回归
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_model.fit(X_train_reg, y_train_reg)
        rf_pred = rf_model.predict(X_test_reg)
        
        mse = mean_squared_error(y_test_reg, rf_pred)
        r2 = r2_score(y_test_reg, rf_pred)
        
        logger.info(f"  Random Forest MSE: {mse:.6f}")
        logger.info(f"  Random Forest R²: {r2:.4f}")
        
        # 计算相关性
        pred_corr, _ = stats.spearmanr(y_test_reg, rf_pred)
        logger.info(f"  Spearman correlation: {pred_corr:.4f}")
        
        results = {
            'logistic_auc': float(lr_auc),
            'rf_mse': float(mse),
            'rf_r2': float(r2),
            'spearman_correlation': float(pred_corr),
            'median_ctr': float(median_ctr)
        }
        
        return results
    
    def _analyze_feature_importance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析特征重要性"""
        # 使用随机森林获取特征重要性
        image_features = df[self.image_feat_cols].values
        text_features = df[self.text_feat_cols].values
        all_features = np.concatenate([image_features, text_features], axis=1)
        y = df['ctr'].values
        
        # 训练模型
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf_model.fit(all_features, y)
        
        # 获取特征重要性
        importances = rf_model.feature_importances_
        
        # 分离图像和文本特征的重要性
        n_image = len(self.image_feat_cols)
        image_importances = importances[:n_image]
        text_importances = importances[n_image:]
        
        # 统计
        results = {
            'image': {
                'mean_importance': float(np.mean(image_importances)),
                'max_importance': float(np.max(image_importances)),
                'top_features': np.argsort(image_importances)[-10:][::-1].tolist()
            },
            'text': {
                'mean_importance': float(np.mean(text_importances)),
                'max_importance': float(np.max(text_importances)),
                'top_features': np.argsort(text_importances)[-10:][::-1].tolist()
            }
        }
        
        logger.info(f"  Image features mean importance: {results['image']['mean_importance']:.6f}")
        logger.info(f"  Text features mean importance: {results['text']['mean_importance']:.6f}")
        
        return results
    
    def _ablation_study(self, df: pd.DataFrame) -> Dict[str, Any]:
        """消融实验：评估image_features的贡献"""
        logger.info("Running ablation study to evaluate image_features contribution...")
        
        from sklearn.preprocessing import LabelEncoder
        
        # 准备目标变量
        y = df['ctr'].values
        
        # 准备其他特征（非CLIP特征）
        other_features = []
        feature_names = []
        
        # 处理类别特征
        categorical_cols = ['type', 'taxonomy1', 'taxonomy2', 'taxonomy3', 'intention_lv1']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # 处理缺失值
                col_values = df[col].fillna('unknown').astype(str)
                encoded = le.fit_transform(col_values)
                other_features.append(encoded.reshape(-1, 1))
                feature_names.append(col)
        
        # 处理数值特征
        numerical_cols = ['question_marks', 'exclamation_marks']
        for col in numerical_cols:
            if col in df.columns:
                values = df[col].fillna(0).values.reshape(-1, 1)
                other_features.append(values)
                feature_names.append(col)
        
        # OCR文本特征（简单处理：是否有OCR文本）
        if 'ocr_text' in df.columns:
            has_ocr = (df['ocr_text'].notna() & (df['ocr_text'] != '')).astype(int).values.reshape(-1, 1)
            other_features.append(has_ocr)
            feature_names.append('has_ocr')
        
        # 合并其他特征
        if other_features:
            X_other = np.concatenate(other_features, axis=1)
        else:
            X_other = np.zeros((len(df), 1))  # 如果没有其他特征
        
        # CLIP特征
        X_image = df[self.image_feat_cols].values if self.image_feat_cols else np.zeros((len(df), 1))
        X_text = df[self.text_feat_cols].values if self.text_feat_cols else np.zeros((len(df), 1))
        
        # 构建不同的特征组合
        feature_sets = {
            'baseline_all': np.concatenate([X_other, X_image, X_text], axis=1),
            'no_image': np.concatenate([X_other, X_text], axis=1),
            'no_text': np.concatenate([X_other, X_image], axis=1),
            'no_clip': X_other,
            'only_image': X_image,
            'only_text': X_text,
            'only_clip': np.concatenate([X_image, X_text], axis=1)
        }
        
        # 对每种特征组合训练模型
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        results = {}
        
        for name, X in feature_sets.items():
            # 跳过空特征集
            if X.shape[1] == 0:
                continue
                
            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练模型
            model = RandomForestRegressor(
                n_estimators=50, 
                max_depth=5, 
                random_state=42,
                n_jobs=-1
            )
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # 在测试集上评估
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Spearman相关性
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(y_test, y_pred)
            
            results[name] = {
                'mse': float(mse),
                'r2': float(r2),
                'spearman': float(spearman_corr),
                'cv_r2_mean': float(np.mean(cv_scores)),
                'cv_r2_std': float(np.std(cv_scores)),
                'n_features': X.shape[1]
            }
            
            logger.info(f"  {name}: R²={r2:.4f}, MSE={mse:.6f}, Spearman={spearman_corr:.4f}")
        
        # 计算特征贡献
        contributions = {}
        
        if 'baseline_all' in results and 'no_image' in results:
            r2_diff = results['baseline_all']['r2'] - results['no_image']['r2']
            contributions['image_features'] = {
                'r2_contribution': float(r2_diff),
                'relative_contribution': float(r2_diff / max(results['baseline_all']['r2'], 0.001) * 100),
                'is_positive': r2_diff > 0,
                'significance': 'significant' if abs(r2_diff) > 0.01 else 'negligible'
            }
            logger.info(f"\n  Image features contribution: ΔR²={r2_diff:.4f} ({contributions['image_features']['relative_contribution']:.1f}%)")
        
        if 'baseline_all' in results and 'no_text' in results:
            r2_diff = results['baseline_all']['r2'] - results['no_text']['r2']
            contributions['text_features'] = {
                'r2_contribution': float(r2_diff),
                'relative_contribution': float(r2_diff / max(results['baseline_all']['r2'], 0.001) * 100),
                'is_positive': r2_diff > 0,
                'significance': 'significant' if abs(r2_diff) > 0.01 else 'negligible'
            }
            logger.info(f"  Text features contribution: ΔR²={r2_diff:.4f} ({contributions['text_features']['relative_contribution']:.1f}%)")
        
        # 对比CLIP vs 非CLIP特征
        if 'only_clip' in results and 'no_clip' in results:
            clip_r2 = results['only_clip']['r2']
            other_r2 = results['no_clip']['r2']
            logger.info(f"\n  CLIP features alone: R²={clip_r2:.4f}")
            logger.info(f"  Other features alone: R²={other_r2:.4f}")
            
            if clip_r2 > other_r2:
                logger.info("  → CLIP features are more predictive than other features")
            else:
                logger.info("  → Other features are more predictive than CLIP features")
        
        return {
            'feature_sets_performance': results,
            'contributions': contributions,
            'recommendation': self._generate_ablation_recommendation(results, contributions)
        }
    
    def _generate_ablation_recommendation(self, results: Dict, contributions: Dict) -> str:
        """基于消融实验生成建议"""
        recommendations = []
        
        # 检查image_features的贡献
        if 'image_features' in contributions:
            img_contrib = contributions['image_features']
            if img_contrib['is_positive'] and img_contrib['significance'] == 'significant':
                recommendations.append("✅ KEEP image_features: Positive contribution to CTR prediction")
            elif not img_contrib['is_positive']:
                recommendations.append("❌ REMOVE image_features: Negative impact on CTR prediction")
            else:
                recommendations.append("⚠️ OPTIONAL image_features: Negligible contribution")
        
        # 检查text_features的贡献
        if 'text_features' in contributions:
            text_contrib = contributions['text_features']
            if text_contrib['is_positive'] and text_contrib['significance'] == 'significant':
                recommendations.append("✅ KEEP text_features: Positive contribution to CTR prediction")
            elif not text_contrib['is_positive']:
                recommendations.append("❌ REMOVE text_features: Negative impact on CTR prediction")
            else:
                recommendations.append("⚠️ OPTIONAL text_features: Negligible contribution")
        
        # 对比分析
        if 'only_clip' in results and 'no_clip' in results:
            clip_r2 = results['only_clip']['r2']
            other_r2 = results['no_clip']['r2']
            
            if clip_r2 < 0.01 and other_r2 > 0.02:
                recommendations.append("💡 SUGGESTION: Focus on improving non-CLIP features")
            elif clip_r2 > other_r2 * 2:
                recommendations.append("💡 SUGGESTION: CLIP features are valuable, consider fine-tuning")
        
        return " | ".join(recommendations) if recommendations else "No clear recommendation"
    
    def analyze_evaluation_results(self, results: Dict) -> str:
        """分析评测结果并生成问题诊断报告"""
        logger.info("=" * 80)
        logger.info("Analyzing evaluation results and generating problem diagnosis...")
        logger.info("=" * 80)
        
        # 创建问题分析报告
        analysis_lines = []
        analysis_lines.append("# CLIP特征问题分析报告")
        analysis_lines.append("基于评测结果分析存在的问题并提供改进方案")
        analysis_lines.append("")
        
        # 1. 视觉理解层分析
        analysis_lines.append("## 1. 视觉理解层问题")
        analysis_lines.append("-" * 40)
        
        if 'visual_understanding' in results:
            vu = results['visual_understanding']
            
            # 聚类质量
            if 'clustering' in vu:
                cluster = vu['clustering']
                sil_score = cluster.get('silhouette_score', 0)
                
                if sil_score < 0.6:
                    analysis_lines.append(f"❌ 问题: 聚类质量中等 (Silhouette Score = {sil_score:.4f})")
                    analysis_lines.append("   原因: 特征缺乏明显的类别区分性")
                    analysis_lines.append("   影响: 相似内容无法很好地聚集在一起")
                elif sil_score > 0.9:
                    analysis_lines.append(f"⚠️ 异常: 聚类质量过高 (Silhouette Score = {sil_score:.4f})")
                    analysis_lines.append("   可能原因: K值过大或数据存在明显的聚集模式")
                    analysis_lines.append("   建议: 检查数据质量和聚类参数设置")
                else:
                    analysis_lines.append(f"✅ 正常: 聚类质量良好 (Silhouette Score = {sil_score:.4f})")
                
                # 类别纯度
                if 'category_purity' in cluster:
                    for cat, purity in cluster['category_purity'].items():
                        if purity < 0.5:
                            analysis_lines.append(f"❌ 问题: {cat}类别纯度低 ({purity:.4f})")
                            analysis_lines.append(f"   原因: CLIP未能捕捉{cat}的视觉特征")
            
            # 图文对齐
            if 'semantic_consistency' in vu:
                sem = vu['semantic_consistency']
                cohens_d = sem.get('cohens_d', 0)
                pos_sim = sem.get('positive_similarity_mean', 0)
                neg_sim = sem.get('negative_similarity_mean', 0)
                
                if cohens_d < 0.5:
                    analysis_lines.append(f"\n❌ 严重问题: 图文语义对齐极弱")
                    analysis_lines.append(f"   数据: Cohen's d = {cohens_d:.4f} (效应极小)")
                    analysis_lines.append(f"   正样本相似度: {pos_sim:.4f}")
                    analysis_lines.append(f"   负样本相似度: {neg_sim:.4f}")
                    analysis_lines.append(f"   差异: {pos_sim - neg_sim:.4f} (几乎无差异)")
                    analysis_lines.append("   原因分析:")
                    analysis_lines.append("   1) CLIP模型未在小红书数据上微调")
                    analysis_lines.append("   2) 图片下载可能大量失败，使用了零向量")
                    analysis_lines.append("   3) 文本和图片的内容关联性本身就弱")
                else:
                    analysis_lines.append(f"✅ 正常: 图文对齐良好 (Cohen's d = {cohens_d:.4f})")
        
        # 2. 表征质量层分析
        analysis_lines.append("\n## 2. 表征质量层问题")
        analysis_lines.append("-" * 40)
        
        if 'feature_quality' in results:
            fq = results['feature_quality']
            
            # 方差分析
            if 'variance' in fq:
                for feat_type in ['image', 'text']:
                    if feat_type in fq['variance']:
                        var_data = fq['variance'][feat_type]
                        low_var_pct = var_data['low_var_dims'] / var_data['total_dims'] * 100
                        zero_var_pct = var_data['zero_var_dims'] / var_data['total_dims'] * 100
                        
                        if low_var_pct > 90:
                            analysis_lines.append(f"\n❌ 严重问题: {feat_type}特征方差过低")
                            analysis_lines.append(f"   数据: {var_data['low_var_dims']}/{var_data['total_dims']} ({low_var_pct:.1f}%) 维度是低方差")
                            analysis_lines.append(f"   平均方差: {var_data['mean_variance']:.6f}")
                            analysis_lines.append("   原因分析:")
                            analysis_lines.append("   1) 特征被过度归一化 (L2 normalization)")
                            analysis_lines.append("   2) CLIP输出本身就是归一化的单位向量")
                            analysis_lines.append("   3) 数据集缺乏多样性")
                        elif zero_var_pct > 10:
                            analysis_lines.append(f"\n⚠️ 问题: {feat_type}特征有零方差维度")
                            analysis_lines.append(f"   数据: {var_data['zero_var_dims']}/{var_data['total_dims']} ({zero_var_pct:.1f}%) 维度零方差")
                            analysis_lines.append("   建议: 移除零方差维度")
            
            # PCA分析
            if 'pca' in fq:
                for feat_type in ['image', 'text']:
                    if feat_type in fq['pca']:
                        pca_data = fq['pca'][feat_type]
                        reduction_95 = 1 - pca_data['dims_95'] / pca_data['total_dims']
                        
                        analysis_lines.append(f"\n💡 发现: {feat_type}特征可压缩性")
                        analysis_lines.append(f"   95%方差只需: {pca_data['dims_95']}/{pca_data['total_dims']} 维度")
                        analysis_lines.append(f"   可减少: {reduction_95*100:.1f}% 维度")
                        if reduction_95 > 0.4:
                            analysis_lines.append("   ✅ 建议: 使用PCA降维到{}维".format(pca_data['dims_95']))
        
        # 3. 任务适配层分析
        analysis_lines.append("\n## 3. 任务适配层问题")
        analysis_lines.append("-" * 40)
        
        if 'task_adaptation' in results:
            ta = results['task_adaptation']
            
            # CTR相关性
            if 'correlation' in ta:
                for feat_type in ['image', 'text']:
                    if feat_type in ta['correlation']:
                        corr_data = ta['correlation'][feat_type]
                        max_corr = corr_data.get('max_correlation', 0)
                        sig_dims = corr_data.get('significant_dims', 0)
                        
                        if max_corr < 0.1:
                            analysis_lines.append(f"\n❌ 严重问题: {feat_type}特征与CTR无相关性")
                            analysis_lines.append(f"   最大相关性: {max_corr:.4f}")
                            analysis_lines.append(f"   显著维度数: {sig_dims}/512")
                            analysis_lines.append("   原因分析:")
                            analysis_lines.append("   1) CLIP特征未针对CTR任务优化")
                            analysis_lines.append("   2) 视觉/文本特征不是CTR的主要决定因素")
                            analysis_lines.append("   3) 需要结合其他特征（用户、时间、上下文）")
            
            # 预测性能
            if 'prediction' in ta:
                pred = ta['prediction']
                auc = pred.get('logistic_auc', 0)
                r2 = pred.get('rf_r2', 0)
                
                if auc < 0.65:
                    analysis_lines.append(f"\n⚠️ 问题: CTR预测性能较差")
                    analysis_lines.append(f"   AUC: {auc:.4f}")
                    analysis_lines.append(f"   R²: {r2:.4f}")
                    analysis_lines.append("   说明: 特征对CTR预测贡献有限")
                else:
                    analysis_lines.append(f"\n✅ 正常: CTR预测性能可接受")
                    analysis_lines.append(f"   AUC: {auc:.4f}, R²: {r2:.4f}")
            
            # 消融实验结果
            if 'ablation' in ta:
                ablation_data = ta['ablation']
                if 'contributions' in ablation_data:
                    contribs = ablation_data['contributions']
                    
                    analysis_lines.append("\n### 消融实验分析:")
                    if 'image_features' in contribs:
                        img_contrib = contribs['image_features']
                        status = "✅" if img_contrib['is_positive'] else "❌"
                        analysis_lines.append(f"   {status} Image features: ΔR²={img_contrib['r2_contribution']:.4f} ({img_contrib['significance']})")
                    
                    if 'text_features' in contribs:
                        text_contrib = contribs['text_features']
                        status = "✅" if text_contrib['is_positive'] else "❌"
                        analysis_lines.append(f"   {status} Text features: ΔR²={text_contrib['r2_contribution']:.4f} ({text_contrib['significance']})")
        
        # 4. 根本原因总结
        analysis_lines.append("\n## 4. 根本原因分析")
        analysis_lines.append("-" * 40)
        analysis_lines.append("""
1. **特征提取问题**
   - CLIP输出是L2归一化的单位向量，导致方差极低
   - 可能在pipeline中又进行了额外的归一化
   - 建议：检查pipeline中的特征处理步骤

2. **模型适配问题**
   - 通用CLIP模型不适合小红书内容
   - 缺乏领域特定的fine-tuning
   - 建议：在小红书数据上fine-tune CLIP

3. **数据质量问题**
   - 图片下载失败率可能很高
   - 文本质量可能较差（标题过短）
   - 建议：检查图片URL有效性，统计下载成功率

4. **任务不匹配**
   - CLIP特征可能不是CTR的主要决定因素
   - 需要结合更多特征（用户行为、时序、社交）
   - 建议：CLIP特征作为辅助，不作为主特征
        """)
        
        # 5. 改进方案
        analysis_lines.append("\n## 5. 具体改进方案")
        analysis_lines.append("-" * 40)
        analysis_lines.append("""
🔧 短期改进（1-2天）:
1. 检查特征处理pipeline，移除多余的归一化
2. 使用原始CLIP特征，不做L2归一化
3. PCA降维，减少噪声
4. 添加特征缩放，恢复方差

🚀 中期改进（1周）:
1. 统计图片下载成功率，修复失败问题
2. 尝试不同的CLIP模型变体（如CLIP-ViT-L/14）
3. 结合其他特征（用户、时间、统计）
4. 使用特征选择算法筛选有效维度

🎯 长期改进（2-4周）:
1. 在小红书数据上fine-tune CLIP
2. 探索其他多模态模型（如ALIGN、Florence）
3. 设计任务特定的特征工程
4. 构建端到端的多模态CTR模型
        """)
        
        # 6. 验证建议
        analysis_lines.append("\n## 6. 验证步骤")
        analysis_lines.append("-" * 40)
        analysis_lines.append("""
1. 检查原始特征：
   - 查看image_pipeline输出的原始特征分布
   - 确认是否有额外的归一化操作
   
2. 验证数据质量：
   - 统计实际下载成功的图片比例
   - 检查text_feat是否都有效
   
3. 对比实验：
   - 使用原始特征 vs 归一化特征
   - 不同CLIP模型的效果对比
   - 有/无CLIP特征的模型性能对比
        """)
        
        # 返回报告文本
        return "\n".join(analysis_lines)
    
    def generate_visualizations(self, df: pd.DataFrame, results: Dict) -> None:
        """生成可视化"""
        logger.info("\nGenerating visualizations...")
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Clustering Quality', 'Semantic Consistency', 'Feature Variance Distribution',
                'PCA Explained Variance', 'Feature-CTR Correlation', 'CTR Prediction Performance',
                'Feature Norm Distribution', 'Dimension Variance', 'Feature Importance'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # 1. 聚类质量图
        if 'visual_understanding' in results and 'clustering' in results['visual_understanding']:
            cluster_data = results['visual_understanding']['clustering']
            if 'k_values' in cluster_data and 'silhouette_scores' in cluster_data:
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data['k_values'],
                        y=cluster_data['silhouette_scores'],
                        mode='lines+markers',
                        name='Silhouette Score',
                        marker=dict(size=10, color='blue')
                    ),
                    row=1, col=1
                )
        
        # 2. 语义一致性对比
        if 'visual_understanding' in results and 'semantic_consistency' in results['visual_understanding']:
            sem_data = results['visual_understanding']['semantic_consistency']
            # 创建两组数据用于箱图
            positive_samples = np.random.normal(
                sem_data['positive_similarity_mean'],
                sem_data['positive_similarity_std'],
                100
            )
            negative_samples = np.random.normal(
                sem_data['negative_similarity_mean'],
                sem_data['negative_similarity_std'],
                100
            )
            
            fig.add_trace(
                go.Box(y=positive_samples, name='Same Note', marker_color='green'),
                row=1, col=2
            )
            fig.add_trace(
                go.Box(y=negative_samples, name='Random Pair', marker_color='red'),
                row=1, col=2
            )
        
        # 3. 特征方差分布
        if 'feature_quality' in results and 'variance' in results['feature_quality']:
            var_data = results['feature_quality']['variance']
            if 'image' in var_data:
                categories = ['Zero', 'Low', 'Medium', 'High']
                image_values = [
                    var_data['image']['zero_var_dims'],
                    var_data['image']['low_var_dims'],
                    var_data['image']['med_var_dims'],
                    var_data['image']['high_var_dims']
                ]
                text_values = [
                    var_data['text']['zero_var_dims'],
                    var_data['text']['low_var_dims'],
                    var_data['text']['med_var_dims'],
                    var_data['text']['high_var_dims']
                ]
                
                fig.add_trace(
                    go.Bar(name='Image', x=categories, y=image_values, marker_color='blue'),
                    row=1, col=3
                )
                fig.add_trace(
                    go.Bar(name='Text', x=categories, y=text_values, marker_color='orange'),
                    row=1, col=3
                )
        
        # 4. PCA解释方差曲线
        if 'feature_quality' in results and 'pca' in results['feature_quality']:
            pca_data = results['feature_quality']['pca']
            if 'image' in pca_data and 'cumulative_variance' in pca_data['image']:
                x_dims = list(range(1, len(pca_data['image']['cumulative_variance']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_dims,
                        y=pca_data['image']['cumulative_variance'],
                        mode='lines',
                        name='Image Features',
                        line=dict(color='blue')
                    ),
                    row=2, col=1
                )
                # 添加95%方差线
                fig.add_hline(y=0.95, line_dash="dash", line_color="red", 
                             annotation_text="95%", row=2, col=1)
            
            if 'text' in pca_data and 'cumulative_variance' in pca_data['text']:
                x_dims = list(range(1, len(pca_data['text']['cumulative_variance']) + 1))
                fig.add_trace(
                    go.Scatter(
                        x=x_dims,
                        y=pca_data['text']['cumulative_variance'],
                        mode='lines',
                        name='Text Features',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
        
        # 5. 特征-CTR相关性
        if 'task_adaptation' in results and 'correlation' in results['task_adaptation']:
            corr_data = results['task_adaptation']['correlation']
            if 'image' in corr_data and 'top_dimensions' in corr_data['image']:
                top_dims = corr_data['image']['top_dimensions'][:10]
                dims = [f"Dim {d[0]}" for d in top_dims]
                corrs = [d[1] for d in top_dims]
                
                fig.add_trace(
                    go.Bar(x=dims, y=corrs, name='Image Features', marker_color='blue'),
                    row=2, col=2
                )
        
        # 6. CTR预测性能
        if 'task_adaptation' in results and 'prediction' in results['task_adaptation']:
            pred_data = results['task_adaptation']['prediction']
            metrics = ['AUC', 'R²', 'Spearman']
            values = [
                pred_data.get('logistic_auc', 0),
                pred_data.get('rf_r2', 0),
                pred_data.get('spearman_correlation', 0)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=metrics,
                    y=values,
                    mode='markers+lines',
                    marker=dict(size=15, color=values, colorscale='RdYlGn', showscale=True),
                    line=dict(color='gray', dash='dash')
                ),
                row=2, col=3
            )
        
        # 7. 特征L2范数分布
        if self.image_feat_cols:
            image_features = df[self.image_feat_cols].values
            norms = np.linalg.norm(image_features, axis=1)
            
            fig.add_trace(
                go.Histogram(x=norms, nbinsx=50, name='Image Feature Norms', 
                           marker_color='blue', opacity=0.7),
                row=3, col=1
            )
        
        # 8. 维度方差散点图
        if 'feature_quality' in results and 'variance' in results['feature_quality']:
            if 'image' in results['feature_quality']['variance']:
                variances = results['feature_quality']['variance']['image']['variances'][:100]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(variances))),
                        y=variances,
                        mode='markers',
                        name='Dimension Variance',
                        marker=dict(size=5, color=variances, colorscale='Viridis')
                    ),
                    row=3, col=2
                )
        
        # 9. 特征重要性
        if 'task_adaptation' in results and 'importance' in results['task_adaptation']:
            imp_data = results['task_adaptation']['importance']
            categories = ['Image', 'Text']
            mean_importances = [
                imp_data['image']['mean_importance'],
                imp_data['text']['mean_importance']
            ]
            max_importances = [
                imp_data['image']['max_importance'],
                imp_data['text']['max_importance']
            ]
            
            fig.add_trace(
                go.Bar(name='Mean Importance', x=categories, y=mean_importances),
                row=3, col=3
            )
            fig.add_trace(
                go.Bar(name='Max Importance', x=categories, y=max_importances),
                row=3, col=3
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="CLIP Feature Evaluation Dashboard",
            title_font_size=20
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="K Value", row=1, col=1)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=1)
        
        fig.update_yaxes(title_text="Cosine Similarity", row=1, col=2)
        
        fig.update_xaxes(title_text="Variance Level", row=1, col=3)
        fig.update_yaxes(title_text="Number of Dimensions", row=1, col=3)
        
        fig.update_xaxes(title_text="Number of Components", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Variance", row=2, col=1)
        
        fig.update_xaxes(title_text="Dimension", row=2, col=2)
        fig.update_yaxes(title_text="Correlation with CTR", row=2, col=2)
        
        fig.update_xaxes(title_text="Metric", row=2, col=3)
        fig.update_yaxes(title_text="Score", row=2, col=3)
        
        fig.update_xaxes(title_text="L2 Norm", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        
        fig.update_xaxes(title_text="Dimension Index", row=3, col=2)
        fig.update_yaxes(title_text="Variance", row=3, col=2)
        
        fig.update_xaxes(title_text="Feature Type", row=3, col=3)
        fig.update_yaxes(title_text="Importance", row=3, col=3)
        
        # 保存图表
        output_file = self.output_dir / "evaluation_plots.html"
        fig.write_html(str(output_file))
        logger.info(f"Saved visualizations to {output_file}")
    
    def generate_report(self, df: pd.DataFrame, results: Dict) -> None:
        """生成评估报告"""
        logger.info("\nGenerating evaluation report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 生成问题分析报告
        analysis_report = self.analyze_evaluation_results(results)
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CLIP Feature Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; }}
                .good {{ background: #d4edda; }}
                .warning {{ background: #fff3cd; }}
                .bad {{ background: #f8d7da; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
                .recommendation {{ margin-top: 30px; padding: 20px; background: #e7f3ff; }}
                .analysis {{ margin-top: 30px; padding: 20px; background: #f8f9fa; border-left: 4px solid #007bff; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>CLIP Feature Evaluation Report</h1>
            <p>Generated: {timestamp}</p>
            <p>Total Samples (after filtering): {len(df):,}</p>
        """
        
        # 添加过滤统计（如果有）
        if hasattr(self, 'filter_stats'):
            fs = self.filter_stats
            retention_rate = 100 - fs['filter_percentage']
            css_class = 'good' if retention_rate > 50 else 'warning' if retention_rate > 20 else 'bad'
            
            html_content += f"""
            <h2>0. Data Filtering Statistics 数据过滤统计</h2>
            <div class="metric {css_class}">
                <strong>Filter Configuration:</strong><br>
                Min Impression Threshold: {fs['min_impression_threshold']:,}<br>
                Filter Zero Features: {'Yes' if fs['filter_zeros'] else 'No'}<br>
                <br>
                <strong>Filtering Results:</strong><br>
                Original Samples: {fs['original_size']:,}<br>
                Filtered by Impression: {fs['filtered_by_impression']:,}<br>
                Filtered by Zero Features: {fs['filtered_by_zeros']:,}<br>
                Total Filtered: {fs['total_filtered']:,} ({fs['filter_percentage']:.1f}%)<br>
                <strong>Final Samples: {fs['final_size']:,} ({retention_rate:.1f}% retained)</strong>
            </div>
            """
            
            if retention_rate < 30:
                html_content += """
                <div class="warning">
                    <strong>⚠️ Warning:</strong> More than 70% of data was filtered out. 
                    This may affect the reliability of the evaluation results. 
                    Consider adjusting the filtering thresholds or investigating why so many samples have low impressions or zero features.
                </div>
                """
        
        html_content += """
            <h2>1. Visual Understanding Layer 视觉理解层</h2>
        """
        
        # 添加视觉理解层结果
        if 'visual_understanding' in results:
            vu_results = results['visual_understanding']
            
            # 聚类质量
            if 'clustering' in vu_results:
                cluster_data = vu_results['clustering']
                score = cluster_data.get('silhouette_score', 0)
                css_class = 'good' if score > 0.5 else 'warning' if score > 0.3 else 'bad'
                html_content += f"""
                <div class="metric {css_class}">
                    <strong>Clustering Quality:</strong><br>
                    Best K: {cluster_data.get('best_k', 'N/A')}<br>
                    Silhouette Score: {score:.4f}<br>
                """
                
                if 'category_purity' in cluster_data:
                    html_content += "<strong>Category Purity:</strong><br>"
                    for cat, purity in cluster_data['category_purity'].items():
                        html_content += f"  {cat}: {purity:.4f}<br>"
                
                html_content += "</div>"
            
            # 语义一致性
            if 'semantic_consistency' in vu_results:
                sem_data = vu_results['semantic_consistency']
                pos_sim = sem_data.get('positive_similarity_mean', 0)
                neg_sim = sem_data.get('negative_similarity_mean', 0)
                diff = pos_sim - neg_sim
                css_class = 'good' if diff > 0.3 else 'warning' if diff > 0.1 else 'bad'
                
                html_content += f"""
                <div class="metric {css_class}">
                    <strong>Semantic Consistency:</strong><br>
                    Same Note Similarity: {pos_sim:.4f} ± {sem_data.get('positive_similarity_std', 0):.4f}<br>
                    Random Pair Similarity: {neg_sim:.4f} ± {sem_data.get('negative_similarity_std', 0):.4f}<br>
                    Effect Size (Cohen's d): {sem_data.get('cohens_d', 0):.4f}<br>
                    P-value: {sem_data.get('p_value', 1):.6f}
                </div>
                """
        
        html_content += "<h2>2. Feature Quality Layer 表征质量层</h2>"
        
        # 添加特征质量结果
        if 'feature_quality' in results:
            fq_results = results['feature_quality']
            
            # 方差分析
            if 'variance' in fq_results:
                var_data = fq_results['variance']
                for feat_type in ['image', 'text']:
                    if feat_type in var_data:
                        data = var_data[feat_type]
                        zero_pct = data['zero_var_dims'] / data['total_dims'] * 100
                        css_class = 'bad' if zero_pct > 30 else 'warning' if zero_pct > 10 else 'good'
                        
                        html_content += f"""
                        <div class="metric {css_class}">
                            <strong>{feat_type.capitalize()} Feature Variance:</strong><br>
                            Zero Variance Dims: {data['zero_var_dims']}/{data['total_dims']} ({zero_pct:.1f}%)<br>
                            High Variance Dims: {data['high_var_dims']}/{data['total_dims']} ({data['high_var_dims']/data['total_dims']*100:.1f}%)<br>
                            Mean Variance: {data['mean_variance']:.6f}
                        </div>
                        """
            
            # PCA分析
            if 'pca' in fq_results:
                pca_data = fq_results['pca']
                for feat_type in ['image', 'text']:
                    if feat_type in pca_data:
                        data = pca_data[feat_type]
                        reduction = 1 - data['dims_95'] / data['total_dims']
                        css_class = 'good' if reduction > 0.5 else 'warning' if reduction > 0.2 else 'bad'
                        
                        html_content += f"""
                        <div class="metric {css_class}">
                            <strong>{feat_type.capitalize()} PCA Analysis:</strong><br>
                            Dims for 95% variance: {data['dims_95']}/{data['total_dims']} (reduction: {reduction*100:.1f}%)<br>
                            Dims for 99% variance: {data['dims_99']}/{data['total_dims']}
                        </div>
                        """
        
        html_content += "<h2>3. Task Adaptation Layer 任务适配层</h2>"
        
        # 添加任务适配结果
        if 'task_adaptation' in results:
            ta_results = results['task_adaptation']
            
            # 预测能力
            if 'prediction' in ta_results:
                pred_data = ta_results['prediction']
                auc = pred_data.get('logistic_auc', 0)
                r2 = pred_data.get('rf_r2', 0)
                css_class = 'good' if auc > 0.7 else 'warning' if auc > 0.6 else 'bad'
                
                html_content += f"""
                <div class="metric {css_class}">
                    <strong>CTR Prediction Performance:</strong><br>
                    Logistic Regression AUC: {auc:.4f}<br>
                    Random Forest R²: {r2:.4f}<br>
                    Spearman Correlation: {pred_data.get('spearman_correlation', 0):.4f}
                </div>
                """
            
            # 消融实验结果
            if 'ablation' in ta_results:
                ablation_data = ta_results['ablation']
                html_content += "<h3>Ablation Study Results 消融实验</h3>"
                
                # 特征贡献
                if 'contributions' in ablation_data:
                    contribs = ablation_data['contributions']
                    
                    if 'image_features' in contribs:
                        img_contrib = contribs['image_features']
                        css_class = 'good' if img_contrib['is_positive'] else 'bad'
                        html_content += f"""
                        <div class="metric {css_class}">
                            <strong>Image Features Contribution:</strong><br>
                            ΔR²: {img_contrib['r2_contribution']:.4f}<br>
                            Relative contribution: {img_contrib['relative_contribution']:.1f}%<br>
                            Status: {img_contrib['significance']}
                        </div>
                        """
                    
                    if 'text_features' in contribs:
                        text_contrib = contribs['text_features']
                        css_class = 'good' if text_contrib['is_positive'] else 'bad'
                        html_content += f"""
                        <div class="metric {css_class}">
                            <strong>Text Features Contribution:</strong><br>
                            ΔR²: {text_contrib['r2_contribution']:.4f}<br>
                            Relative contribution: {text_contrib['relative_contribution']:.1f}%<br>
                            Status: {text_contrib['significance']}
                        </div>
                        """
                
                # 性能对比表
                if 'feature_sets_performance' in ablation_data:
                    perf_data = ablation_data['feature_sets_performance']
                    html_content += """
                    <table>
                        <tr><th>Feature Set</th><th>R²</th><th>MSE</th><th>Spearman</th></tr>
                    """
                    for name, metrics in sorted(perf_data.items()):
                        html_content += f"""
                        <tr>
                            <td>{name}</td>
                            <td>{metrics['r2']:.4f}</td>
                            <td>{metrics['mse']:.6f}</td>
                            <td>{metrics['spearman']:.4f}</td>
                        </tr>
                        """
                    html_content += "</table>"
                
                # 建议
                if 'recommendation' in ablation_data:
                    html_content += f"""
                    <div class="recommendation">
                        <strong>Ablation Study Recommendation:</strong><br>
                        {ablation_data['recommendation']}
                    </div>
                    """
        
        # 添加问题分析报告
        html_content += f"""
        <div class="analysis">
            <h2>4. Problem Analysis Report 问题分析报告</h2>
            <pre>{analysis_report}</pre>
        </div>
        """
        
        # 添加建议
        html_content += self._generate_recommendations(results)
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存报告
        report_file = self.output_dir / "evaluation_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved report to {report_file}")
        
        # 同时保存JSON格式的结果（转换numpy类型）
        json_file = self.output_dir / "evaluation_results.json"
        
        # 递归转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        results_json = convert_numpy_types(results)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON results to {json_file}")
        
        # 保存问题分析报告为Markdown文件
        analysis_file = self.output_dir / "problem_analysis.md"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_report)
        
        logger.info(f"Saved problem analysis to {analysis_file}")
    
    def _generate_recommendations(self, results: Dict) -> str:
        """生成优化建议"""
        recommendations = []
        
        # 基于评估结果生成建议
        if 'feature_quality' in results:
            fq = results['feature_quality']
            
            # 检查零方差维度
            if 'variance' in fq:
                for feat_type in ['image', 'text']:
                    if feat_type in fq['variance']:
                        zero_dims = fq['variance'][feat_type]['zero_var_dims']
                        total_dims = fq['variance'][feat_type]['total_dims']
                        if zero_dims / total_dims > 0.2:
                            recommendations.append(
                                f"❗ {feat_type.capitalize()} features have {zero_dims}/{total_dims} dead dimensions. "
                                f"Consider removing zero-variance dimensions."
                            )
            
            # 检查PCA压缩潜力
            if 'pca' in fq:
                for feat_type in ['image', 'text']:
                    if feat_type in fq['pca']:
                        dims_95 = fq['pca'][feat_type]['dims_95']
                        total_dims = fq['pca'][feat_type]['total_dims']
                        if dims_95 < total_dims * 0.7:
                            recommendations.append(
                                f"✅ {feat_type.capitalize()} features can be reduced to {dims_95} dimensions "
                                f"(from {total_dims}) while preserving 95% variance."
                            )
        
        # 基于语义一致性
        if 'visual_understanding' in results:
            vu = results['visual_understanding']
            if 'semantic_consistency' in vu:
                cohens_d = vu['semantic_consistency'].get('cohens_d', 0)
                if cohens_d < 0.5:
                    recommendations.append(
                        "⚠️ Weak image-text alignment (Cohen's d < 0.5). "
                        "Consider fine-tuning CLIP on domain-specific data."
                    )
                elif cohens_d > 1.0:
                    recommendations.append(
                        "✅ Strong image-text alignment detected. "
                        "CLIP features capture semantic relationships well."
                    )
        
        # 基于CTR预测
        if 'task_adaptation' in results:
            ta = results['task_adaptation']
            if 'prediction' in ta:
                auc = ta['prediction'].get('logistic_auc', 0)
                if auc < 0.6:
                    recommendations.append(
                        "❗ Poor CTR prediction (AUC < 0.6). "
                        "Features may need task-specific fine-tuning or additional features."
                    )
                elif auc > 0.75:
                    recommendations.append(
                        "✅ Good CTR prediction performance (AUC > 0.75). "
                        "Features are well-suited for the task."
                    )
        
        if not recommendations:
            recommendations.append("No specific recommendations at this time.")
        
        html = '<div class="recommendation"><h3>Recommendations 优化建议</h3><ul>'
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul></div>"
        
        return html
    
    def run_evaluation(self, data_path: str, max_samples: Optional[int] = None,
                      min_impression_threshold: int = 5000, filter_zeros: bool = True) -> Dict:
        """运行完整评估"""
        logger.info("Starting CLIP Feature Evaluation")
        logger.info("=" * 60)
        
        # 加载数据
        df = self.load_data(data_path, max_samples, min_impression_threshold, filter_zeros)
        
        # 运行各层评估
        results = {}
        
        # 添加过滤统计到结果中
        if hasattr(self, 'filter_stats'):
            results['filter_statistics'] = self.filter_stats
        
        # 1. 视觉理解层
        results['visual_understanding'] = self.evaluate_visual_understanding(df)
        
        # 2. 表征质量层
        results['feature_quality'] = self.evaluate_feature_quality(df)
        
        # 3. 任务适配层
        results['task_adaptation'] = self.evaluate_task_adaptation(df)
        
        # 生成可视化和报告
        self.generate_visualizations(df, results)
        self.generate_report(df, results)
        
        logger.info("=" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return results
    
    def analyze_existing_results(self, results_path: str) -> None:
        """分析现有的评估结果"""
        logger.info("Analyzing existing evaluation results...")
        
        # 加载评测结果
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 生成问题分析报告
        analysis_report = self.analyze_evaluation_results(results)
        
        # 保存分析报告
        analysis_file = self.output_dir / "problem_analysis.md"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_report)
        
        logger.info(f"Problem analysis saved to {analysis_file}")
        
        # 打印摘要到控制台
        print("\n" + "=" * 80)
        print("CLIP特征问题分析报告 - 摘要")
        print("=" * 80)
        print(analysis_report.split('\n\n')[0])  # 打印第一段作为摘要


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CLIP Feature Evaluation - Unified Version')
    parser.add_argument('--input', '-i', 
                        default='/Volumes/home/raw_data/image_features_chinese_clip_parquet/',
                        help='Input parquet file or directory')
    parser.add_argument('--output', '-o', default='./evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--max-samples', '-n', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--min-impression', '-m', type=int, default=5000,
                        help='Minimum impression threshold (default: 5000)')
    parser.add_argument('--no-filter-zeros', action='store_true',
                        help='Do not filter out rows with all-zero features')
    parser.add_argument('--analyze-only', '-a', action='store_true',
                        help='Only analyze existing results (skip evaluation)')
    parser.add_argument('--results', '-r', 
                        default='./evaluation_results/evaluation_results.json',
                        help='Path to existing results JSON (for analyze-only mode)')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = CLIPFeatureEvaluator(output_dir=args.output)
    
    if args.analyze_only:
        # 仅分析现有结果
        if not Path(args.results).exists():
            print(f"错误: 找不到结果文件 {args.results}")
            print("请先运行完整评估: python evaluate_clip_features.py --input /path/to/data")
            return
        
        evaluator.analyze_existing_results(args.results)
    else:
        # 运行完整评估
        results = evaluator.run_evaluation(
            data_path=args.input,
            max_samples=args.max_samples,
            min_impression_threshold=args.min_impression,
            filter_zeros=not args.no_filter_zeros  # 注意这里取反
        )
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # 过滤统计摘要
        if 'filter_statistics' in results:
            fs = results['filter_statistics']
            print(f"Filter Statistics:")
            print(f"  Samples after filtering: {fs['final_size']:,}/{fs['original_size']:,} ({100-fs['filter_percentage']:.1f}% retained)")
            print(f"  Filtered by impression < {fs['min_impression_threshold']}: {fs['filtered_by_impression']:,}")
            print(f"  Filtered by zero features: {fs['filtered_by_zeros']:,}")
        
        # 视觉理解层摘要
        if 'visual_understanding' in results:
            vu = results['visual_understanding']
            if 'clustering' in vu:
                print(f"Clustering: Silhouette Score = {vu['clustering'].get('silhouette_score', 0):.4f}")
            if 'semantic_consistency' in vu:
                sem = vu['semantic_consistency']
                print(f"Semantic Consistency: Cohen's d = {sem.get('cohens_d', 0):.4f}")
        
        # 特征质量摘要
        if 'feature_quality' in results:
            fq = results['feature_quality']
            if 'variance' in fq and 'image' in fq['variance']:
                zero_dims = fq['variance']['image']['zero_var_dims']
                total_dims = fq['variance']['image']['total_dims']
                print(f"Feature Quality: {zero_dims}/{total_dims} zero-variance dims")
        
        # 任务适配摘要
        if 'task_adaptation' in results:
            ta = results['task_adaptation']
            if 'prediction' in ta:
                print(f"CTR Prediction: AUC = {ta['prediction'].get('logistic_auc', 0):.4f}")
            if 'ablation' in ta and 'contributions' in ta['ablation']:
                contribs = ta['ablation']['contributions']
                if 'image_features' in contribs:
                    img_contrib = contribs['image_features']
                    print(f"Image Features: ΔR² = {img_contrib['r2_contribution']:.4f} ({'Positive' if img_contrib['is_positive'] else 'Negative'})")
        
        print("=" * 60)
        print(f"\nFull report saved to: {args.output}/evaluation_report.html")
        print(f"Problem analysis saved to: {args.output}/problem_analysis.md")


if __name__ == "__main__":
    main()
