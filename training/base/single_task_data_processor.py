#!/usr/bin/env python3
"""
单任务学习数据处理基类

提供所有单任务训练器的通用数据处理功能：
- 统一数据加载（文件/目录支持）
- 数据质量过滤和清洗
- CTR标签准备和验证
- 数据统计和质量检查

消除单任务训练脚本中的数据处理代码重复
"""

import logging
import random
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseSingleTaskDataProcessor:
    """单任务学习数据处理基类
    
    封装所有单任务训练器通用的数据处理逻辑，包括：
    - 数据加载和采样
    - 质量过滤和清洗
    - 标签准备和验证
    - 统计分析和报告
    """
    
    def __init__(self,
                 input_path: str,
                 sample_size: Optional[int] = None,
                 min_impression_threshold: int = 5000,
                 filter_zeros: bool = False,
                 label_column: str = 'ctr'):
        """初始化数据处理器
        
        Args:
            input_path: 输入文件或目录路径
            sample_size: 随机采样文件数量（仅目录加载时有效）
            min_impression_threshold: 最小曝光量阈值
            filter_zeros: 是否过滤全零CLIP特征样本
            label_column: 标签列名
        """
        self.input_path = input_path
        self.sample_size = sample_size
        self.min_impression_threshold = min_impression_threshold
        self.filter_zeros = filter_zeros
        self.label_column = label_column
        
        # 数据统计
        self.data = None
        self.original_size = 0
        self.filtered_stats = {
            'by_impression': 0,
            'by_zeros': 0,
            'total_filtered': 0
        }
        
        logger.info(f"Initialized single-task data processor")
        logger.info(f"  Input path: {input_path}")
        logger.info(f"  Label column: {label_column}")
        logger.info(f"  Min impression threshold: {min_impression_threshold}")
        logger.info(f"  Filter zero features: {filter_zeros}")
    
    def load_data(self) -> pd.DataFrame:
        """加载数据（支持文件或目录）
        
        Returns:
            加载并预处理后的DataFrame
        """
        logger.info(f"📚 Loading data from: {self.input_path}")
        
        input_path = Path(self.input_path)
        
        if input_path.is_dir():
            self.data = self._load_from_directory(input_path)
        else:
            self.data = self._load_from_file(input_path)
        
        self.original_size = len(self.data)
        
        # 应用数据过滤
        self._apply_filters()
        
        # 准备标签
        self._prepare_label()
        
        # 生成数据统计报告
        self._generate_data_report()
        
        return self.data
    
    def _load_from_directory(self, input_path: Path) -> pd.DataFrame:
        """从目录加载多个parquet文件"""
        parquet_files = list(input_path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {input_path}")
        
        # 随机采样文件
        if self.sample_size and len(parquet_files) > self.sample_size:
            logger.info(f"📌 Random sampling {self.sample_size} files from {len(parquet_files)} files")
            parquet_files = random.sample(parquet_files, self.sample_size)
        
        logger.info(f"📌 Loading {len(parquet_files)} files")
        
        # 加载多个文件
        dfs = []
        for file_path in parquet_files:
            logger.debug(f"   Loading: {file_path.name}")
            df = pd.read_parquet(file_path)
            dfs.append(df)
        
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"✅ Loaded {len(parquet_files)} files, total {len(data):,} rows")
        
        return data
    
    def _load_from_file(self, input_path: Path) -> pd.DataFrame:
        """从单个文件加载数据"""
        data = pd.read_parquet(input_path)
        logger.info(f"✅ Loaded {len(data):,} rows from single file")
        return data
    
    def _apply_filters(self) -> None:
        """应用数据过滤"""
        if self.min_impression_threshold <= 0 and not self.filter_zeros:
            return
        
        logger.info("🔍 Applying data filters...")
        
        # 1. 过滤低曝光量的行
        if self.min_impression_threshold > 0:
            self._filter_by_impression()
        
        # 2. 过滤全零CLIP特征的行
        if self.filter_zeros:
            self._filter_by_zero_features()
        
        # 更新过滤统计
        final_size = len(self.data)
        self.filtered_stats['total_filtered'] = self.original_size - final_size
    
    def _filter_by_impression(self) -> None:
        """根据曝光量阈值过滤"""
        impression_cols = ['imp_num', 'impression']
        impression_col = None
        
        for col in impression_cols:
            if col in self.data.columns:
                impression_col = col
                break
        
        if impression_col is None:
            logger.warning(f"⚠️ No impression column found in {impression_cols}, skipping impression filter")
            return
        
        mask = self.data[impression_col] >= self.min_impression_threshold
        filtered_count = (~mask).sum()
        self.data = self.data[mask]
        self.filtered_stats['by_impression'] = filtered_count
        
        logger.info(f"🔍 Filtered {filtered_count} rows with {impression_col} < {self.min_impression_threshold}")
    
    def _filter_by_zero_features(self) -> None:
        """过滤全零CLIP特征的样本"""
        # 获取CLIP特征列
        image_feat_cols = [col for col in self.data.columns if col.startswith('image_feat_')]
        text_feat_cols = [col for col in self.data.columns if col.startswith('text_feat_')]
        
        if not image_feat_cols and not text_feat_cols:
            logger.warning("⚠️ No CLIP features found, skipping zero feature filter")
            return
        
        logger.info("🔍 Filtering samples with all-zero CLIP features...")
        
        non_zero_mask = pd.Series(True, index=self.data.index)
        
        # 检查图像特征
        if image_feat_cols:
            image_features = self.data[image_feat_cols].values
            image_non_zero = np.any(image_features != 0, axis=1)
            non_zero_mask = non_zero_mask & image_non_zero
        
        # 检查文本特征
        if text_feat_cols:
            text_features = self.data[text_feat_cols].values
            text_non_zero = np.any(text_features != 0, axis=1)
            # 至少有一个非零特征即可
            if image_feat_cols:
                non_zero_mask = non_zero_mask | text_non_zero
            else:
                non_zero_mask = non_zero_mask & text_non_zero
        
        filtered_count = (~non_zero_mask).sum()
        self.data = self.data[non_zero_mask]
        self.filtered_stats['by_zeros'] = filtered_count
        
        logger.info(f"🔍 Filtered {filtered_count} rows with all-zero CLIP features")
    
    def _prepare_label(self) -> None:
        """准备标签列"""
        # 检查是否有指定的标签列
        if self.label_column in self.data.columns:
            logger.info(f"✅ Using existing '{self.label_column}' column")
            self._validate_label()
            return
        
        # 尝试从impression和click计算CTR
        if self._try_create_ctr_from_columns():
            return
        
        # 生成随机CTR用于测试
        self._create_random_ctr()
    
    def _try_create_ctr_from_columns(self) -> bool:
        """尝试从现有列创建CTR"""
        # 候选的列名组合
        impression_click_pairs = [
            ('imp_num', 'click_num'),
            ('impression', 'click'),
            ('imp', 'clk')
        ]
        
        for imp_col, click_col in impression_click_pairs:
            if imp_col in self.data.columns and click_col in self.data.columns:
                self.data[self.label_column] = self.data.apply(
                    lambda x: x[click_col] / x[imp_col] if x[imp_col] > 0 else 0,
                    axis=1
                )
                logger.info(f"✅ Created '{self.label_column}' column from {imp_col} and {click_col}")
                self._validate_label()
                return True
        
        return False
    
    def _create_random_ctr(self) -> None:
        """生成随机CTR用于测试"""
        logger.warning("⚠️ No CTR or impression/click columns found, generating random CTR for testing")
        self.data[self.label_column] = np.random.beta(2, 10, size=len(self.data))
        logger.warning(f"⚠️ Generated random '{self.label_column}' column using Beta(2,10) distribution")
    
    def _validate_label(self) -> None:
        """验证标签列的有效性"""
        label_data = self.data[self.label_column]
        
        # 基本统计
        stats = label_data.describe()
        logger.info(f"📊 Label '{self.label_column}' statistics:")
        logger.info(f"   Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        logger.info(f"   Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
        logger.info(f"   Non-null count: {label_data.count()}/{len(label_data)}")
        
        # 质量检查
        issues = []
        
        if label_data.isnull().sum() > 0:
            null_count = label_data.isnull().sum()
            issues.append(f"Contains {null_count} null values")
        
        if label_data.std() < 1e-8:
            issues.append("Label has very low variance (nearly constant)")
        
        if (label_data < 0).sum() > 0:
            negative_count = (label_data < 0).sum()
            issues.append(f"Contains {negative_count} negative values")
        
        if (label_data > 1).sum() > 0 and self.label_column == 'ctr':
            high_count = (label_data > 1).sum()
            issues.append(f"CTR contains {high_count} values > 1 (unusual for CTR)")
        
        if issues:
            logger.warning(f"⚠️ Label quality issues detected:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ Label quality check passed")
    
    def _generate_data_report(self) -> None:
        """生成数据统计报告"""
        final_size = len(self.data)
        
        if self.filtered_stats['total_filtered'] > 0:
            filter_percentage = (self.filtered_stats['total_filtered'] / self.original_size * 100)
            
            logger.info("=" * 60)
            logger.info("📊 DATA LOADING AND FILTERING REPORT")
            logger.info("=" * 60)
            logger.info(f"  Original samples: {self.original_size:,}")
            logger.info(f"  Filtered by impression threshold: {self.filtered_stats['by_impression']:,}")
            logger.info(f"  Filtered by zero features: {self.filtered_stats['by_zeros']:,}")
            logger.info(f"  Total filtered: {self.filtered_stats['total_filtered']:,} ({filter_percentage:.1f}%)")
            logger.info(f"  Final samples: {final_size:,}")
            logger.info("=" * 60)
        
        # 如果过滤后样本太少，发出警告
        if final_size < 100:
            logger.warning(f"⚠️ WARNING: Only {final_size} samples remain after filtering!")
            logger.warning("This may affect model training reliability.")
        
        # 数据形状信息
        logger.info(f"📋 Columns available: {len(self.data.columns)} columns")
        logger.info(f"📊 Final data shape: {self.data.shape}")
    
    def get_data_info(self) -> dict:
        """获取数据信息摘要
        
        Returns:
            包含数据统计信息的字典
        """
        if self.data is None:
            return {'error': 'No data loaded'}
        
        return {
            'original_size': self.original_size,
            'final_size': len(self.data),
            'filtered_stats': self.filtered_stats.copy(),
            'columns_count': len(self.data.columns),
            'label_column': self.label_column,
            'label_stats': self.data[self.label_column].describe().to_dict() if self.label_column in self.data.columns else None
        }
    
    def validate_data_quality(self) -> dict:
        """验证数据质量
        
        Returns:
            数据质量报告字典
        """
        if self.data is None:
            return {'error': 'No data to validate'}
        
        report = {
            'total_samples': len(self.data),
            'total_columns': len(self.data.columns),
            'missing_values': {},
            'data_types': {},
            'issues': []
        }
        
        # 检查缺失值
        for col in self.data.columns:
            null_count = self.data[col].isnull().sum()
            if null_count > 0:
                report['missing_values'][col] = int(null_count)
        
        # 检查数据类型
        for col in self.data.columns:
            report['data_types'][col] = str(self.data[col].dtype)
        
        # 检查常见问题
        if len(self.data) == 0:
            report['issues'].append('Dataset is empty')
        
        if self.label_column not in self.data.columns:
            report['issues'].append(f'Label column {self.label_column} not found')
        
        # 检查CLIP特征
        clip_features = [col for col in self.data.columns if 
                        col.startswith(('image_feat_', 'text_feat_', 'cover_image_feat_', 'title_feat_'))]
        if not clip_features:
            report['issues'].append('No CLIP features found')
        
        return report