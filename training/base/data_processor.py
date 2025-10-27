#!/usr/bin/env python3
"""
多任务学习数据处理基类

提供所有MTL训练器的通用数据处理功能：
- 统一数据加载逻辑
- 统一目标变量工程化
- 统一数据过滤和质量检查
- 任务名映射管理

消除MTL训练脚本中的数据处理代码重复
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseMTLDataProcessor:
    """多任务学习数据处理基类
    
    封装所有MTL训练器通用的数据处理逻辑，包括：
    - 数据加载和合并
    - 目标变量工程化
    - 数据过滤和质量检查
    - 任务配置管理
    """
    
    def __init__(self, 
                 input_path: str,
                 sample_size: Optional[int] = None,
                 min_impression_threshold: int = 5000,
                 tasks: Optional[List[str]] = None):
        """初始化数据处理器
        
        Args:
            input_path: 输入数据路径
            sample_size: 随机采样文件数量
            min_impression_threshold: 最小曝光数阈值
            tasks: 任务列表
        """
        self.input_path = input_path
        self.sample_size = sample_size
        self.min_impression_threshold = min_impression_threshold
        
        # 定义所有可用任务的映射关系
        self.available_tasks = {
            'ctr': 'ctr',
            'like_rate': 'like_rate',
            'fav_rate': 'fav_rate',
            'comment_rate': 'comment_rate',
            'share_rate': 'share_rate',
            'follow_rate': 'follow_rate',
            'interaction_rate': 'interaction_rate',
            'ces_rate': 'ces_rate',
            'impression': 'impression_log',
            'sort_score': 'sort_score2'
        }
        
        # 选择任务
        if tasks:
            invalid_tasks = [t for t in tasks if t not in self.available_tasks]
            if invalid_tasks:
                raise ValueError(f"Invalid tasks: {invalid_tasks}. Available: {list(self.available_tasks.keys())}")
            self.selected_tasks = tasks
        else:
            self.selected_tasks = list(self.available_tasks.keys())
        
        logger.info(f"Initialized MTL data processor for {len(self.selected_tasks)} tasks: {self.selected_tasks}")
    
    def get_column_name(self, task_name: str) -> str:
        """获取任务对应的实际DataFrame列名
        
        Args:
            task_name: 用户友好的任务名
            
        Returns:
            DataFrame中的实际列名
        """
        return self.available_tasks.get(task_name, task_name)
    
    def load_data(self) -> pd.DataFrame:
        """加载和合并数据
        
        Returns:
            加载后的DataFrame
        """
        logger.info(f"Loading data from {self.input_path}")
        
        input_path = Path(self.input_path)
        
        if input_path.is_file():
            # 单个文件
            df = pd.read_parquet(input_path)
            logger.info(f"Loaded single file: {len(df)} rows")
        else:
            # 目录：读取所有parquet文件
            parquet_files = list(input_path.glob("**/*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {input_path}")
            
            # 随机采样文件
            if self.sample_size and len(parquet_files) > self.sample_size:
                parquet_files = random.sample(parquet_files, self.sample_size)
                logger.info(f"Randomly sampled {len(parquet_files)} files")
            
            # 读取并合并
            dfs = []
            for file_path in parquet_files:
                try:
                    file_df = pd.read_parquet(file_path)
                    dfs.append(file_df)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
            
            if not dfs:
                raise ValueError("No valid parquet files could be loaded")
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(parquet_files)} files: {len(df)} total rows")
        
        # 应用基础数据过滤
        df = self._apply_basic_filtering(df)
        
        return df
    
    def _apply_basic_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用基础数据过滤
        
        Args:
            df: 原始DataFrame
            
        Returns:
            过滤后的DataFrame
        """
        original_size = len(df)
        
        # 过滤低曝光数据
        if 'imp_num' in df.columns:
            df = df[df['imp_num'] >= self.min_impression_threshold]
            logger.info(f"Filtered by impression threshold: {len(df)}/{original_size} rows retained")
        
        # 移除缺失的关键列
        required_cols = ['imp_num', 'click_num']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        df = df.dropna(subset=[col for col in required_cols if col in df.columns])
        logger.info(f"After removing NaN: {len(df)} rows")
        
        return df
    
    def engineer_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """工程化目标变量
        
        为所有MTL任务创建目标变量列
        
        Args:
            df: 输入DataFrame
            
        Returns:
            包含所有目标变量的DataFrame
        """
        logger.info("Engineering target variables...")
        
        # CTR (点击率)
        if 'ctr' not in df.columns and 'imp_num' in df.columns and 'click_num' in df.columns:
            df['ctr'] = np.where(df['imp_num'] > 0, df['click_num'] / df['imp_num'], 0.0)
        
        # 基于曝光数的率计算
        if 'imp_num' in df.columns:
            # 点赞率
            if 'like_rate' not in df.columns and 'like_num' in df.columns:
                df['like_rate'] = np.where(df['imp_num'] > 0, df['like_num'] / df['imp_num'], 0.0)
            
            # 收藏率
            if 'fav_rate' not in df.columns and 'fav_num' in df.columns:
                df['fav_rate'] = np.where(df['imp_num'] > 0, df['fav_num'] / df['imp_num'], 0.0)
            
            # 评论率
            if 'comment_rate' not in df.columns and 'cmt_num' in df.columns:
                df['comment_rate'] = np.where(df['imp_num'] > 0, df['cmt_num'] / df['imp_num'], 0.0)
            
            # 分享率
            if 'share_rate' not in df.columns and 'share_num' in df.columns:
                df['share_rate'] = np.where(df['imp_num'] > 0, df['share_num'] / df['imp_num'], 0.0)
            
            # 加粉率
            if 'follow_rate' not in df.columns and 'follow_from_discovery_num' in df.columns:
                df['follow_rate'] = np.where(df['imp_num'] > 0, df['follow_from_discovery_num'] / df['imp_num'], 0.0)
            
            # 互动率
            if 'interaction_rate' not in df.columns:
                interaction_cols = ['like_num', 'fav_num', 'cmt_num', 'share_num', 'follow_from_discovery_num']
                available_cols = [col for col in interaction_cols if col in df.columns]
                if available_cols:
                    df['total_interactions'] = df[available_cols].fillna(0).sum(axis=1)
                    df['interaction_rate'] = np.where(df['imp_num'] > 0, df['total_interactions'] / df['imp_num'], 0.0)
            
            # CES率（加权评分）
            if 'ces_rate' not in df.columns:
                ces_weights = {'like_num': 1, 'fav_num': 1, 'cmt_num': 4, 'share_num': 4, 'follow_from_discovery_num': 4}
                ces_score = 0
                for col, weight in ces_weights.items():
                    if col in df.columns:
                        ces_score += df[col].fillna(0) * weight
                df['ces_rate'] = np.where(df['imp_num'] > 0, ces_score / df['imp_num'], 0.0)
        
        # 曝光数预测（对数变换，便于回归）
        if 'impression_log' not in df.columns and 'imp_num' in df.columns:
            df['impression_log'] = np.log(np.clip(df['imp_num'], 1, 10000000))  # 防止log(0)， 假设曝光上线1000万。
            logger.info(f"Transformed impression: range {df['impression_log'].min():.2f} - {df['impression_log'].max():.2f}")
        
        # 排序分数（直接使用sort_score2）
        if 'sort_score2' in df.columns:
            # 标准化排序分数
            df['sort_score2'] = df['sort_score2'].fillna(df['sort_score2'].median())
            logger.info(f"Sort score range: {df['sort_score2'].min():.2f} - {df['sort_score2'].max():.2f}")
        
        # 执行目标变量质量分析
        self._analyze_target_quality(df)
        
        return df
    
    def _analyze_target_quality(self, df: pd.DataFrame) -> None:
        """分析目标变量质量
        
        Args:
            df: 包含目标变量的DataFrame
        """
        logger.info("="*60)
        logger.info("TARGET VARIABLES DISTRIBUTION ANALYSIS")
        logger.info("="*60)
        
        for task_name in self.selected_tasks:
            target_col = self.available_tasks[task_name]
            if target_col in df.columns:
                data = df[target_col].fillna(0)
                stats = data.describe()
                
                # 基础统计
                logger.info(f"{task_name} ({target_col}):")
                logger.info(f"  Count: {len(data)}")
                logger.info(f"  Mean: {stats['mean']:.6f}")
                logger.info(f"  Std: {stats['std']:.6f}")
                logger.info(f"  Min: {stats['min']:.6f}")
                logger.info(f"  Max: {stats['max']:.6f}")
                logger.info(f"  Median: {stats['50%']:.6f}")
                
                # 数据质量检查
                zero_count = (data == 0).sum()
                nan_count = df[target_col].isna().sum()
                unique_count = data.nunique()
                
                logger.info(f"  Zero values: {zero_count} ({zero_count/len(data)*100:.2f}%)")
                logger.info(f"  NaN values: {nan_count} ({nan_count/len(df)*100:.2f}%)")
                logger.info(f"  Unique values: {unique_count}")
                logger.info(f"  Variance: {data.var():.8f}")
                
                # 常数检查
                if unique_count <= 1:
                    logger.warning(f"  ⚠️  CONSTANT TARGET: Only {unique_count} unique value(s)!")
                elif data.var() < 1e-8:
                    logger.warning(f"  ⚠️  NEAR-CONSTANT: Very low variance ({data.var():.2e})")
                elif unique_count < 10:
                    logger.warning(f"  ⚠️  LIMITED DIVERSITY: Only {unique_count} unique values")
                
                # 分位数分析
                if unique_count > 1:
                    percentiles = [1, 5, 25, 75, 95, 99]
                    pct_values = [data.quantile(p/100) for p in percentiles]
                    pct_str = ", ".join([f"P{p}={v:.6f}" for p, v in zip(percentiles, pct_values)])
                    logger.info(f"  Percentiles: {pct_str}")
                
                logger.info("")  # 空行分隔
            else:
                logger.warning(f"Target column {target_col} not found for task {task_name}")
        
        logger.info("="*60)
    
    def get_task_columns(self) -> List[str]:
        """获取所有任务对应的列名
        
        Returns:
            任务列名列表
        """
        return [self.available_tasks[task] for task in self.selected_tasks if self.available_tasks[task]]
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证数据质量
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            数据质量报告
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_targets': [],
            'constant_targets': [],
            'low_variance_targets': []
        }
        
        # 检查目标变量
        for task_name in self.selected_tasks:
            target_col = self.available_tasks[task_name]
            if target_col not in df.columns:
                report['missing_targets'].append(task_name)
            else:
                data = df[target_col].fillna(0)
                if data.nunique() <= 1:
                    report['constant_targets'].append(task_name)
                elif data.var() < 1e-8:
                    report['low_variance_targets'].append(task_name)
        
        return report