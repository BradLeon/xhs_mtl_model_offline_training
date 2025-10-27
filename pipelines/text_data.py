#!/usr/bin/env python3
"""
文本pipeline数据处理模块
从run_local_text_pipeline_batch_ssd.py迁移而来
包含BatchDataReader、DataProcessor和DataWriter
"""

import os
import glob
import logging
from datetime import datetime
from typing import List, Optional

from pyspark.sql import SparkSession, DataFrame

from .text_config import TextConfig

logger = logging.getLogger(__name__)


class BatchDataReader:
    """批次数据读取器 - 从LocalBatchTextPipeline._read_note_info_batch迁移"""
    
    def __init__(self, spark: SparkSession, config: TextConfig):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _find_batch_files(self, base_path: str) -> List[str]:
        """查找指定批次的文件，支持多种命名格式"""
        file_paths = []
        
        for i in range(self.config.batch_start, self.config.batch_end + 1):
            # 支持不同的文件命名格式
            patterns = [
                f"{base_path}/part-{i:05d}*.parquet",
                f"{base_path}/part-{i:05d}*.snappy.parquet"
            ]
            
            for pattern in patterns:
                matched_files = glob.glob(pattern)
                file_paths.extend(matched_files)
        
        # 去重并排序
        file_paths = sorted(set(file_paths))
        
        if not file_paths:
            self.logger.warning(f"未找到批次{self.config.batch_start}-{self.config.batch_end}的文件")
            # 备用模式：生成文件名（可能不存在）
            file_paths = [f"{base_path}/part-{i:05d}.parquet" 
                         for i in range(self.config.batch_start, self.config.batch_end + 1)]
        
        self.logger.info(f"找到批次 {self.config.batch_start}-{self.config.batch_end}: {len(file_paths)}个文件")
        return file_paths
    
    def read_note_info_batch(self) -> DataFrame:
        """读取指定批次的note_info文件"""
        file_paths = self._find_batch_files(self.config.note_info_path)
        
        if not file_paths:
            raise ValueError(f"无法找到批次{self.config.batch_start}-{self.config.batch_end}的文件")
        
        try:
            # 尝试读取批次文件
            self.logger.info(f"开始读取{len(file_paths)}个note_info批次文件...")
            df = self.spark.read.parquet(*file_paths)
            
            # 基本验证
            count = df.count()
            self.logger.info(f"成功读取note_info批次数据: {count:,}行")
            
            return df
            
        except Exception as e:
            self.logger.error(f"读取批次文件失败: {e}")
            # 备用方案：读取整个目录（不推荐）
            self.logger.warning("使用备用方案：读取整个note_info目录")
            return self.spark.read.parquet(self.config.note_info_path)
    
    def read_note_stats_full(self) -> DataFrame:
        """读取完整的note_stats数据"""
        self.logger.info("开始读取note_stats数据...")
        
        try:
            df = self.spark.read.parquet(self.config.note_stats_path)
            count = df.count()
            self.logger.info(f"成功读取note_stats数据: {count:,}行")
            return df
            
        except Exception as e:
            self.logger.error(f"读取note_stats失败: {e}")
            raise


class DataProcessor:
    """数据处理器 - 从LocalBatchTextPipeline的处理逻辑迁移"""
    
    def __init__(self, spark: SparkSession, config: TextConfig):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimized_join(self, note_info_df: DataFrame, note_stats_df: DataFrame) -> DataFrame:
        """优化的join操作"""
        self.logger.info("开始执行优化的数据JOIN...")
        
        # 确保join key存在
        if "note_id" not in note_info_df.columns:
            raise ValueError("note_info_df缺少note_id列")
        if "note_id" not in note_stats_df.columns:
            raise ValueError("note_stats_df缺少note_id列")
        
        # 预过滤：只保留有统计数据的note
        note_stats_ids = note_stats_df.select("note_id").distinct()
        filtered_note_info = note_info_df.join(note_stats_ids, "note_id", "inner")
        
        self.logger.info(f"预过滤后的note_info: {filtered_note_info.count():,}行")
        
        # 执行主要的join
        merged_df = filtered_note_info.join(note_stats_df, "note_id", "inner")
        
        # 应用最小曝光过滤
        if self.config.min_impression > 0:
            before_count = merged_df.count()
            merged_df = merged_df.filter(merged_df.imp_num >= self.config.min_impression)
            after_count = merged_df.count()
            self.logger.info(f"最小曝光过滤 (>={self.config.min_impression}): {before_count:,} -> {after_count:,}行")
        
        self.logger.info("数据JOIN完成")
        return merged_df
    
    def extract_features(self, df: DataFrame) -> DataFrame:
        """提取文本特征（如果需要的话）"""
        # 这里可以添加文本特征提取逻辑
        # 目前直接返回原始数据
        self.logger.info("特征提取（当前为直通模式）")
        return df
    
    def apply_data_quality_filters(self, df: DataFrame) -> DataFrame:
        """应用数据质量过滤器"""
        self.logger.info("应用数据质量过滤器...")
        
        original_count = df.count()
        
        # 过滤空的note_id
        df = df.filter(df.note_id.isNotNull() & (df.note_id != ""))
        
        # 过滤异常的统计数据
        df = df.filter(
            (df.imp_num >= 0) & 
            (df.click_num >= 0) & 
            (df.click_num <= df.imp_num)
        )
        
        final_count = df.count()
        self.logger.info(f"数据质量过滤: {original_count:,} -> {final_count:,}行")
        
        return df


class DataWriter:
    """数据写入器 - 从LocalBatchTextPipeline.save_data迁移"""
    
    def __init__(self, spark: SparkSession, config: TextConfig):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_partitions(self, df: DataFrame) -> int:
        """计算最优分区数"""
        current_partitions = df.rdd.getNumPartitions()
        optimal_partitions = min(200, max(10, current_partitions))
        
        self.logger.info(f"分区优化: {current_partitions} -> {optimal_partitions}")
        return optimal_partitions
    
    def save_batch_data(self, df: DataFrame, custom_output_path: Optional[str] = None) -> str:
        """保存批次数据"""
        # 确定输出路径
        if custom_output_path:
            output_path = custom_output_path
        else:
            output_path = self.config.get_output_path()
        
        self.logger.info(f"开始保存数据到: {output_path}")
        
        # 优化分区数
        optimal_partitions = self.calculate_optimal_partitions(df)
        
        # 重新分区并保存
        df.repartition(optimal_partitions) \
          .write \
          .mode("overwrite") \
          .option("compression", self.config.parquet_compression) \
          .parquet(output_path)
        
        # 计算输出文件大小
        total_size_mb = self._calculate_output_size(output_path)
        
        self.logger.info(f"数据保存完成: {output_path}")
        self.logger.info(f"输出文件大小: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
        
        return output_path
    
    def _calculate_output_size(self, output_path: str) -> float:
        """计算输出文件大小（MB）"""
        try:
            total_size = 0
            for root, _, files in os.walk(output_path):
                for file in files:
                    if file.endswith('.parquet'):
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            
            return total_size / (1024 * 1024)  # 转换为MB
            
        except Exception as e:
            self.logger.warning(f"无法计算输出文件大小: {e}")
            return 0.0
    
    def verify_output(self, output_path: str) -> dict:
        """验证输出文件"""
        try:
            # 尝试读取保存的文件
            test_df = self.spark.read.parquet(output_path)
            row_count = test_df.count()
            column_count = len(test_df.columns)
            
            return {
                'success': True,
                'row_count': row_count,
                'column_count': column_count,
                'size_mb': self._calculate_output_size(output_path)
            }
            
        except Exception as e:
            self.logger.error(f"输出验证失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class DataPipeline:
    """数据处理流水线 - 整合所有数据操作"""
    
    def __init__(self, spark: SparkSession, config: TextConfig):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        self.reader = BatchDataReader(spark, config)
        self.processor = DataProcessor(spark, config)
        self.writer = DataWriter(spark, config)
    
    def run_full_pipeline(self) -> dict:
        """运行完整的数据处理流水线"""
        self.logger.info("开始执行完整数据处理流水线...")
        
        try:
            # 1. 读取数据
            self.logger.info("步骤1: 读取数据")
            note_info_df = self.reader.read_note_info_batch()
            note_stats_df = self.reader.read_note_stats_full()
            
            # 2. 数据处理
            self.logger.info("步骤2: 数据处理")
            merged_df = self.processor.optimized_join(note_info_df, note_stats_df)
            filtered_df = self.processor.apply_data_quality_filters(merged_df)
            processed_df = self.processor.extract_features(filtered_df)
            
            # 3. 保存数据
            self.logger.info("步骤3: 保存数据")
            output_path = self.writer.save_batch_data(processed_df)
            
            # 4. 验证输出
            self.logger.info("步骤4: 验证输出")
            verification = self.writer.verify_output(output_path)
            
            # 返回处理结果
            result = {
                'success': True,
                'output_path': output_path,
                'batch_start': self.config.batch_start,
                'batch_end': self.config.batch_end,
                'rows_processed': verification.get('row_count', 0),
                'columns': verification.get('column_count', 0),
                'output_size_mb': verification.get('size_mb', 0),
                'verification': verification
            }
            
            self.logger.info("数据处理流水线完成")
            return result
            
        except Exception as e:
            self.logger.error(f"数据处理流水线失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'batch_start': self.config.batch_start,
                'batch_end': self.config.batch_end
            }