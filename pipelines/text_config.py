#!/usr/bin/env python3
"""
文本特征提取配置模块（重构版）
继承自BasePipelineConfig，使用统一的配置抽象
"""

import os
import psutil
from dataclasses import dataclass
from typing import Optional

from .base_config import (
    BasePipelineConfig, BatchProcessingMixin, SparkConfigMixin, 
    ConfigValidator, create_base_config_from_args
)


@dataclass
class TextConfig(BasePipelineConfig, BatchProcessingMixin, SparkConfigMixin):
    """文本特征提取配置（重构版）
    
    继承自多个混入类，提供完整的文本pipeline配置功能
    """
    
    # 重写基类的必需字段（具体的默认值）
    input_path: str = "/Volumes/home/raw_data/note_info_parquet"  # note_info路径作为主输入
    output_path: str = "/Volumes/home/raw_data/text_features_parquet"
    
    # 文本特征专用路径配置
    note_stats_path: str = "/Volumes/home/raw_data/note_stats_parquet"
    note_info_path: str = "/Volumes/home/raw_data/note_info_parquet"
    
    # 重写Spark配置的默认值
    app_name: str = "XHS_Text_Pipeline"
    
    # 文本处理特定配置
    min_impression: int = 500  # 最小曝光阈值
    
    # 文本特定的Spark配置
    parquet_compression: str = "snappy"
    batch_times_window_size: int = 10  # 批次时间窗口大小
    
    def _setup_dynamic_config(self):
        """设置动态配置"""
        # 自动计算内存和CPU配置
        if self.driver_memory_gb is None:
            available_memory = psutil.virtual_memory().available
            self.driver_memory_gb = min(56, int(available_memory * 0.7 / (1024**3)))
        
        if self.executor_cores is None:
            num_cores = psutil.cpu_count(logical=True)
            self.executor_cores = max(4, num_cores - 2)
    
    def validate(self) -> None:
        """验证配置的有效性"""
        # 使用基类验证器
        ConfigValidator.validate_paths(self)
        ConfigValidator.validate_performance_config(self)
        ConfigValidator.validate_batch_config(self)
        ConfigValidator.validate_spark_config(self)
        
        # 文本特定验证
        if self.min_impression < 0:
            raise ValueError("min_impression must be non-negative")
        
        if not os.path.exists(self.note_stats_path):
            raise ValueError(f"note_stats_path does not exist: {self.note_stats_path}")
    
    def get_output_path(self) -> str:
        """获取实际输出路径"""
        batch_info = self.get_batch_info()
        timestamp = self.get_timestamp_suffix()
        return f"{self.output_path}/{batch_info}/dt={timestamp}"
    
    def setup_environment(self) -> None:
        """设置环境变量和目录"""
        # 调用Spark混入的环境设置
        self.setup_spark_environment()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    def get_spark_configs(self) -> dict:
        """获取完整的Spark配置字典"""
        base_configs = {
            "spark.app.name": self.app_name,
            "spark.master": f"local[{self.executor_cores}]",
            "spark.driver.memory": f"{self.driver_memory_gb}g",
            "spark.driver.maxResultSize": "10g",
            "spark.local.dir": self.spark_tmp_dir,
            "spark.sql.warehouse.dir": self.spark_warehouse_dir,
            "spark.checkpoint.dir": self.spark_checkpoint_dir,
            
            # SQL优化配置
            "spark.sql.shuffle.partitions": str(self.shuffle_partitions),
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256MB",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.join.preferSortMergeJoin": "true",
            "spark.sql.autoBroadcastJoinThreshold": "-1",
            
            # 内存配置
            "spark.memory.fraction": str(self.memory_fraction),
            "spark.memory.storageFraction": str(self.storage_fraction),
            
            # Shuffle优化
            "spark.shuffle.file.buffer": "2m",
            "spark.reducer.maxSizeInFlight": "128m",
            "spark.shuffle.compress": "true",
            
            # 压缩和序列化
            "spark.io.compression.codec": "lz4",
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "512m",
            
            # Parquet优化
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.parquet.compression.codec": self.parquet_compression,
            "spark.sql.parquet.filterPushdown": "true",
            "spark.sql.parquet.mergeSchema": "false",
            "spark.sql.parquet.enableVectorizedReader": "true",
            
            # 文件处理优化
            "spark.sql.files.maxPartitionBytes": "256MB",
            "spark.sql.files.openCostInBytes": "8MB",
            "spark.hadoop.fs.local.block.size": "134217728",
            
            # 清理和GC优化
            "spark.cleaner.referenceTracking": "true",
            "spark.cleaner.referenceTracking.blocking": "true",
            "spark.cleaner.periodicGC.interval": self.gc_interval,
            
            # 文件输出优化
            "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": "2",
            
            # 并行发现优化
            "spark.sql.sources.parallelPartitionDiscovery.parallelism": "8",
            "spark.sql.sources.parallelPartitionDiscovery.threshold": "32"
        }
        
        return base_configs
    
    def get_timestamp_suffix(self) -> str:
        """获取时间戳后缀"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_config_from_args(args) -> TextConfig:
    """从命令行参数创建TextConfig（兼容性函数）"""
    return create_base_config_from_args(TextConfig, args)


def get_default_config() -> TextConfig:
    """获取默认TextConfig（兼容性函数）"""
    return TextConfig(
        input_path="/Volumes/home/raw_data/note_info_parquet",
        output_path="/Volumes/home/raw_data/text_features_parquet"
    )