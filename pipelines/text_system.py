#!/usr/bin/env python3
"""
文本pipeline系统管理模块（重构版）
继承自base_system的抽象类，提供文本pipeline专用的系统管理功能
从run_local_text_pipeline_batch_ssd.py迁移而来
"""

import os
import logging
import resource
from typing import Optional

from pyspark.sql import SparkSession

from .text_config import TextConfig
from .base_system import BaseSystemMonitor, BaseResourceManager

logger = logging.getLogger(__name__)


class TextSystemMonitor(BaseSystemMonitor):
    """文本pipeline专用系统监控器
    
    继承自BaseSystemMonitor，添加文本处理特定的监控功能
    """
    
    def _print_custom_metrics(self):
        """打印文本pipeline特定的监控指标"""
        if hasattr(self.config, 'batch_times_window_size'):
            self.logger.info(f"批次时间窗口大小: {self.config.batch_times_window_size}")
        if hasattr(self.config, 'min_impression'):
            self.logger.info(f"最小曝光阈值: {self.config.min_impression}")
        if hasattr(self.config, 'parquet_compression'):
            self.logger.info(f"Parquet压缩: {self.config.parquet_compression}")


# 兼容性别名
SystemMonitor = TextSystemMonitor


class SparkSessionManager:
    """Spark会话管理器（兼容性包装）
    
    提供文本pipeline专用的Spark会话创建和管理功能
    """
    
    @staticmethod
    def increase_file_limit(target_limit: int = 10240):
        """增加系统文件描述符限制"""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"当前文件描述符限制: soft={soft}, hard={hard}")
            
            new_limit = min(target_limit, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
            
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"文件描述符限制已更新: soft={soft}, hard={hard}")
            
            os.system(f'ulimit -n {target_limit}')
            
        except Exception as e:
            logger.warning(f"无法增加文件描述符限制: {e}")
            logger.warning(f"请尝试在终端执行: ulimit -n {target_limit}")
    
    @staticmethod
    def create_optimized_spark_session(config: TextConfig) -> SparkSession:
        """创建优化的本地Spark会话"""
        logger.info("开始创建优化的Spark会话...")
        
        # 首先增加文件描述符限制
        if hasattr(config, 'file_descriptor_limit'):
            SparkSessionManager.increase_file_limit(config.file_descriptor_limit)
        else:
            SparkSessionManager.increase_file_limit(10240)
        
        # 设置环境和创建目录
        config.setup_environment()
        
        # 获取Spark配置
        spark_configs = config.get_spark_configs()
        
        # 创建SparkSession Builder
        builder = SparkSession.builder
        
        # 应用所有配置
        for key, value in spark_configs.items():
            if key == "spark.app.name":
                builder = builder.appName(value)
            elif key == "spark.master":
                builder = builder.master(value)
            else:
                builder = builder.config(key, value)
        
        # 创建会话
        spark = builder.getOrCreate()
        
        # 设置日志级别
        spark.sparkContext.setLogLevel("WARN")
        
        # 打印关键配置信息
        logger.info("="*60)
        logger.info("Spark配置（优化SSD存储）:")
        logger.info(f"  应用名称: {config.app_name}")
        logger.info(f"  executor.cores: {config.executor_cores} (高并发)")
        logger.info(f"  driver.memory: {config.driver_memory_gb}g")
        logger.info(f"  shuffle.partitions: {config.shuffle_partitions}")
        logger.info(f"  使用SSD: {config.spark_tmp_dir}")
        logger.info(f"  cleaner.periodicGC.interval: {config.gc_interval}")
        if hasattr(config, 'batch_start') and hasattr(config, 'batch_end'):
            logger.info(f"  批次范围: {config.batch_start} - {config.batch_end}")
        logger.info("="*60)
        
        return spark
    
    @staticmethod
    def cleanup_spark_session(spark: SparkSession):
        """安全清理Spark会话"""
        if spark:
            try:
                logger.info("正在关闭Spark会话...")
                spark.stop()
                logger.info("Spark会话已安全关闭")
            except Exception as e:
                logger.warning(f"关闭Spark会话时出现警告: {e}")


class TextResourceManager(BaseResourceManager):
    """文本pipeline专用资源管理器
    
    继承自BaseResourceManager，添加文本处理特定的资源管理功能
    """
    
    def cleanup_temp_files(self, additional_patterns: list = None):
        """清理文本pipeline相关的临时文件"""
        # 文本pipeline特定的临时文件模式
        text_patterns = [
            f"{self.config.spark_tmp_dir}/*",
        ]
        
        if additional_patterns:
            text_patterns.extend(additional_patterns)
        
        # 调用基类方法
        super().cleanup_temp_files(text_patterns)


# 兼容性别名
ResourceManager = TextResourceManager