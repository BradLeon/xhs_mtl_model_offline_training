#!/usr/bin/env python3
"""
Pipeline基础配置抽象层
提供所有pipeline的通用配置接口和基础功能
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class BasePipelineConfig(ABC):
    """Pipeline配置抽象基类
    
    定义所有pipeline的通用配置接口，子类需要实现特定的配置逻辑
    """
    
    # 通用路径配置
    input_path: str
    output_path: str
    
    # 通用批次配置
    batch_size: int = 1000
    checkpoint_interval: int = 5
    resume: bool = False
    
    # 通用系统配置
    max_workers: int = 2
    monitor_interval_seconds: int = 30
    
    # 通用性能配置
    memory_limit_gb: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理，子类可以重写"""
        self._setup_dynamic_config()
        self.validate()
        
    def _setup_dynamic_config(self):
        """设置动态配置，子类可以重写"""
        pass
    
    @abstractmethod
    def validate(self) -> None:
        """验证配置的有效性，子类必须实现"""
        pass
    
    @abstractmethod 
    def get_output_path(self) -> str:
        """获取实际输出路径，子类必须实现"""
        pass
    
    @abstractmethod
    def setup_environment(self) -> None:
        """设置环境变量和目录，子类必须实现"""
        pass
    
    def get_pipeline_name(self) -> str:
        """获取pipeline名称，默认使用类名"""
        return self.__class__.__name__.replace('Config', '').lower()
    
    def get_log_file_path(self, custom_name: Optional[str] = None) -> str:
        """获取日志文件路径"""
        if custom_name:
            log_name = custom_name
        else:
            pipeline_name = self.get_pipeline_name()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"{pipeline_name}_pipeline_{timestamp}.log"
        
        # 确保logs目录存在
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        return os.path.join(log_dir, log_name)
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        return {
            'pipeline_type': self.get_pipeline_name(),
            'input_path': self.input_path,
            'output_path': self.output_path,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'resume_enabled': self.resume,
            'checkpoint_interval': self.checkpoint_interval
        }


@dataclass
class BatchProcessingMixin:
    """批处理配置混入类"""
    batch_start: int = 0
    batch_end: int = 100
    
    def get_batch_info(self) -> str:
        """获取批次信息字符串"""
        return f"batch_{self.batch_start:05d}_{self.batch_end:05d}"
    
    def validate_batch_config(self) -> None:
        """验证批次配置"""
        if self.batch_start < 0:
            raise ValueError("batch_start must be non-negative")
        if self.batch_end < self.batch_start:
            raise ValueError("batch_end must be >= batch_start")


@dataclass  
class SparkConfigMixin:
    """Spark配置混入类"""
    # Spark基础配置
    app_name: str = "XHS_Pipeline"
    driver_memory_gb: Optional[int] = None
    executor_cores: Optional[int] = None
    shuffle_partitions: int = 200
    
    # Spark存储配置
    spark_tmp_dir: str = "/Volumes/ORICO/spark_tmp"
    spark_warehouse_dir: str = "/Volumes/ORICO/spark_warehouse" 
    spark_checkpoint_dir: str = "/Volumes/ORICO/spark_checkpoint"
    
    # Spark优化配置
    memory_fraction: float = 0.7
    storage_fraction: float = 0.2
    gc_interval: str = "5min"
    
    def setup_spark_environment(self) -> None:
        """设置Spark环境变量"""
        os.environ['SPARK_LOCAL_DIRS'] = self.spark_tmp_dir
        os.environ['TMPDIR'] = self.spark_tmp_dir
        
        # 创建Spark目录
        os.makedirs(self.spark_tmp_dir, exist_ok=True)
        os.makedirs(self.spark_warehouse_dir, exist_ok=True)
        os.makedirs(self.spark_checkpoint_dir, exist_ok=True)


@dataclass
class MLConfigMixin:
    """机器学习配置混入类"""
    # ML模型配置
    model_name: str = "default"
    target_dim: int = 512
    
    # 数据过滤配置
    min_impression_threshold: int = 500
    
    # GPU配置
    gpu_batch_size: int = 8
    enable_gpu: bool = True


def create_base_config_from_args(config_class, args) -> BasePipelineConfig:
    """从命令行参数创建配置的通用函数
    
    Args:
        config_class: 配置类
        args: 命令行参数对象
        
    Returns:
        配置实例
    """
    # 基础参数映射
    config_kwargs = {}
    
    # 通用参数
    if hasattr(args, 'input') and args.input:
        config_kwargs['input_path'] = args.input
    if hasattr(args, 'output') and args.output:
        config_kwargs['output_path'] = args.output
    if hasattr(args, 'batch_size') and args.batch_size:
        config_kwargs['batch_size'] = args.batch_size
    if hasattr(args, 'max_workers') and args.max_workers:
        config_kwargs['max_workers'] = args.max_workers
    if hasattr(args, 'resume') and args.resume:
        config_kwargs['resume'] = args.resume
    if hasattr(args, 'checkpoint_interval') and args.checkpoint_interval:
        config_kwargs['checkpoint_interval'] = args.checkpoint_interval
    
    # 批处理参数
    if hasattr(args, 'batch_start') and args.batch_start is not None:
        config_kwargs['batch_start'] = args.batch_start
    if hasattr(args, 'batch_end') and args.batch_end is not None:
        config_kwargs['batch_end'] = args.batch_end
    
    # Spark参数
    if hasattr(args, 'driver_memory') and args.driver_memory:
        config_kwargs['driver_memory_gb'] = args.driver_memory
    if hasattr(args, 'executor_cores') and args.executor_cores:
        config_kwargs['executor_cores'] = args.executor_cores
    
    # ML参数 - 只有继承了MLConfigMixin的配置类才支持
    if hasattr(config_class, '__annotations__') and 'model_name' in config_class.__annotations__:
        if hasattr(args, 'model_name') and args.model_name:
            config_kwargs['model_name'] = args.model_name
        if hasattr(args, 'gpu_batch_size') and args.gpu_batch_size:
            config_kwargs['gpu_batch_size'] = args.gpu_batch_size
        if hasattr(args, 'min_impression') and args.min_impression is not None:
            config_kwargs['min_impression_threshold'] = args.min_impression
        if hasattr(args, 'min_impression_threshold') and args.min_impression_threshold is not None:
            config_kwargs['min_impression_threshold'] = args.min_impression_threshold
    
    # 创建配置实例
    return config_class(**config_kwargs)


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_paths(config: BasePipelineConfig) -> None:
        """验证路径配置"""
        if not config.input_path:
            raise ValueError("input_path cannot be empty")
        if not config.output_path:
            raise ValueError("output_path cannot be empty")
            
        # 检查输入路径是否存在
        if not os.path.exists(config.input_path):
            raise ValueError(f"Input path does not exist: {config.input_path}")
    
    @staticmethod
    def validate_performance_config(config: BasePipelineConfig) -> None:
        """验证性能配置"""
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if config.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if config.monitor_interval_seconds <= 0:
            raise ValueError("monitor_interval_seconds must be positive")
    
    @staticmethod
    def validate_batch_config(config) -> None:
        """验证批处理配置（如果配置有批处理混入）"""
        if hasattr(config, 'batch_start') and hasattr(config, 'batch_end'):
            if config.batch_start < 0:
                raise ValueError("batch_start must be non-negative")
            if config.batch_end < config.batch_start:
                raise ValueError("batch_end must be >= batch_start")
    
    @staticmethod
    def validate_spark_config(config) -> None:
        """验证Spark配置（如果配置有Spark混入）"""
        if hasattr(config, 'driver_memory_gb') and config.driver_memory_gb:
            if config.driver_memory_gb <= 0:
                raise ValueError("driver_memory_gb must be positive")
        if hasattr(config, 'executor_cores') and config.executor_cores:
            if config.executor_cores <= 0:
                raise ValueError("executor_cores must be positive")
        if hasattr(config, 'shuffle_partitions'):
            if config.shuffle_partitions <= 0:
                raise ValueError("shuffle_partitions must be positive")


class ConfigFactory:
    """配置工厂类，用于创建不同类型的配置"""
    
    _config_registry = {}
    
    @classmethod
    def register_config(cls, name: str, config_class):
        """注册配置类"""
        cls._config_registry[name] = config_class
    
    @classmethod
    def create_config(cls, name: str, **kwargs) -> BasePipelineConfig:
        """根据名称创建配置"""
        if name not in cls._config_registry:
            raise ValueError(f"Unknown config type: {name}")
        
        config_class = cls._config_registry[name]
        return config_class(**kwargs)
    
    @classmethod
    def get_available_configs(cls) -> list:
        """获取可用的配置类型"""
        return list(cls._config_registry.keys())