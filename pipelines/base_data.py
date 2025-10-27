#!/usr/bin/env python3
"""
Pipeline数据处理抽象层
统一的数据读取、写入和checkpoint管理功能
合并text_data和multimodal_data中的数据处理功能
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_config import BasePipelineConfig

logger = logging.getLogger(__name__)


class BaseCheckpointManager:
    """Checkpoint管理器抽象基类
    
    统一text和multimodal的checkpoint功能
    """
    
    def __init__(self, checkpoint_dir: str = ".checkpoints", checkpoint_prefix: str = "progress"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_file = self.checkpoint_dir / f"{checkpoint_prefix}.json"
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_progress(self, progress: Dict[str, Any], version: str = "1.0"):
        """保存进度信息"""
        try:
            progress['version'] = version
            progress['last_update'] = datetime.now().isoformat()
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            self.logger.debug(f"Checkpoint saved: {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"保存checkpoint失败: {e}")
    
    def load_progress(self, required_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """加载进度信息"""
        try:
            if not self.checkpoint_file.exists():
                return None
            
            with open(self.checkpoint_file, 'r') as f:
                progress = json.load(f)
            
            # 版本兼容性检查
            if required_version:
                version = progress.get('version', '1.0')
                if not version.startswith(required_version.split('.')[0]):
                    self.logger.warning(f"Checkpoint版本不匹配: {version} vs {required_version}")
                    return None
            
            self.logger.info(f"Checkpoint loaded: {self.checkpoint_file}")
            return progress
            
        except Exception as e:
            self.logger.error(f"加载checkpoint失败: {e}")
            return None
    
    def clear_progress(self):
        """清除进度信息"""
        try:
            if self.checkpoint_file.exists():
                os.remove(self.checkpoint_file)
                self.logger.info("Checkpoint已清除")
        except Exception as e:
            self.logger.error(f"清除checkpoint失败: {e}")
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.pkl"):
        """保存元数据（如文件映射）"""
        try:
            metadata_file = self.checkpoint_dir / filename
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            self.logger.debug(f"元数据已保存: {metadata_file}")
        except Exception as e:
            self.logger.error(f"保存元数据失败: {e}")
    
    def load_metadata(self, filename: str = "metadata.pkl") -> Optional[Dict[str, Any]]:
        """加载元数据"""
        try:
            metadata_file = self.checkpoint_dir / filename
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.logger.debug(f"元数据已加载: {metadata_file}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"加载元数据失败: {e}")
            return None


class BaseDataReader(ABC):
    """数据读取器抽象基类"""
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def read_data(self) -> Any:
        """读取数据，子类必须实现"""
        pass
    
    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息，子类必须实现"""
        pass
    
    def validate_input_path(self) -> bool:
        """验证输入路径"""
        if not os.path.exists(self.config.input_path):
            self.logger.error(f"输入路径不存在: {self.config.input_path}")
            return False
        return True


class BaseDataWriter(ABC):
    """数据写入器抽象基类"""
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
    
    @abstractmethod
    def write_data(self, data: Any, output_path: Optional[str] = None) -> str:
        """写入数据，子类必须实现"""
        pass
    
    @abstractmethod
    def verify_output(self, output_path: str) -> Dict[str, Any]:
        """验证输出，子类必须实现"""
        pass
    
    def calculate_file_size(self, file_path: str) -> float:
        """计算文件大小（MB）"""
        try:
            if os.path.isfile(file_path):
                return os.path.getsize(file_path) / (1024 * 1024)
            elif os.path.isdir(file_path):
                total_size = 0
                for root, _, files in os.walk(file_path):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
                return total_size / (1024 * 1024)
            else:
                return 0.0
        except Exception as e:
            self.logger.warning(f"计算文件大小失败 {file_path}: {e}")
            return 0.0
    
    def get_timestamp_suffix(self) -> str:
        """获取时间戳后缀"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')


class BatchDataMixin:
    """批次数据处理混入类"""
    
    def generate_batch_output_path(self, base_path: str, batch_info: str, 
                                  include_timestamp: bool = True) -> str:
        """生成批次输出路径"""
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{base_path}/{batch_info}/dt={timestamp}"
        else:
            return f"{base_path}/{batch_info}"
    
    def extract_batch_info_from_config(self, config) -> str:
        """从配置中提取批次信息"""
        if hasattr(config, 'batch_start') and hasattr(config, 'batch_end'):
            return f"batch_{config.batch_start:05d}_{config.batch_end:05d}"
        else:
            return "batch_default"


class FilePatternMatcher:
    """文件模式匹配器"""
    
    @staticmethod
    def find_files_by_patterns(base_path: str, patterns: List[str]) -> List[str]:
        """根据模式列表查找文件"""
        import glob
        
        all_files = []
        for pattern in patterns:
            full_pattern = os.path.join(base_path, pattern)
            matched_files = glob.glob(full_pattern)
            all_files.extend(matched_files)
        
        # 去重并排序
        return sorted(set(all_files))
    
    @staticmethod
    def generate_batch_patterns(batch_start: int, batch_end: int, 
                              extensions: List[str] = None) -> List[str]:
        """生成批次文件模式"""
        if extensions is None:
            extensions = ["*.parquet", "*.snappy.parquet"]
        
        patterns = []
        for i in range(batch_start, batch_end + 1):
            for ext in extensions:
                patterns.append(f"part-{i:05d}{ext}")
        
        return patterns


class DataProcessor(ABC):
    """数据处理器抽象基类"""
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process_data(self, data: Any) -> Any:
        """处理数据，子类必须实现"""
        pass
    
    def apply_common_filters(self, data: Any) -> Any:
        """应用通用过滤器"""
        # 子类可以重写此方法添加通用的数据过滤逻辑
        return data
    
    def validate_processed_data(self, data: Any) -> bool:
        """验证处理后的数据"""
        # 子类可以重写此方法添加数据验证逻辑
        return True


class DataPipelineStats:
    """数据处理统计信息"""
    
    def __init__(self):
        self.stats = {
            'start_time': datetime.now(),
            'rows_processed': 0,
            'files_processed': 0,
            'total_input_size_mb': 0,
            'total_output_size_mb': 0,
            'processing_time_seconds': 0,
            'errors': []
        }
    
    def update_input_stats(self, rows: int, size_mb: float, files: int = 1):
        """更新输入统计"""
        self.stats['rows_processed'] += rows
        self.stats['total_input_size_mb'] += size_mb
        self.stats['files_processed'] += files
    
    def update_output_stats(self, size_mb: float):
        """更新输出统计"""
        self.stats['total_output_size_mb'] += size_mb
    
    def add_error(self, error: str):
        """添加错误记录"""
        self.stats['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
    
    def finalize_stats(self):
        """完成统计"""
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time_seconds'] = (
            self.stats['end_time'] - self.stats['start_time']
        ).total_seconds()
        
        # 计算处理速度
        if self.stats['processing_time_seconds'] > 0:
            self.stats['processing_rate_rows_per_sec'] = (
                self.stats['rows_processed'] / self.stats['processing_time_seconds']
            )
        
        return self.stats.copy()
    
    def get_summary(self) -> str:
        """获取统计摘要"""
        stats = self.finalize_stats()
        return (
            f"处理完成: {stats['rows_processed']:,}行, "
            f"{stats['files_processed']}文件, "
            f"耗时{stats['processing_time_seconds']:.1f}秒, "
            f"输出{stats['total_output_size_mb']:.1f}MB"
        )


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def retry_operation(self, operation, *args, **kwargs):
        """重试操作"""
        import time
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    self.logger.warning(f"操作失败，第{attempt + 1}次重试: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"操作最终失败，已重试{self.max_retries}次: {e}")
        
        raise last_exception
    
    def handle_recoverable_error(self, error: Exception, context: str = "") -> bool:
        """处理可恢复的错误"""
        error_msg = str(error)
        
        # 定义可恢复的错误模式
        recoverable_patterns = [
            "timeout",
            "connection",
            "temporary",
            "retry",
            "network"
        ]
        
        is_recoverable = any(pattern in error_msg.lower() for pattern in recoverable_patterns)
        
        if is_recoverable:
            self.logger.warning(f"检测到可恢复错误 {context}: {error}")
        else:
            self.logger.error(f"检测到不可恢复错误 {context}: {error}")
        
        return is_recoverable


# 工厂函数
def create_checkpoint_manager(pipeline_name: str, checkpoint_dir: str = ".checkpoints") -> BaseCheckpointManager:
    """创建checkpoint管理器"""
    return BaseCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=f"progress_{pipeline_name}"
    )


def create_data_stats() -> DataPipelineStats:
    """创建数据处理统计对象"""
    return DataPipelineStats()


def create_error_handler(max_retries: int = 3) -> ErrorHandler:
    """创建错误处理器"""
    return ErrorHandler(max_retries=max_retries)