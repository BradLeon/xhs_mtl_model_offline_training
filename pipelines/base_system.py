#!/usr/bin/env python3
"""
Pipeline系统管理抽象层
统一的监控、资源管理和清理功能
合并text_system和multimodal中的系统管理功能
"""

import os
import gc
import time
import logging
import threading
import resource
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

import psutil

from .base_config import BasePipelineConfig

logger = logging.getLogger(__name__)


class BaseSystemMonitor(ABC):
    """系统监控器抽象基类
    
    统一text_system.SystemMonitor和multimodal的内存监控功能
    """
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.current_stage = "初始化"
        self.stage_start_time = time.time()
        self.batch_times_window = deque(maxlen=getattr(config, 'batch_times_window_size', 10))
        self.total_records_processed = 0
        self.total_output_size_mb = 0
        self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 启动监控线程
        self._start_monitor_thread()
    
    def _start_monitor_thread(self):
        """启动后台监控线程"""
        def monitor_loop():
            while not self.stop_monitor.is_set():
                # 等待指定间隔
                for _ in range(self.config.monitor_interval_seconds * 10):
                    if self.stop_monitor.is_set():
                        break
                    time.sleep(0.1)
                
                if not self.stop_monitor.is_set():
                    self._print_status_summary()
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"✅ 后台监控线程已启动（每{self.config.monitor_interval_seconds}秒更新状态）")
    
    def _print_status_summary(self):
        """打印当前执行状态摘要"""
        try:
            # 获取系统资源
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.logger.info("\n" + "="*60)
            self.logger.info(f"[定时状态报告] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"当前阶段: {self.current_stage}")
            self.logger.info(f"阶段已执行: {time.time() - self.stage_start_time:.1f}秒")
            
            # 性能统计
            if self.batch_times_window:
                avg_batch_time = sum(self.batch_times_window) / len(self.batch_times_window)
                self.logger.info(f"最近{len(self.batch_times_window)}批平均耗时: {avg_batch_time:.1f}秒")
                
                # 趋势分析
                if len(self.batch_times_window) >= 5:
                    recent_3 = list(self.batch_times_window)[-3:]
                    earlier_3 = list(self.batch_times_window)[-6:-3]
                    recent_avg = sum(recent_3) / len(recent_3)
                    earlier_avg = sum(earlier_3) / len(earlier_3)
                    
                    if recent_avg > earlier_avg * 1.2:
                        self.logger.warning("⚠️ 处理速度变慢趋势")
                    elif recent_avg < earlier_avg * 0.8:
                        self.logger.info("✅ 处理速度加快趋势")
                    else:
                        self.logger.info("➡️ 处理速度稳定")
            
            # 系统资源
            self.logger.info(f"内存使用: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent:.1f}%)")
            self.logger.info(f"CPU使用: {cpu_percent:.1f}%")
            
            # 自定义监控信息
            self._print_custom_metrics()
            
            # 磁盘空间监控
            self._print_disk_usage()
            
            if self.total_records_processed > 0:
                self.logger.info(f"累计处理记录: {self.total_records_processed:,}")
            if self.total_output_size_mb > 0:
                self.logger.info(f"输出文件大小: {self.total_output_size_mb:.2f} MB ({self.total_output_size_mb/1024:.2f} GB)")
            
            self.logger.info("="*60 + "\n")
        except Exception as e:
            self.logger.error(f"状态报告生成失败: {e}")
    
    def _print_disk_usage(self):
        """打印磁盘使用情况"""
        try:
            # 检查常见的磁盘挂载点
            disk_paths = ['/Volumes/home', '/', self.config.output_path]
            
            for path in disk_paths:
                if os.path.exists(path):
                    disk = psutil.disk_usage(path)
                    disk_name = path if path != '/' else '系统磁盘'
                    self.logger.info(f"{disk_name}使用: {disk.used/(1024**3):.1f}GB / {disk.total/(1024**3):.1f}GB ({disk.percent:.1f}%)")
                    
                    if disk.percent > 90:
                        self.logger.warning(f"⚠️ {disk_name}空间即将耗尽！")
                    break
        except Exception as e:
            self.logger.debug(f"磁盘使用监控失败: {e}")
    
    @abstractmethod
    def _print_custom_metrics(self):
        """打印自定义监控指标，子类实现"""
        pass
    
    def update_stage(self, stage_name: str):
        """更新当前执行阶段"""
        prev_stage = self.current_stage
        prev_time = time.time() - self.stage_start_time
        if prev_stage != "初始化":
            self.logger.info(f"[STAGE_END] {prev_stage} 完成，耗时: {prev_time:.2f}秒")
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.logger.info(f"[STAGE_START] 开始执行: {stage_name}")
    
    def add_batch_time(self, batch_time: float):
        """添加批次处理时间"""
        self.batch_times_window.append(batch_time)
    
    def update_records_processed(self, count: int):
        """更新处理记录数"""
        self.total_records_processed += count
    
    def update_output_size(self, size_mb: float):
        """更新输出文件大小"""
        self.total_output_size_mb = size_mb
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况（MB）"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss / 1024 / 1024,  # 物理内存
                'vms': memory_info.vms / 1024 / 1024,  # 虚拟内存
                'percent': process.memory_percent()     # 内存使用百分比
            }
        except Exception as e:
            self.logger.warning(f"获取内存使用情况失败: {e}")
            return {'rss': 0, 'vms': 0, 'percent': 0}
    
    def log_memory_usage(self, stage: str, batch_id: Optional[int] = None):
        """记录内存使用情况"""
        memory = self.get_current_memory_usage()
        batch_info = f" (batch {batch_id})" if batch_id is not None else ""
        self.logger.info(f"🧠 Memory {stage}{batch_info}: RSS={memory['rss']:.1f}MB, VMS={memory['vms']:.1f}MB, %={memory['percent']:.1f}%")
    
    def cleanup(self):
        """清理监控资源"""
        self.logger.info("开始清理监控资源...")
        
        # 停止监控线程
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitor.set()
            self.monitor_thread.join(timeout=5)
            self.logger.info("监控线程已停止")


class BaseResourceManager:
    """资源管理器抽象基类
    
    统一内存清理、临时文件清理、磁盘空间检查等功能
    """
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def increase_file_limit(self, target_limit: int = 10240):
        """增加系统文件描述符限制"""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.logger.info(f"当前文件描述符限制: soft={soft}, hard={hard}")
            
            new_limit = min(target_limit, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
            
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.logger.info(f"文件描述符限制已更新: soft={soft}, hard={hard}")
            
            os.system(f'ulimit -n {target_limit}')
            
        except Exception as e:
            self.logger.warning(f"无法增加文件描述符限制: {e}")
            self.logger.warning(f"请尝试在终端执行: ulimit -n {target_limit}")
    
    def cleanup_memory(self):
        """清理内存"""
        try:
            gc.collect()
            self.logger.debug("内存垃圾回收完成")
        except Exception as e:
            self.logger.warning(f"内存清理失败: {e}")
    
    def cleanup_temp_files(self, additional_patterns: list = None):
        """清理临时文件"""
        try:
            # 基础临时目录列表
            temp_patterns = [
                "/tmp/spark-*",
                "/var/tmp/spark-*",
                "/tmp/py*"
            ]
            
            # 添加额外的清理模式
            if additional_patterns:
                temp_patterns.extend(additional_patterns)
            
            # 添加配置中的临时目录
            if hasattr(self.config, 'spark_tmp_dir'):
                temp_patterns.append(f"{self.config.spark_tmp_dir}/*")
            
            for pattern in temp_patterns:
                try:
                    os.system(f"rm -rf {pattern} 2>/dev/null")
                except Exception as e:
                    self.logger.debug(f"清理临时文件 {pattern} 失败: {e}")
            
            self.logger.debug("临时文件清理完成")
        except Exception as e:
            self.logger.warning(f"临时文件清理失败: {e}")
    
    def get_disk_usage(self, path: str) -> Dict[str, float]:
        """获取磁盘使用情况"""
        try:
            usage = psutil.disk_usage(path)
            return {
                'total_gb': usage.total / (1024**3),
                'used_gb': usage.used / (1024**3),
                'free_gb': usage.free / (1024**3),
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            self.logger.warning(f"无法获取磁盘使用情况 {path}: {e}")
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'percent': 0}
    
    def check_available_space(self, required_gb: float = 10.0, path: Optional[str] = None) -> bool:
        """检查可用磁盘空间是否充足"""
        try:
            check_path = path or self.config.output_path
            usage = self.get_disk_usage(check_path)
            free_gb = usage['free_gb']
            
            if free_gb < required_gb:
                self.logger.warning(f"磁盘空间不足: 需要{required_gb}GB，可用{free_gb:.1f}GB")
                return False
            
            self.logger.info(f"磁盘空间检查通过: 需要{required_gb}GB，可用{free_gb:.1f}GB")
            return True
        except Exception as e:
            self.logger.warning(f"检查磁盘空间失败: {e}")
            return True  # 检查失败时假设空间充足
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息摘要"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            
            return {
                'cpu_count': cpu_count,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'max_workers': self.config.max_workers,
                'disk_usage': self.get_disk_usage(self.config.output_path)
            }
        except Exception as e:
            self.logger.warning(f"获取系统信息失败: {e}")
            return {}


class SparkSystemMonitor(BaseSystemMonitor):
    """Spark专用系统监控器"""
    
    def _print_custom_metrics(self):
        """打印Spark相关的监控指标"""
        if hasattr(self.config, 'driver_memory_gb'):
            self.logger.info(f"Spark驱动内存: {self.config.driver_memory_gb}GB")
        if hasattr(self.config, 'executor_cores'):
            self.logger.info(f"Spark执行器核心: {self.config.executor_cores}")


class MultimodalSystemMonitor(BaseSystemMonitor):
    """多模态专用系统监控器"""
    
    def _print_custom_metrics(self):
        """打印多模态相关的监控指标"""
        if hasattr(self.config, 'model_name'):
            self.logger.info(f"CLIP模型: {self.config.model_name}")
        if hasattr(self.config, 'gpu_batch_size'):
            self.logger.info(f"GPU批次大小: {self.config.gpu_batch_size}")


def create_system_monitor(config: BasePipelineConfig) -> BaseSystemMonitor:
    """工厂函数：根据配置类型创建相应的系统监控器"""
    pipeline_name = config.get_pipeline_name()
    
    if 'text' in pipeline_name or hasattr(config, 'spark_tmp_dir'):
        return SparkSystemMonitor(config)
    elif 'multimodal' in pipeline_name or hasattr(config, 'model_name'):
        return MultimodalSystemMonitor(config)
    else:
        # 默认监控器
        class DefaultSystemMonitor(BaseSystemMonitor):
            def _print_custom_metrics(self):
                pass
        return DefaultSystemMonitor(config)


def create_resource_manager(config: BasePipelineConfig) -> BaseResourceManager:
    """工厂函数：创建资源管理器"""
    return BaseResourceManager(config)