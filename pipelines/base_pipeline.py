#!/usr/bin/env python3
"""
Pipeline统一基类
提供所有pipeline的通用执行框架和接口
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional

from .base_config import BasePipelineConfig
from .base_system import BaseSystemMonitor, BaseResourceManager, create_system_monitor, create_resource_manager
from .base_data import DataPipelineStats, ErrorHandler, create_data_stats, create_error_handler

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    """Pipeline统一基类
    
    提供统一的执行流程模板，子类只需实现特定的业务逻辑
    """
    
    def __init__(self, config: BasePipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化组件
        self.monitor: Optional[BaseSystemMonitor] = None
        self.resource_manager: Optional[BaseResourceManager] = None
        self.stats: Optional[DataPipelineStats] = None
        self.error_handler: Optional[ErrorHandler] = None
        
        # 执行状态
        self.is_initialized = False
        self.is_running = False
        
        # 执行结果
        self.execution_result = {
            'status': 'initialized',
            'start_time': datetime.now(),
            'config_info': self.config.get_stats_dict()
        }
    
    def run(self) -> Dict[str, Any]:
        """统一的执行流程模板"""
        self.logger.info("="*60)
        self.logger.info(f"开始执行 {self.__class__.__name__}")
        self.logger.info(f"配置: {self.config.get_pipeline_name()}")
        self.logger.info(f"时间: {datetime.now()}")
        self.logger.info("="*60)
        
        try:
            # 标记开始执行
            self.is_running = True
            self.execution_result['status'] = 'running'
            
            # 第1步：初始化所有组件
            self._initialize_components()
            
            # 第2步：预检查
            self._perform_pre_checks()
            
            # 第3步：执行核心pipeline逻辑
            self.monitor.update_stage("执行核心Pipeline逻辑")
            start_time = time.time()
            
            result = self._execute_pipeline()
            
            execution_time = time.time() - start_time
            
            # 第4步：后处理和结果整理
            self._finalize_results(result, execution_time)
            
            # 标记成功完成
            self.execution_result['status'] = 'completed'
            self.monitor.update_stage("Pipeline执行完成")
            
            # 打印完成摘要
            self._print_completion_summary()
            
            return self.execution_result
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            self.execution_result.update({
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            })
            
            raise
        finally:
            self.is_running = False
            self._cleanup_resources()
    
    def _initialize_components(self):
        """初始化所有组件"""
        self.logger.info("初始化Pipeline组件...")
        
        # 创建系统监控器
        self.monitor = create_system_monitor(self.config)
        self.monitor.update_stage("初始化组件")
        
        # 创建资源管理器
        self.resource_manager = create_resource_manager(self.config)
        
        # 创建统计对象
        self.stats = create_data_stats()
        
        # 创建错误处理器
        self.error_handler = create_error_handler(
            max_retries=getattr(self.config, 'max_retries', 3)
        )
        
        # 调用子类的初始化逻辑
        self._initialize_custom_components()
        
        self.is_initialized = True
        self.logger.info("所有组件初始化完成")
    
    def _perform_pre_checks(self):
        """执行预检查"""
        self.monitor.update_stage("执行预检查")
        
        # 检查磁盘空间
        if not self.resource_manager.check_available_space(10.0):
            raise RuntimeError("磁盘空间不足，无法继续处理")
        
        # 调用子类的预检查逻辑
        self._perform_custom_pre_checks()
        
        self.logger.info("预检查完成")
    
    def _finalize_results(self, pipeline_result: Dict[str, Any], execution_time: float):
        """完成结果整理"""
        self.monitor.update_stage("整理执行结果")
        
        # 完成统计
        final_stats = self.stats.finalize_stats()
        
        # 更新执行结果
        self.execution_result.update({
            'end_time': datetime.now(),
            'execution_time_seconds': execution_time,
            'execution_time_hours': execution_time / 3600,
            'pipeline_result': pipeline_result,
            'statistics': final_stats,
            'system_info': self.resource_manager.get_system_info()
        })
        
        # 更新监控统计
        self.monitor.update_records_processed(final_stats['rows_processed'])
        self.monitor.update_output_size(final_stats['total_output_size_mb'])
        self.monitor.add_batch_time(execution_time)
        
        self.logger.info("结果整理完成")
    
    def _print_completion_summary(self):
        """打印完成摘要"""
        result = self.execution_result
        stats = result.get('statistics', {})
        
        self.logger.info("="*60)
        self.logger.info(f"✅ {self.__class__.__name__} 执行完成!")
        self.logger.info(f"总耗时: {result.get('execution_time_hours', 0):.2f} 小时")
        self.logger.info(f"处理行数: {stats.get('rows_processed', 'N/A'):,}")
        self.logger.info(f"处理文件: {stats.get('files_processed', 'N/A')}")
        
        output_size = stats.get('total_output_size_mb', 0)
        if output_size > 0:
            self.logger.info(f"输出大小: {output_size:.2f} MB ({output_size/1024:.2f} GB)")
        
        # 处理速度
        processing_rate = stats.get('processing_rate_rows_per_sec', 0)
        if processing_rate > 0:
            self.logger.info(f"处理速度: {processing_rate:.1f} 行/秒")
        
        # 输出路径
        pipeline_result = result.get('pipeline_result', {})
        output_path = pipeline_result.get('output_path')
        if output_path:
            self.logger.info(f"输出路径: {output_path}")
        
        self.logger.info("="*60)
    
    def _cleanup_resources(self):
        """清理所有资源"""
        self.logger.info("开始清理Pipeline资源...")
        
        try:
            # 清理监控器
            if self.monitor:
                self.monitor.cleanup()
            
            # 清理资源管理器
            if self.resource_manager:
                self.resource_manager.cleanup_memory()
                self.resource_manager.cleanup_temp_files()
            
            # 调用子类的清理逻辑
            self._cleanup_custom_resources()
            
            self.logger.info("Pipeline资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"资源清理时出现警告: {e}")
    
    # 抽象方法，子类必须实现
    @abstractmethod
    def _execute_pipeline(self) -> Dict[str, Any]:
        """执行核心pipeline逻辑，子类必须实现"""
        pass
    
    # 可选的钩子方法，子类可以重写
    def _initialize_custom_components(self):
        """初始化自定义组件，子类可以重写"""
        pass
    
    def _perform_custom_pre_checks(self):
        """执行自定义预检查，子类可以重写"""
        pass
    
    def _cleanup_custom_resources(self):
        """清理自定义资源，子类可以重写"""
        pass
    
    # 兼容性方法，保持与原有接口的兼容
    def get_output_prefix(self) -> str:
        """获取输出前缀，为了兼容性保留"""
        return f"{self.config.get_pipeline_name()}_features"
    
    def read_data(self):
        """读取数据，为了兼容性保留，子类可以重写"""
        self.logger.warning("read_data() method called but not implemented")
        return None
    
    def process(self):
        """处理数据，为了兼容性保留"""
        return self.run()


class SparkPipelineBase(BasePipeline):
    """Spark Pipeline基类
    
    为使用Spark的pipeline提供特定的初始化和清理逻辑
    """
    
    def __init__(self, config: BasePipelineConfig):
        super().__init__(config)
        self.spark = None
    
    def _initialize_custom_components(self):
        """初始化Spark组件"""
        # 子类需要实现Spark会话的创建
        self._create_spark_session()
    
    def _cleanup_custom_resources(self):
        """清理Spark资源"""
        if self.spark:
            try:
                self.logger.info("正在关闭Spark会话...")
                self.spark.stop()
                self.spark = None
                self.logger.info("Spark会话已关闭")
            except Exception as e:
                self.logger.warning(f"关闭Spark会话时出现警告: {e}")
    
    @abstractmethod
    def _create_spark_session(self):
        """创建Spark会话，子类必须实现"""
        pass


class AsyncPipelineBase(BasePipeline):
    """异步Pipeline基类
    
    为使用异步处理的pipeline提供特定的执行框架
    """
    
    def __init__(self, config: BasePipelineConfig):
        super().__init__(config)
        import asyncio
        self.loop = None
    
    def _initialize_custom_components(self):
        """初始化异步组件"""
        import asyncio
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def _cleanup_custom_resources(self):
        """清理异步资源"""
        if self.loop:
            try:
                self.loop.close()
                self.loop = None
                self.logger.info("异步事件循环已关闭")
            except Exception as e:
                self.logger.warning(f"关闭异步事件循环时出现警告: {e}")
    
    def _execute_pipeline(self) -> Dict[str, Any]:
        """执行异步pipeline"""
        if not self.loop:
            raise RuntimeError("异步事件循环未初始化")
        
        return self.loop.run_until_complete(self._execute_async_pipeline())
    
    @abstractmethod
    async def _execute_async_pipeline(self) -> Dict[str, Any]:
        """执行异步pipeline逻辑，子类必须实现"""
        pass


# 工厂函数
def create_pipeline_from_config(config: BasePipelineConfig, pipeline_class=None):
    """根据配置创建pipeline实例"""
    if pipeline_class:
        return pipeline_class(config)
    
    # 根据配置类型自动选择pipeline类
    pipeline_name = config.get_pipeline_name()
    
    if 'text' in pipeline_name:
        # 动态导入text pipeline
        from .text_pipeline import TextPipeline
        return TextPipeline(config)
    elif 'multimodal' in pipeline_name:
        # 动态导入multimodal pipeline
        from .multimodal_pipeline import MultimodalPipeline
        return MultimodalPipeline(config)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_name}")


# 便利函数
def run_pipeline_with_config(config: BasePipelineConfig, pipeline_class=None) -> Dict[str, Any]:
    """使用配置运行pipeline的便利函数"""
    pipeline = create_pipeline_from_config(config, pipeline_class)
    return pipeline.run()