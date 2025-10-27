#!/usr/bin/env python3
"""
小红书CTR预估模型 - 文本特征提取管道（完整重构版）
从run_local_text_pipeline_batch_ssd.py完整迁移而来
集成了模块化的配置、系统管理、数据处理组件
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .base_pipeline import SparkPipelineBase
except ImportError:
    # 如果没有SparkPipelineBase，定义一个简单的基类
    class SparkPipelineBase:
        def __init__(self, config):
            self.config = config
            self.spark = None
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def get_output_prefix(self) -> str:
            return "text_features"
        
        def read_data(self):
            """抽象方法 - 读取数据"""
            raise NotImplementedError
        
        def process(self):
            """抽象方法 - 处理数据"""
            raise NotImplementedError

from .text_config import TextConfig, create_config_from_args
from .text_system import TextSystemMonitor, SparkSessionManager, TextResourceManager
from .text_data import DataPipeline

logger = logging.getLogger(__name__)


class TextPipeline(SparkPipelineBase):
    """文本特征提取Pipeline - 完整重构版"""
    
    def __init__(self, config: TextConfig):
        # 调用父类的__init__
        super().__init__(config)
        
        # TextPipeline特有的属性
        self.data_pipeline = None
        
        # 向后兼容的统计信息格式
        self.stats = {
            'start_time': datetime.now(),
            'batch_start': config.batch_start if hasattr(config, 'batch_start') else 0,
            'batch_end': config.batch_end if hasattr(config, 'batch_end') else 100,
            'status': 'initialized'
        }
    
    def get_output_prefix(self) -> str:
        """获取输出前缀"""
        return f"text_features_batch_{self.config.batch_start:05d}_{self.config.batch_end:05d}"
    
    def read_data(self):
        """实现BasePipeline的抽象方法 - 读取数据"""
        # 这个方法在新架构中由DataPipeline处理
        if self.data_pipeline:
            return self.data_pipeline.reader.read_note_info_batch()
        return None
    
    def process(self):
        """实现BasePipeline的抽象方法 - 处理数据"""
        # 这个方法在新架构中由run()方法处理
        return self.run()
    
    def _create_spark_session(self):
        """创建Spark会话，实现SparkPipelineBase的抽象方法"""
        from .text_system import SparkSessionManager
        self.spark = SparkSessionManager.create_optimized_spark_session(self.config)
    
    def _execute_pipeline(self) -> Dict[str, Any]:
        """执行核心pipeline逻辑，实现BasePipeline的抽象方法"""
        # 创建数据处理pipeline
        if not self.data_pipeline:
            from .text_data import DataPipeline
            self.data_pipeline = DataPipeline(self.spark, self.config)
        
        # 运行数据处理流水线
        result = self.data_pipeline.run_full_pipeline()
        
        return {
            'status': 'completed',
            'output_path': result.get('output_path', 'unknown'),
            'processed_batches': result.get('processed_batches', 0),
            'total_records': result.get('total_records', 0)
        }
    
    def _initialize_components(self):
        """初始化所有组件"""
        self.logger.info("初始化Pipeline组件...")
        
        # 1. 创建系统监控器
        self.monitor = TextSystemMonitor(self.config)
        self.monitor.update_stage("初始化组件")
        
        # 2. 创建资源管理器
        self.resource_manager = TextResourceManager(self.config)
        
        # 3. 检查磁盘空间
        if not self.resource_manager.check_available_space(10.0):
            raise RuntimeError("磁盘空间不足，无法继续处理")
        
        # 4. 创建Spark会话
        self.monitor.update_stage("创建Spark会话")
        self.spark = SparkSessionManager.create_optimized_spark_session(self.config)
        
        # 5. 创建数据处理pipeline
        self.data_pipeline = DataPipeline(self.spark, self.config)
        
        self.logger.info("所有组件初始化完成")
    
    def run(self) -> Dict[str, Any]:
        """运行完整的文本特征提取pipeline"""
        self.logger.info("="*60)
        self.logger.info("开始文本特征提取Pipeline")
        self.logger.info(f"批次范围: {self.config.batch_start} - {self.config.batch_end}")
        self.logger.info(f"时间: {datetime.now()}")
        self.logger.info("="*60)
        
        try:
            # 初始化所有组件
            self._initialize_components()
            
            # 运行数据处理流水线
            self.monitor.update_stage("执行数据处理")
            start_time = time.time()
            
            result = self.data_pipeline.run_full_pipeline()
            
            # 计算处理时间
            duration = time.time() - start_time
            hours = duration / 3600
            
            # 更新统计信息
            self.stats.update({
                'end_time': datetime.now(),
                'duration_seconds': duration,
                'duration_hours': hours,
                'status': 'completed' if result['success'] else 'failed',
                'rows_processed': result.get('rows_processed', 0),
                'output_path': result.get('output_path', ''),
                'output_size_mb': result.get('output_size_mb', 0)
            })
            
            # 更新监控统计
            if self.monitor:
                self.monitor.update_records_processed(result.get('rows_processed', 0))
                self.monitor.update_output_size(result.get('output_size_mb', 0))
                self.monitor.add_batch_time(duration)
                self.monitor.update_stage("Pipeline完成")
            
            # 打印结果摘要
            self._print_completion_summary(result, duration, hours)
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            self.stats.update({
                'end_time': datetime.now(),
                'status': 'failed',
                'error': str(e)
            })
            
            raise
        finally:
            self._cleanup_resources()
    
    def _print_completion_summary(self, result: dict, duration: float, hours: float):
        """打印完成摘要"""
        self.logger.info("="*60)
        if result['success']:
            self.logger.info(f"✅ 处理成功!")
            self.logger.info(f"总耗时: {hours:.2f} 小时 ({duration:.1f} 秒)")
            self.logger.info(f"处理行数: {result.get('rows_processed', 'N/A'):,}")
            self.logger.info(f"输出列数: {result.get('columns', 'N/A')}")
            self.logger.info(f"输出路径: {result.get('output_path', 'N/A')}")
            
            size_mb = result.get('output_size_mb', 0)
            if size_mb > 0:
                self.logger.info(f"输出文件大小: {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
            
            # 处理速度统计
            rows_processed = result.get('rows_processed', 0)
            if rows_processed > 0 and duration > 0:
                rate = rows_processed / duration
                self.logger.info(f"处理速度: {rate:.1f} 行/秒")
        else:
            self.logger.error(f"❌ 处理失败: {result.get('error', '未知错误')}")
        
        self.logger.info("="*60)
    
    def _cleanup_resources(self):
        """清理所有资源"""
        self.logger.info("开始清理Pipeline资源...")
        
        try:
            # 清理监控器
            if self.monitor:
                self.monitor.cleanup()
            
            # 清理Spark会话
            if self.spark:
                SparkSessionManager.cleanup_spark_session(self.spark)
                self.spark = None
            
            # 清理内存和临时文件
            if self.resource_manager:
                self.resource_manager.cleanup_memory()
                self.resource_manager.cleanup_temp_files()
            
            self.logger.info("Pipeline资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"资源清理时出现警告: {e}")


def run_text_pipeline(batch_start: int = 0, batch_end: int = 100,
                     min_impression: int = 500, log_file: Optional[str] = None) -> Dict[str, Any]:
    """运行文本特征提取pipeline的统一入口函数
    
    Args:
        batch_start: 起始批次号
        batch_end: 结束批次号  
        min_impression: 最小曝光阈值
        log_file: 日志文件路径
        
    Returns:
        Dict: 处理结果统计
    """
    # 配置日志
    if not log_file:
        log_file = f'text_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 确保logs目录存在
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果log_file没有目录部分，放到logs目录
    if not os.path.dirname(log_file):
        log_file = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 创建配置
    config = TextConfig(
        batch_start=batch_start,
        batch_end=batch_end,
        min_impression=min_impression
    )
    config.validate()
    
    # 创建并运行pipeline
    pipeline = TextPipeline(config)
    return pipeline.run()


def run_text_pipeline_from_args(args) -> Dict[str, Any]:
    """从命令行参数运行文本pipeline"""
    config = create_config_from_args(args)
    
    # 配置日志
    log_file = getattr(args, 'log_file', None)
    if not log_file:
        log_file = f'text_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # 确保logs目录存在
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    if not os.path.dirname(log_file):
        log_file = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 创建并运行pipeline
    pipeline = TextPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='文本特征提取管道')
    parser.add_argument('--batch-start', type=int, default=0, help='起始批次号')
    parser.add_argument('--batch-end', type=int, default=100, help='结束批次号')
    parser.add_argument('--min-impression', type=int, default=500, help='最小曝光阈值')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    
    # 可选的高级配置
    parser.add_argument('--driver-memory', type=int, help='Spark驱动内存(GB)')
    parser.add_argument('--executor-cores', type=int, help='执行器核心数')
    parser.add_argument('--note-info-path', type=str, help='note_info数据路径')
    parser.add_argument('--note-stats-path', type=str, help='note_stats数据路径')
    parser.add_argument('--output-path', type=str, help='输出基础路径')
    
    args = parser.parse_args()
    
    # 参数验证
    if args.batch_start < 0:
        print("批次起始号不能为负数")
        sys.exit(1)
    if args.batch_end < args.batch_start:
        print("批次结束号不能小于起始号")
        sys.exit(1)
    
    try:
        result = run_text_pipeline_from_args(args)
        if result['status'] == 'completed':
            print(f"✅ 处理完成: {result['rows_processed']:,}行")
            sys.exit(0)
        else:
            print(f"❌ 处理失败: {result.get('error', '未知错误')}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        sys.exit(1)