#!/usr/bin/env python3
"""
多模态特征提取配置模块（重构版）
继承自BasePipelineConfig，使用统一的配置抽象
从run_local_image_pipeline_chinese-clip_for_PLE.py迁移而来
"""

import os
import psutil
from dataclasses import dataclass
from typing import Optional

from .base_config import (
    BasePipelineConfig, BatchProcessingMixin, MLConfigMixin,
    ConfigValidator, create_base_config_from_args
)


@dataclass
class MultimodalConfig(BasePipelineConfig, BatchProcessingMixin, MLConfigMixin):
    """多模态特征抽取配置（重构版）
    
    继承自多个混入类，提供完整的多模态pipeline配置功能
    """
    
    # 重写基类的必需字段（具体的默认值）
    input_path: str = "/Volumes/home/raw_data/note_info_parquet"
    output_path: str = "/Volumes/home/raw_data/image_features_ple_parquet"
    
    # 重写批次配置默认值（适应PLE复杂度）
    batch_size: int = 3000
    
    # 重写ML配置默认值
    model_name: str = "ViT-B-16"
    target_dim: int = 512
    min_impression_threshold: int = 5000
    gpu_batch_size: int = 8
    
    # 多模态特定配置
    num_downloaders: int = 8       # 下载并发数
    download_queue_size: int = 800
    num_ocr_workers: int = 2       # OCR worker数量
    download_timeout: int = 30
    
    # PLE特定配置
    enable_cover_image: bool = True  # 封面图特征
    enable_inner_images: bool = True  # 内页图特征
    enable_title_text: bool = True   # 标题文本特征
    enable_content_text: bool = True # 内容文本特征
    enable_tag_text: bool = True     # 标签文本特征
    
    # 内页图片配置
    max_inner_images: int = 5        # 最大内页图片数量
    pooling_strategy: str = "mean"   # 池化策略: mean/max
    
    # 文本处理配置
    max_content_length: int = 200    # 内容文本最大长度
    content_chunk_size: int = 52     # 内容分块大小
    max_chunks: int = 4              # 最大分块数量
    
    def _setup_dynamic_config(self):
        """设置动态配置"""
        # 自动调整worker数量基于系统资源
        if self.max_workers == 2:  # 如果是默认值，则根据系统动态调整
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            # 多模态处理内存密集，保守设置
            if available_memory > 32:
                self.max_workers = 3
            elif available_memory > 16:
                self.max_workers = 2
            else:
                self.max_workers = 1
    
    def validate(self) -> None:
        """验证配置的有效性"""
        # 使用基类验证器
        ConfigValidator.validate_paths(self)
        ConfigValidator.validate_performance_config(self)
        ConfigValidator.validate_batch_config(self)
        
        # 多模态特定验证
        if self.num_downloaders <= 0:
            raise ValueError("num_downloaders must be positive")
        if self.download_queue_size <= 0:
            raise ValueError("download_queue_size must be positive")
        if self.num_ocr_workers < 0:
            raise ValueError("num_ocr_workers must be non-negative")
        if self.download_timeout <= 0:
            raise ValueError("download_timeout must be positive")
        if self.max_inner_images < 0:
            raise ValueError("max_inner_images must be non-negative")
        if self.max_content_length <= 0:
            raise ValueError("max_content_length must be positive")
        if self.content_chunk_size <= 0:
            raise ValueError("content_chunk_size must be positive")
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError("pooling_strategy must be 'mean' or 'max'")
    
    def get_output_path(self) -> str:
        """获取实际输出路径"""
        batch_info = self.get_batch_info()
        timestamp = self.get_timestamp_suffix()
        return f"{self.output_path}/{batch_info}/dt={timestamp}"
    
    def setup_environment(self) -> None:
        """设置环境变量和目录"""
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 创建临时下载目录
        temp_dir = "/tmp/multimodal_download"
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['MULTIMODAL_TEMP_DIR'] = temp_dir
    
    def get_feature_config(self) -> dict:
        """获取特征提取配置字典"""
        return {
            'enable_cover_image': self.enable_cover_image,
            'enable_inner_images': self.enable_inner_images,
            'enable_title_text': self.enable_title_text,
            'enable_content_text': self.enable_content_text,
            'enable_tag_text': self.enable_tag_text,
            'max_inner_images': self.max_inner_images,
            'pooling_strategy': self.pooling_strategy,
            'max_content_length': self.max_content_length,
            'content_chunk_size': self.content_chunk_size,
            'max_chunks': self.max_chunks
        }
    
    def get_timestamp_suffix(self) -> str:
        """获取时间戳后缀"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def create_config_from_args(args) -> MultimodalConfig:
    """从命令行参数创建MultimodalConfig（兼容性函数）"""
    # 使用基类工厂函数处理通用参数
    config = create_base_config_from_args(MultimodalConfig, args)
    
    # 处理多模态特定参数
    # 特征配置
    if getattr(args, 'enable_all_features', True):
        config.enable_cover_image = True
        config.enable_inner_images = True
        config.enable_title_text = True
        config.enable_content_text = True
        config.enable_tag_text = True
    else:
        config.enable_cover_image = getattr(args, 'enable_cover_image', True)
        config.enable_inner_images = getattr(args, 'enable_inner_images', False)
        config.enable_title_text = getattr(args, 'enable_title_text', True)
        config.enable_content_text = getattr(args, 'enable_content_text', False)
        config.enable_tag_text = getattr(args, 'enable_tag_text', False)
    
    # 其他多模态特定参数
    if hasattr(args, 'num_downloaders') and args.num_downloaders:
        config.num_downloaders = args.num_downloaders
    if hasattr(args, 'download_queue_size') and args.download_queue_size:
        config.download_queue_size = args.download_queue_size
    if hasattr(args, 'num_ocr_workers') and args.num_ocr_workers is not None:
        config.num_ocr_workers = args.num_ocr_workers
    if hasattr(args, 'download_timeout') and args.download_timeout:
        config.download_timeout = args.download_timeout
    if hasattr(args, 'pooling_strategy') and args.pooling_strategy:
        config.pooling_strategy = args.pooling_strategy
    if hasattr(args, 'max_inner_images') and args.max_inner_images is not None:
        config.max_inner_images = args.max_inner_images
    if hasattr(args, 'max_content_length') and args.max_content_length:
        config.max_content_length = args.max_content_length
    if hasattr(args, 'content_chunk_size') and args.content_chunk_size:
        config.content_chunk_size = args.content_chunk_size
    if hasattr(args, 'max_chunks') and args.max_chunks:
        config.max_chunks = args.max_chunks
    
    return config


def get_default_config() -> MultimodalConfig:
    """获取默认MultimodalConfig（兼容性函数）"""
    return MultimodalConfig(
        input_path="/Volumes/home/raw_data/note_info_parquet",
        output_path="/Volumes/home/raw_data/image_features_ple_parquet"
    )