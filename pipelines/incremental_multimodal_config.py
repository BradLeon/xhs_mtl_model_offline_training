#!/usr/bin/env python3
"""
增量多模态特征提取配置模块
从已有的 image_features_parquet 提取缺失特征，避免重复 CLIP 计算
"""

import os
import psutil
from dataclasses import dataclass, field
from typing import Dict, Optional

from .base_config import (
    BasePipelineConfig, BatchProcessingMixin, MLConfigMixin,
    ConfigValidator, create_base_config_from_args
)


@dataclass
class IncrementalMultimodalConfig(BasePipelineConfig, BatchProcessingMixin, MLConfigMixin):
    """增量多模态特征提取配置

    从已有的 image_features_parquet 提取缺失特征：
    - 重命名现有特征 (image_feat_* -> cover_image_feat_*, text_feat_* -> title_feat_*)
    - 提取缺失的 CLIP 特征 (inner_image, content, tag, cover_ocr, inner_ocr)
    """

    # 输入路径（已有 image_feat_* 和 text_feat_* 的 parquet）
    input_path: str = "/Volumes/home/raw_data/image_features_parquet"
    output_path: str = "/Volumes/home/raw_data/multimodal_features_for_mtl_parquet"

    # 重写批次配置默认值
    batch_size: int = 3000

    # 重写ML配置默认值
    model_name: str = "ViT-B-16"
    target_dim: int = 512
    min_impression_threshold: int = 0  # 增量模式不再过滤，已在之前过滤过
    gpu_batch_size: int = 8

    # 特征重命名配置
    rename_features: bool = True

    # 跳过已有特征的提取（这些特征会被重命名）
    skip_cover_image: bool = True       # 跳过封面图（已有 image_feat_*）
    skip_title: bool = True             # 跳过标题（已有 text_feat_*）

    # 需要新增提取的特征
    enable_inner_images: bool = True    # 内页图 CLIP
    enable_content_text: bool = True    # 内容文本 CLIP
    enable_tag_text: bool = True        # tag 话题 CLIP
    enable_cover_ocr_clip: bool = True  # 封面 OCR CLIP
    enable_inner_ocr_clip: bool = True  # 内页 OCR CLIP

    # 多模态处理配置
    num_downloaders: int = 8            # 下载并发数
    download_queue_size: int = 800
    num_ocr_workers: int = 2            # OCR worker数量
    download_timeout: int = 30

    # 内页图片配置
    max_inner_images: int = 3           # 最大内页图片数量
    pooling_strategy: str = "mean"      # 池化策略: mean/max

    # 文本处理配置
    max_content_length: int = 200       # 内容文本最大长度
    content_chunk_size: int = 52        # 内容分块大小
    max_chunks: int = 4                 # 最大分块数量

    def get_rename_map(self) -> Dict[str, str]:
        """获取特征重命名映射"""
        return {
            'image_feat_': 'cover_image_feat_',  # 512 个特征
            'text_feat_': 'title_feat_',          # 512 个特征
        }

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

        # 增量特定验证
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
        temp_dir = "/tmp/incremental_multimodal_download"
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['MULTIMODAL_TEMP_DIR'] = temp_dir

    def get_feature_config(self) -> dict:
        """获取特征提取配置字典"""
        return {
            # 跳过的特征（已有）
            'skip_cover_image': self.skip_cover_image,
            'skip_title': self.skip_title,
            'rename_features': self.rename_features,
            # 需要提取的特征
            'enable_inner_images': self.enable_inner_images,
            'enable_content_text': self.enable_content_text,
            'enable_tag_text': self.enable_tag_text,
            'enable_cover_ocr_clip': self.enable_cover_ocr_clip,
            'enable_inner_ocr_clip': self.enable_inner_ocr_clip,
            # 其他配置
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


def create_incremental_config_from_args(args) -> IncrementalMultimodalConfig:
    """从命令行参数创建 IncrementalMultimodalConfig"""
    # 使用基类工厂函数处理通用参数
    config = create_base_config_from_args(IncrementalMultimodalConfig, args)

    # 处理增量特定参数
    if hasattr(args, 'skip_cover_image') and args.skip_cover_image is not None:
        config.skip_cover_image = args.skip_cover_image
    if hasattr(args, 'skip_title') and args.skip_title is not None:
        config.skip_title = args.skip_title
    if hasattr(args, 'rename_features') and args.rename_features is not None:
        config.rename_features = args.rename_features

    # 特征配置
    if hasattr(args, 'enable_inner_images') and args.enable_inner_images is not None:
        config.enable_inner_images = args.enable_inner_images
    if hasattr(args, 'enable_content_text') and args.enable_content_text is not None:
        config.enable_content_text = args.enable_content_text
    if hasattr(args, 'enable_tag_text') and args.enable_tag_text is not None:
        config.enable_tag_text = args.enable_tag_text
    if hasattr(args, 'enable_cover_ocr_clip') and args.enable_cover_ocr_clip is not None:
        config.enable_cover_ocr_clip = args.enable_cover_ocr_clip
    if hasattr(args, 'enable_inner_ocr_clip') and args.enable_inner_ocr_clip is not None:
        config.enable_inner_ocr_clip = args.enable_inner_ocr_clip

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


def get_default_incremental_config() -> IncrementalMultimodalConfig:
    """获取默认 IncrementalMultimodalConfig"""
    return IncrementalMultimodalConfig(
        input_path="/Volumes/home/raw_data/image_features_parquet",
        output_path="/Volumes/home/raw_data/image_features_ple_parquet"
    )
