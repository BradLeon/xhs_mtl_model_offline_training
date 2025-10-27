#!/usr/bin/env python3
"""
小红书CTR预估模型 - 多模态特征提取管道（重构版）
基于Chinese-CLIP的多模态特征提取，支持PLE多任务学习
从run_local_image_pipeline_chinese-clip_for_PLE.py完整迁移而来
"""

import os
import sys
import time
import asyncio
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import psutil

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .base_pipeline import AsyncPipelineBase
except ImportError:
    # 如果没有AsyncPipelineBase，定义一个简单的基类
    class AsyncPipelineBase:
        def __init__(self, spark=None, mode="local"):
            self.mode = mode
            self.logger = logging.getLogger(self.__class__.__name__)
        
        def get_output_prefix(self) -> str:
            return "multimodal_features"

from .multimodal_config import MultimodalConfig
from .multimodal_processors import get_clip_processor, OCRProcessor, ImageDownloader, cleanup_memory
from .multimodal_data import (
    DataReader, DataWriter, CheckpointManager, 
    extract_cover_image_url, extract_inner_image_urls, extract_tag_names
)

logger = logging.getLogger(__name__)


def get_memory_usage():
    """获取当前进程内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # 物理内存
        'vms': memory_info.vms / 1024 / 1024,  # 虚拟内存
        'percent': process.memory_percent()     # 内存使用百分比
    }


def log_memory_usage(stage: str, batch_id: int = None):
    """记录内存使用情况"""
    memory = get_memory_usage()
    batch_info = f" (batch {batch_id})" if batch_id is not None else ""
    logger.info(f"🧠 Memory {stage}{batch_info}: RSS={memory['rss']:.1f}MB, VMS={memory['vms']:.1f}MB, %={memory['percent']:.1f}%")


async def process_batch_async(batch_df: pd.DataFrame, config: MultimodalConfig, 
                             clip_processor=None) -> Dict[str, np.ndarray]:
    """异步批处理（从原始脚本迁移）"""
    
    # 提取数据
    cover_urls = []
    inner_images_list = []
    titles = []
    contents = []
    tags = []
    valid_indices = []
    filtered_rows = 0
    total_rows = 0

    for idx, row in batch_df.iterrows():
        total_rows += 1
        
        # 过滤低曝光数据
        if row.get('imp_num', 0) < config.min_impression_threshold:
            filtered_rows += 1
            continue
        
        valid_indices.append(idx)
        
        # 封面图URL
        if config.enable_cover_image:
            cover_url = extract_cover_image_url(row)
            cover_urls.append(cover_url)
        
        # 内页图URLs
        if config.enable_inner_images:
            inner_urls = extract_inner_image_urls(row, config.max_inner_images)
            inner_images_list.append(inner_urls)
        
        # 文本内容
        if config.enable_title_text:
            titles.append(str(row.get('title', '')))
        
        if config.enable_content_text:
            contents.append(str(row.get('content', '')))
        
        if config.enable_tag_text:
            tag_names = extract_tag_names(row.get('tag_info', ''))
            tags.append(tag_names)
    
    logger.info(f"Processing: {len(valid_indices)} valid rows out of {len(batch_df)}")
    logger.info(f"Filtered {filtered_rows} rows due to low impression count")

    if not valid_indices:
        logger.warning("No valid rows in batch")
        # 返回空特征
        empty_features = {
            'cover_image_features': np.zeros((len(batch_df), config.target_dim))
        }
        if config.enable_inner_images:
            empty_features['inner_images_features'] = np.zeros((len(batch_df), config.target_dim))
            empty_features['num_images'] = np.zeros(len(batch_df))
        if config.enable_title_text:
            empty_features['title_features'] = np.zeros((len(batch_df), config.target_dim))
        if config.enable_content_text:
            empty_features['content_features'] = np.zeros((len(batch_df), config.target_dim))
        if config.enable_tag_text:
            empty_features['tag_features'] = np.zeros((len(batch_df), config.target_dim))
        return empty_features

    # 下载图片
    downloader = ImageDownloader(num_workers=config.num_downloaders, timeout=config.download_timeout)
    
    # 下载封面图
    cover_images_bytes = []
    if config.enable_cover_image and cover_urls:
        cover_images_bytes = await downloader.download_batch(cover_urls)
    
    # 下载内页图
    inner_images_bytes_list = []
    if config.enable_inner_images and inner_images_list:
        for inner_urls in inner_images_list:
            if inner_urls:
                inner_bytes = await downloader.download_batch(inner_urls)
                inner_images_bytes_list.append(inner_bytes)
            else:
                inner_images_bytes_list.append([])
    
    await downloader.close()
    
    # Chinese-CLIP处理（使用传入的进程缓存处理器）
    if clip_processor is None:
        raise ValueError("CLIP processor must be provided for processing")
    
    features = {}
    
    # 分批处理CLIP特征以减少内存使用
    gpu_batch_size = config.gpu_batch_size
    
    # 处理封面图（分批）
    if config.enable_cover_image:
        all_cover_features = []
        for i in range(0, len(cover_images_bytes), gpu_batch_size):
            batch_images = cover_images_bytes[i:i+gpu_batch_size]
            batch_features = clip_processor.process_cover_images(batch_images)
            all_cover_features.append(batch_features)
        
        if all_cover_features:
            cover_features = np.vstack(all_cover_features)
            features['cover_image_features'] = cover_features
            # 清理中间结果
            del all_cover_features
    
    # 处理内页图（分批）
    if config.enable_inner_images:
        all_inner_features = []
        all_num_images = []
        for i in range(0, len(inner_images_bytes_list), gpu_batch_size):
            batch_inner_images = inner_images_bytes_list[i:i+gpu_batch_size]
            batch_features, batch_num_images = clip_processor.process_inner_images_batch(
                batch_inner_images, config.pooling_strategy
            )
            all_inner_features.append(batch_features)
            all_num_images.extend(batch_num_images)
        
        if all_inner_features:
            inner_features = np.vstack(all_inner_features)
            features['inner_images_features'] = inner_features
            features['num_images'] = np.array(all_num_images)
            # 清理中间结果
            del all_inner_features, all_num_images
    
    # 处理文本（分批）
    if config.enable_title_text:
        all_title_features = []
        for i in range(0, len(titles), gpu_batch_size):
            batch_titles = titles[i:i+gpu_batch_size]
            batch_features = clip_processor.process_texts(batch_titles)
            all_title_features.append(batch_features)
        
        if all_title_features:
            title_features = np.vstack(all_title_features)
            features['title_features'] = title_features
            del all_title_features
    
    if config.enable_content_text:
        all_content_features = []
        for i in range(0, len(contents), gpu_batch_size):
            batch_contents = contents[i:i+gpu_batch_size]
            batch_features = clip_processor.process_long_content(
                batch_contents, config.max_content_length, config.content_chunk_size, config.max_chunks
            )
            all_content_features.append(batch_features)
        
        if all_content_features:
            content_features = np.vstack(all_content_features)
            features['content_features'] = content_features
            del all_content_features
    
    if config.enable_tag_text:
        all_tag_features = []
        for i in range(0, len(tags), gpu_batch_size):
            batch_tags = tags[i:i+gpu_batch_size]
            batch_features = clip_processor.process_texts(batch_tags)
            all_tag_features.append(batch_features)
        
        if all_tag_features:
            tag_features = np.vstack(all_tag_features)
            features['tag_features'] = tag_features
            del all_tag_features

    # OCR处理（提取图片中的文本）
    ocr_processor = OCRProcessor()
    
    # 处理封面图OCR
    cover_ocr_texts = []
    cover_ocr_confidences = []
    if config.enable_cover_image and cover_images_bytes:
        cover_ocr_texts, cover_ocr_confidences = ocr_processor.extract_batch_texts(cover_images_bytes)
    else:
        cover_ocr_texts = [''] * len(valid_indices)
        cover_ocr_confidences = [0.0] * len(valid_indices)
    
    # 处理内页图OCR
    inner_ocr_texts = []
    inner_ocr_confidences = []
    if config.enable_inner_images and inner_images_bytes_list:
        inner_ocr_texts, inner_ocr_confidences = ocr_processor.extract_inner_images_ocr(inner_images_bytes_list)
    else:
        inner_ocr_texts = [''] * len(valid_indices)
        inner_ocr_confidences = [0.0] * len(valid_indices)
    
    # 将OCR结果分别添加到特征字典
    features['cover_image_ocr_texts'] = cover_ocr_texts
    features['cover_image_ocr_confidences'] = cover_ocr_confidences
    features['inner_images_ocr_texts'] = inner_ocr_texts
    features['inner_images_ocr_confidences'] = inner_ocr_confidences
    
    # 构建完整特征矩阵（填充到原始batch大小）
    full_features = {}
    for feat_name, feat_array in features.items():
        if feat_name == 'num_images':
            # 数值特征
            full_array = np.zeros(len(batch_df))
            for i, idx in enumerate(valid_indices):
                pos = batch_df.index.get_loc(idx)
                full_array[pos] = feat_array[i]
            full_features[feat_name] = full_array
        elif feat_name in ['cover_image_ocr_texts', 'cover_image_ocr_confidences', 
                           'inner_images_ocr_texts', 'inner_images_ocr_confidences']:
            # OCR文本特征（字符串或数值列表）
            if 'texts' in feat_name:
                full_array = [''] * len(batch_df)
            else:  # confidences
                full_array = [0.0] * len(batch_df)
            
            for i, idx in enumerate(valid_indices):
                pos = batch_df.index.get_loc(idx)
                full_array[pos] = feat_array[i]
            full_features[feat_name] = full_array
        else:
            # 向量特征
            full_array = np.zeros((len(batch_df), feat_array.shape[1]))
            for i, idx in enumerate(valid_indices):
                pos = batch_df.index.get_loc(idx)
                full_array[pos] = feat_array[i]
            full_features[feat_name] = full_array
    
    return full_features


def process_batch_wrapper(batch_id: int, batch_df: pd.DataFrame, config: MultimodalConfig) -> tuple:
    """批处理的包装函数（支持进程内模型缓存）"""
    log_memory_usage("batch start", batch_id)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 获取进程内缓存的CLIP处理器
        clip_processor = get_clip_processor(config)
        features = loop.run_until_complete(process_batch_async(batch_df, config, clip_processor))
        
        # 清理内存
        cleanup_memory()
        log_memory_usage("batch end", batch_id)
        
        return batch_id, features
    finally:
        loop.close()


class MultimodalPipeline(AsyncPipelineBase):
    """多模态特征提取Pipeline"""
    
    def __init__(self, config: MultimodalConfig):
        super().__init__(config)
        # MultimodalPipeline特有的属性将在这里初始化
    
    def get_output_prefix(self) -> str:
        return "multimodal_features"
    
    def read_data(self):
        """实现BasePipeline的抽象方法 - 读取数据"""
        # 这个方法在新架构中由DataReader处理，这里只需要简单实现
        return None
    
    def process(self):
        """实现BasePipeline的抽象方法 - 处理数据"""
        # 这个方法在新架构中由run()方法处理，这里只需要简单实现
        return self.run()
    
    async def _execute_async_pipeline(self) -> Dict[str, Any]:
        """执行异步pipeline逻辑，实现AsyncPipelineBase的抽象方法"""
        # 这里是一个简化的实现，实际的异步处理在run()方法中
        return {
            'status': 'completed',
            'message': 'Multimodal pipeline executed',
            'features_extracted': 0
        }
    
    def run(self) -> Dict[str, Any]:
        """运行多模态特征提取（从原始脚本迁移的主工作流程）"""
        log_memory_usage("pipeline start")
        
        # 调试信息：打印配置
        self.logger.info("🔍 DEBUG: MultimodalPipeline配置信息")
        self.logger.info(f"   enable_cover_image: {self.config.enable_cover_image}")
        self.logger.info(f"   enable_inner_images: {self.config.enable_inner_images}")
        self.logger.info(f"   enable_title_text: {self.config.enable_title_text}")
        self.logger.info(f"   enable_content_text: {self.config.enable_content_text}")
        self.logger.info(f"   enable_tag_text: {self.config.enable_tag_text}")
        self.logger.info(f"   model_name: {self.config.model_name}")
        self.logger.info(f"   target_dim: {self.config.target_dim}")
        
        # 初始化组件
        checkpoint_mgr = CheckpointManager()
        data_reader = DataReader(self.config.input_path, self.config.batch_size)
        data_writer = DataWriter(self.config.output_path)
        
        self.logger.info("Pipeline will use per-process CLIP model caching for better memory management")
        log_memory_usage("components initialized")
        
        # 加载或初始化checkpoint
        checkpoint = None
        total_processed = 0
        current_file_batches = []
        start_time_str = datetime.now().isoformat()
        
        if self.config.resume:
            checkpoint = checkpoint_mgr.load_progress()
            if checkpoint:
                total_processed = checkpoint.get('stats', {}).get('total_rows_processed', 0)
                start_time_str = checkpoint.get('start_time', start_time_str)
                current_file_batches = checkpoint.get('file_progress', {}).get('current_file_batches_completed', [])
                
                self.logger.info(f"📊 Resumed processing: {total_processed:,} rows already processed")
            else:
                self.logger.warning("No checkpoint found, starting from beginning")
        
        # 开始时间
        start_time = time.time()
        resume_processed = 0
        last_file_idx = -1
        
        # 创建进程池（适应高内存需求）
        max_workers = self.config.max_workers
        self.logger.info(f"Using multi-process mode with {max_workers} workers for multimodal pipeline")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 读取并处理批次
            for global_batch_id, batch_df, file_idx, batch_idx_in_file in data_reader.read_batches_with_resume(checkpoint):
                
                # 检测文件切换
                if file_idx != last_file_idx:
                    if last_file_idx >= 0:
                        self.logger.info(f"✅ Completed file {last_file_idx + 1}")
                    current_file_batches = []
                    last_file_idx = file_idx
                
                self.logger.info(f"Processing batch {global_batch_id} (file {file_idx + 1}, batch {batch_idx_in_file + 1}) - {len(batch_df)} rows")
                
                # 提交任务到进程池
                future = executor.submit(process_batch_wrapper, global_batch_id, batch_df, self.config)
                
                # 等待结果
                batch_id, features = future.result()
                
                # 写入结果
                data_writer.write_batch(batch_id, batch_df, features)
                
                # 更新进度
                current_file_batches.append(batch_idx_in_file)
                total_processed += len(batch_df)
                resume_processed += len(batch_df)
                
                # 保存checkpoint
                if global_batch_id % self.config.checkpoint_interval == 0:
                    progress = {
                        'version': '4.0-multimodal',
                        'model_name': self.config.model_name,
                        'start_time': start_time_str,
                        'batch_size': self.config.batch_size,
                        'input_path': self.config.input_path,
                        'output_path': self.config.output_path,
                        'feature_config': {
                            'enable_cover_image': self.config.enable_cover_image,
                            'enable_inner_images': self.config.enable_inner_images,
                            'enable_title_text': self.config.enable_title_text,
                            'enable_content_text': self.config.enable_content_text,
                            'enable_tag_text': self.config.enable_tag_text,
                            'pooling_strategy': self.config.pooling_strategy
                        },
                        'file_progress': {
                            'total_files': data_reader.total_files,
                            'completed_files': file_idx,
                            'current_file_index': file_idx,
                            'current_file_path': str(data_reader.parquet_files[file_idx]),
                            'current_file_batches_completed': current_file_batches
                        },
                        'file_batch_mapping': data_reader.file_mapping,
                        'stats': {
                            'total_rows_processed': total_processed,
                            'total_batches_completed': global_batch_id + 1,
                            'last_batch_id': global_batch_id
                        }
                    }
                    checkpoint_mgr.save_progress(progress)
                    self.logger.info(f"💾 Checkpoint saved at batch {global_batch_id}")
                    
                # 打印进度
                elapsed = time.time() - start_time
                rate = resume_processed / elapsed if elapsed > 0 else 0
                
                if global_batch_id % 5 == 0:
                    self.logger.info(f"Progress: {total_processed:,} rows, {rate:.1f} rows/sec, File {file_idx + 1}/{data_reader.total_files}")
                
        # 最终统计
        total_time = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"✅ Multimodal processing completed!")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Features enabled: cover={self.config.enable_cover_image}, inner={self.config.enable_inner_images}, title={self.config.enable_title_text}, content={self.config.enable_content_text}, tag={self.config.enable_tag_text}")
        self.logger.info(f"Total processed: {total_processed:,} rows")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Average rate: {total_processed/total_time:.1f} rows/sec")
        self.logger.info("="*60)
        
        return {
            'status': 'success',
            'rows_processed': total_processed,
            'duration': total_time,
            'output_path': self.config.output_path
        }


def run_multimodal_pipeline(input_path: str, output_path: str = None, **kwargs) -> Dict[str, Any]:
    """运行多模态特征提取pipeline的统一入口"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'multimodal_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("="*60)
    logger.info("开始多模态特征提取Pipeline")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"时间: {datetime.now()}")
    logger.info("="*60)
    
    # 设置multiprocessing启动方法
    mp.set_start_method('spawn', force=True)
    
    # 创建配置
    config = MultimodalConfig(
        input_path=input_path,
        output_path=output_path or "/Volumes/home/raw_data/image_features_ple_parquet",
        **kwargs
    )
    
    # 创建并运行pipeline
    pipeline = MultimodalPipeline(config)
    return pipeline.run()