#!/usr/bin/env python3
"""
增量多模态特征提取 Pipeline
从已有的 image_features_parquet 提取缺失特征，避免重复 CLIP 计算

主要功能：
1. 重命名现有特征 (image_feat_* -> cover_image_feat_*, text_feat_* -> title_feat_*)
2. 提取缺失的 CLIP 特征 (inner_image, content, tag, cover_ocr, inner_ocr)
3. 合并输出到新的 parquet 文件
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

from .incremental_multimodal_config import IncrementalMultimodalConfig
from .multimodal_processors import get_clip_processor, OCRProcessor, ImageDownloader, cleanup_memory
from .multimodal_data import extract_inner_image_urls, extract_tag_names, CheckpointManager

logger = logging.getLogger(__name__)


def get_memory_usage():
    """获取当前进程内存使用情况（MB）"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,
        'vms': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }


def log_memory_usage(stage: str, batch_id: int = None):
    """记录内存使用情况"""
    memory = get_memory_usage()
    batch_info = f" (batch {batch_id})" if batch_id is not None else ""
    logger.info(f"🧠 Memory {stage}{batch_info}: RSS={memory['rss']:.1f}MB, VMS={memory['vms']:.1f}MB, %={memory['percent']:.1f}%")


def rename_existing_features(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """重命名已有的 CLIP 特征列

    Args:
        df: 输入 DataFrame
        rename_map: 前缀重命名映射 {'image_feat_': 'cover_image_feat_', ...}

    Returns:
        重命名后的 DataFrame
    """
    rename_dict = {}
    for old_prefix, new_prefix in rename_map.items():
        for i in range(512):
            old_name = f'{old_prefix}{i}'
            new_name = f'{new_prefix}{i}'
            if old_name in df.columns:
                rename_dict[old_name] = new_name

    if rename_dict:
        logger.info(f"Renaming {len(rename_dict)} feature columns")
        df = df.rename(columns=rename_dict)

    return df


async def process_incremental_batch_async(
    batch_df: pd.DataFrame,
    config: IncrementalMultimodalConfig,
    clip_processor=None
) -> Dict[str, np.ndarray]:
    """异步处理增量特征提取

    只提取缺失的特征，跳过已有的 cover_image 和 title 特征
    """

    # 提取数据
    inner_images_list = []
    contents = []
    tags = []
    valid_indices = list(batch_df.index)  # 增量模式下所有行都是有效的

    for idx, row in batch_df.iterrows():
        # 内页图URLs（只有在需要提取时才收集）
        if config.enable_inner_images:
            inner_urls = extract_inner_image_urls(row, config.max_inner_images)
            inner_images_list.append(inner_urls)

        # 内容文本
        if config.enable_content_text:
            contents.append(str(row.get('content', '') or ''))

        # tag 话题标签
        if config.enable_tag_text:
            tag_names = extract_tag_names(row.get('tag_info', ''))
            tags.append(tag_names)

    logger.info(f"Processing {len(valid_indices)} rows for incremental features")

    if not valid_indices:
        logger.warning("No valid rows in batch")
        return {}

    # 下载内页图片（封面图已处理过，跳过）
    downloader = ImageDownloader(num_workers=config.num_downloaders, timeout=config.download_timeout)

    inner_images_bytes_list = []
    if config.enable_inner_images and inner_images_list:
        for inner_urls in inner_images_list:
            if inner_urls:
                inner_bytes = await downloader.download_batch(inner_urls)
                inner_images_bytes_list.append(inner_bytes)
            else:
                inner_images_bytes_list.append([])

    await downloader.close()

    # CLIP 处理
    if clip_processor is None:
        raise ValueError("CLIP processor must be provided for processing")

    features = {}
    gpu_batch_size = config.gpu_batch_size

    # 处理内页图（分批）
    if config.enable_inner_images and inner_images_bytes_list:
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
            del all_inner_features, all_num_images

    # 处理内容文本（分批）
    if config.enable_content_text and contents:
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

    # 处理 tag 文本（分批）
    if config.enable_tag_text and tags:
        all_tag_features = []
        for i in range(0, len(tags), gpu_batch_size):
            batch_tags = tags[i:i+gpu_batch_size]
            batch_features = clip_processor.process_texts(batch_tags)
            all_tag_features.append(batch_features)

        if all_tag_features:
            tag_features = np.vstack(all_tag_features)
            features['tag_features'] = tag_features
            del all_tag_features

    # OCR 处理（提取封面图和内页图中的文本）
    ocr_processor = OCRProcessor()

    # 封面图 OCR → CLIP（需要重新下载封面图）
    cover_ocr_texts = [''] * len(valid_indices)
    if config.enable_cover_ocr_clip:
        # 从 file_url_list 提取封面图 URL 并下载
        cover_urls = []
        for idx, row in batch_df.iterrows():
            file_url_list = row.get('file_url_list', '')
            if file_url_list and isinstance(file_url_list, str):
                urls = file_url_list.strip('[]').split(',')
                if urls and urls[0]:
                    cover_urls.append(urls[0].strip().strip('"\''))
                else:
                    cover_urls.append('')
            else:
                cover_urls.append('')

        # 下载封面图用于 OCR
        downloader2 = ImageDownloader(num_workers=config.num_downloaders, timeout=config.download_timeout)
        cover_images_bytes = await downloader2.download_batch(cover_urls)
        await downloader2.close()

        # OCR 提取
        if cover_images_bytes:
            cover_ocr_texts, _ = ocr_processor.extract_batch_texts(cover_images_bytes)

    # 内页图 OCR → CLIP
    inner_ocr_texts = [''] * len(valid_indices)
    if config.enable_inner_ocr_clip and inner_images_bytes_list:
        inner_ocr_texts, _ = ocr_processor.extract_inner_images_ocr(inner_images_bytes_list)

    # 将 OCR 文本转为 CLIP 特征
    if config.enable_cover_ocr_clip and cover_ocr_texts:
        all_cover_ocr_features = []
        for i in range(0, len(cover_ocr_texts), gpu_batch_size):
            batch_texts = cover_ocr_texts[i:i+gpu_batch_size]
            batch_texts = [t if t and t.strip() else '' for t in batch_texts]
            batch_features = clip_processor.process_texts(batch_texts)
            all_cover_ocr_features.append(batch_features)

        if all_cover_ocr_features:
            cover_ocr_features = np.vstack(all_cover_ocr_features)
            features['cover_ocr_features'] = cover_ocr_features
            del all_cover_ocr_features

    if config.enable_inner_ocr_clip and inner_ocr_texts:
        all_inner_ocr_features = []
        for i in range(0, len(inner_ocr_texts), gpu_batch_size):
            batch_texts = inner_ocr_texts[i:i+gpu_batch_size]
            batch_texts = [t if t and t.strip() else '' for t in batch_texts]
            batch_features = clip_processor.process_texts(batch_texts)
            all_inner_ocr_features.append(batch_features)

        if all_inner_ocr_features:
            inner_ocr_features = np.vstack(all_inner_ocr_features)
            features['inner_ocr_features'] = inner_ocr_features
            del all_inner_ocr_features

    return features


def process_incremental_batch_wrapper(batch_id: int, batch_df: pd.DataFrame, config: IncrementalMultimodalConfig) -> tuple:
    """批处理的包装函数"""
    log_memory_usage("batch start", batch_id)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        clip_processor = get_clip_processor(config)
        features = loop.run_until_complete(process_incremental_batch_async(batch_df, config, clip_processor))

        cleanup_memory()
        log_memory_usage("batch end", batch_id)

        return batch_id, features
    finally:
        loop.close()


def merge_features_to_df(df: pd.DataFrame, features: Dict[str, np.ndarray], target_dim: int = 512) -> pd.DataFrame:
    """将提取的特征合并到 DataFrame

    使用 pd.concat 一次性添加所有列，避免 DataFrame 碎片化

    Args:
        df: 原始 DataFrame
        features: 特征字典
        target_dim: 特征维度

    Returns:
        合并后的 DataFrame
    """
    # 收集所有需要添加的 DataFrame
    dfs_to_concat = [df]

    # 添加内页图特征
    if 'inner_images_features' in features:
        inner_image_df = pd.DataFrame(
            features['inner_images_features'],
            columns=[f'inner_image_feat_{i}' for i in range(target_dim)],
            index=df.index
        )
        dfs_to_concat.append(inner_image_df)

    if 'num_images' in features:
        num_images_df = pd.DataFrame({'num_images': features['num_images']}, index=df.index)
        dfs_to_concat.append(num_images_df)

    # 添加内容文本特征
    if 'content_features' in features:
        content_df = pd.DataFrame(
            features['content_features'],
            columns=[f'content_feat_{i}' for i in range(target_dim)],
            index=df.index
        )
        dfs_to_concat.append(content_df)

    # 添加 tag 特征
    if 'tag_features' in features:
        tag_df = pd.DataFrame(
            features['tag_features'],
            columns=[f'tag_feat_{i}' for i in range(target_dim)],
            index=df.index
        )
        dfs_to_concat.append(tag_df)

    # 添加 OCR CLIP 特征
    if 'cover_ocr_features' in features:
        cover_ocr_df = pd.DataFrame(
            features['cover_ocr_features'],
            columns=[f'cover_ocr_feat_{i}' for i in range(target_dim)],
            index=df.index
        )
        dfs_to_concat.append(cover_ocr_df)

    if 'inner_ocr_features' in features:
        inner_ocr_df = pd.DataFrame(
            features['inner_ocr_features'],
            columns=[f'inner_ocr_feat_{i}' for i in range(target_dim)],
            index=df.index
        )
        dfs_to_concat.append(inner_ocr_df)

    # 一次性合并所有 DataFrame，避免碎片化
    if len(dfs_to_concat) > 1:
        result_df = pd.concat(dfs_to_concat, axis=1)
    else:
        result_df = df.copy()

    return result_df


class IncrementalMultimodalPipeline:
    """增量多模态特征提取 Pipeline"""

    def __init__(self, config: IncrementalMultimodalConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> Dict[str, Any]:
        """运行增量特征提取"""
        log_memory_usage("pipeline start")

        # 打印配置
        self.logger.info("="*60)
        self.logger.info("🚀 开始增量多模态特征提取 Pipeline")
        self.logger.info(f"输入路径: {self.config.input_path}")
        self.logger.info(f"输出路径: {self.config.output_path}")
        self.logger.info("="*60)
        self.logger.info("特征配置:")
        self.logger.info(f"  - 跳过 cover_image: {self.config.skip_cover_image} (重命名 image_feat_*)")
        self.logger.info(f"  - 跳过 title: {self.config.skip_title} (重命名 text_feat_*)")
        self.logger.info(f"  - 提取 inner_images: {self.config.enable_inner_images}")
        self.logger.info(f"  - 提取 content_text: {self.config.enable_content_text}")
        self.logger.info(f"  - 提取 tag_text: {self.config.enable_tag_text}")
        self.logger.info(f"  - 提取 cover_ocr_clip: {self.config.enable_cover_ocr_clip}")
        self.logger.info(f"  - 提取 inner_ocr_clip: {self.config.enable_inner_ocr_clip}")
        self.logger.info("="*60)

        # 查找输入文件
        input_path = Path(self.config.input_path)
        if input_path.is_file():
            parquet_files = [input_path]
        else:
            parquet_files = sorted(input_path.glob("**/*.parquet"))

        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.config.input_path}")

        self.logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")

        # 创建输出目录
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 初始化统计
        start_time = time.time()
        total_processed = 0
        output_files = []

        # 创建进程池
        max_workers = self.config.max_workers
        self.logger.info(f"使用 {max_workers} 个 worker 进行处理")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for file_idx, parquet_file in enumerate(parquet_files):
                self.logger.info(f"\n处理文件 {file_idx + 1}/{len(parquet_files)}: {parquet_file.name}")

                # 读取文件
                df = pd.read_parquet(parquet_file)
                self.logger.info(f"读取 {len(df)} 行, {len(df.columns)} 列")

                # 文件级过滤：过滤低曝光数据（在 CLIP 计算前过滤，节省计算资源）
                if self.config.min_impression_threshold > 0 and 'imp_num' in df.columns:
                    before_count = len(df)
                    df = df[df['imp_num'] >= self.config.min_impression_threshold].reset_index(drop=True)
                    after_count = len(df)
                    filtered_count = before_count - after_count
                    self.logger.info(f"曝光过滤 (>={self.config.min_impression_threshold}): {before_count:,} -> {after_count:,} 行 (过滤 {filtered_count:,} 行)")

                    if len(df) == 0:
                        self.logger.warning(f"文件 {parquet_file.name} 过滤后无数据，跳过")
                        continue

                # 重命名已有特征
                if self.config.rename_features:
                    df = rename_existing_features(df, self.config.get_rename_map())

                # 分批处理
                batch_size = self.config.batch_size
                num_batches = (len(df) + batch_size - 1) // batch_size

                # 收集所有批次结果
                batch_results = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx].copy()
                    global_batch_id = file_idx * 1000 + batch_idx

                    self.logger.info(f"处理批次 {batch_idx + 1}/{num_batches} ({len(batch_df)} 行)")

                    # 提交任务
                    future = executor.submit(
                        process_incremental_batch_wrapper,
                        global_batch_id, batch_df, self.config
                    )
                    batch_id, features = future.result()

                    # 合并特征到批次 DataFrame
                    batch_result_df = merge_features_to_df(batch_df, features, self.config.target_dim)
                    batch_results.append(batch_result_df)

                    total_processed += len(batch_df)

                    # 打印进度
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    self.logger.info(f"进度: {total_processed:,} 行, {rate:.1f} 行/秒")

                # 合并所有批次结果
                if batch_results:
                    df = pd.concat(batch_results, axis=0, ignore_index=True)
                    del batch_results  # 释放内存

                # 保存结果
                output_file = output_path / f"incremental_{parquet_file.stem}.parquet"
                df.to_parquet(output_file, index=False)
                output_files.append(str(output_file))
                self.logger.info(f"✅ 保存: {output_file}")

        # 最终统计
        total_time = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"✅ 增量特征提取完成!")
        self.logger.info(f"处理行数: {total_processed:,}")
        self.logger.info(f"总耗时: {total_time/3600:.2f} 小时")
        self.logger.info(f"平均速度: {total_processed/total_time:.1f} 行/秒")
        self.logger.info(f"输出文件: {len(output_files)} 个")
        self.logger.info("="*60)

        return {
            'status': 'success',
            'rows_processed': total_processed,
            'duration': total_time,
            'output_path': str(output_path),
            'output_files': output_files
        }


def run_incremental_multimodal_pipeline(input_path: str, output_path: str = None, **kwargs) -> Dict[str, Any]:
    """运行增量多模态特征提取 pipeline 的统一入口"""

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'incremental_multimodal_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    logger.info("="*60)
    logger.info("开始增量多模态特征提取 Pipeline")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"时间: {datetime.now()}")
    logger.info("="*60)

    # 设置 multiprocessing 启动方法
    mp.set_start_method('spawn', force=True)

    # 创建配置
    config = IncrementalMultimodalConfig(
        input_path=input_path,
        output_path=output_path or "/Volumes/home/raw_data/image_features_ple_parquet",
        **kwargs
    )

    # 创建并运行 pipeline
    pipeline = IncrementalMultimodalPipeline(config)
    return pipeline.run()
