#!/usr/bin/env python3
"""
多模态数据管理模块
从run_local_image_pipeline_chinese-clip_for_PLE.py迁移而来
包含数据读取、写入和checkpoint管理
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from .multimodal_config import MultimodalConfig

# 导入特征规范工具
try:
    from src.utils.feature_spec import FeatureSpec
except ImportError:
    # 兜底：如果无法导入，使用本地定义
    logging.warning("无法导入FeatureSpec，使用本地特征命名")
    FeatureSpec = None

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Checkpoint管理器"""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress_multimodal.json"
        self.file_mapping_file = self.checkpoint_dir / "file_mapping_multimodal.pkl"
        
    def save_progress(self, progress: Dict[str, Any]):
        """保存进度"""
        progress['version'] = '4.0-multimodal'
        progress['last_update'] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    def load_progress(self) -> Optional[Dict[str, Any]]:
        """加载进度"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                progress = json.load(f)
                # 检查版本兼容性
                if not progress.get('version', '').startswith('4.0'):
                    logger.warning(f"Checkpoint version mismatch: {progress.get('version')}")
                return progress
        return None
        
    def save_file_mapping(self, mapping: List[Dict]):
        """保存文件映射"""
        with open(self.file_mapping_file, 'wb') as f:
            pickle.dump(mapping, f)
    
    def load_file_mapping(self) -> Optional[List[Dict]]:
        """加载文件映射"""
        if self.file_mapping_file.exists():
            with open(self.file_mapping_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def build_file_mapping(self, parquet_files: List[Path], batch_size: int) -> List[Dict]:
        """构建文件映射"""
        logger.info(f"Building file mapping for {len(parquet_files)} files...")
        
        file_mapping = []
        current_batch_id = 0
        total_rows = 0
        
        for file_idx, file_path in enumerate(parquet_files):
            try:
                metadata = pq.ParquetFile(file_path).metadata
                num_rows = metadata.num_rows
                num_batches = (num_rows + batch_size - 1) // batch_size
                
                file_info = {
                    'file_index': file_idx,
                    'path': str(file_path),
                    'name': file_path.name,
                    'start_batch': current_batch_id,
                    'end_batch': current_batch_id + num_batches - 1,
                    'num_batches': num_batches,
                    'rows': num_rows,
                    'size_bytes': file_path.stat().st_size
                }
                file_mapping.append(file_info)
                current_batch_id += num_batches
                total_rows += num_rows
                
                if (file_idx + 1) % 100 == 0:
                    logger.info(f"  Mapped {file_idx + 1}/{len(parquet_files)} files...")
                    
            except Exception as e:
                logger.warning(f"Failed to map file {file_path}: {e}")
                file_info = {
                    'file_index': file_idx,
                    'path': str(file_path),
                    'name': file_path.name,
                    'start_batch': current_batch_id,
                    'end_batch': current_batch_id,
                    'num_batches': 0,
                    'rows': 0,
                    'size_bytes': 0,
                    'error': str(e)
                }
                file_mapping.append(file_info)
        
        logger.info(f"File mapping complete: {len(file_mapping)} files, {current_batch_id} total batches, {total_rows:,} total rows")
        return file_mapping


class DataReader:
    """数据读取器"""

    def __init__(self, input_path: str, batch_size: int = 3000, min_impression_threshold: int = 0):
        self.input_path = Path(input_path)
        self.batch_size = batch_size
        self.min_impression_threshold = min_impression_threshold
        self.parquet_files = self._get_parquet_files()
        self.total_files = len(self.parquet_files)
        self.file_mapping = None
        
    def _get_parquet_files(self) -> List[Path]:
        """获取所有parquet文件"""
        if self.input_path.is_file():
            return [self.input_path]
        
        files = []
        for pattern in ["**/*.parquet", "**/*.snappy.parquet"]:
            files.extend(self.input_path.glob(pattern))
        
        return sorted(set(files))
        
    def read_batches_with_resume(self, checkpoint: Optional[Dict] = None):
        """支持恢复的批处理读取"""
        start_file_idx = 0
        completed_batches_in_current_file = set()
        
        if checkpoint and checkpoint.get('version', '').startswith('4.0'):
            file_progress = checkpoint.get('file_progress', {})
            start_file_idx = file_progress.get('current_file_index', 0)
            completed_batches_in_current_file = set(
                file_progress.get('current_file_batches_completed', [])
            )
            
            self.file_mapping = checkpoint.get('file_batch_mapping')
            
            logger.info("="*60)
            logger.info(f"📌 Resuming multimodal pipeline from checkpoint")
            logger.info(f"  Starting from file {start_file_idx + 1}/{self.total_files}")
            logger.info("="*60)
        
        # 构建或加载文件映射
        if not self.file_mapping:
            checkpoint_mgr = CheckpointManager()
            self.file_mapping = checkpoint_mgr.load_file_mapping()
            
            if not self.file_mapping:
                self.file_mapping = checkpoint_mgr.build_file_mapping(
                    self.parquet_files, self.batch_size
                )
                checkpoint_mgr.save_file_mapping(self.file_mapping)
        
        # 处理文件
        for file_idx in range(start_file_idx, self.total_files):
            if file_idx >= len(self.file_mapping):
                logger.error(f"❌ File index {file_idx} out of range")
                break
                
            file_path = self.parquet_files[file_idx]
            file_info = self.file_mapping[file_idx]
            
            if 'error' in file_info:
                logger.warning(f"Skipping errored file {file_idx + 1}: {file_info['error']}")
                continue
            
            logger.info(f"📂 Processing file {file_idx + 1}/{self.total_files}: {file_path.name}")
            
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                logger.error(f"Failed to read file {file_path}: {e}")
                continue

            # 文件级过滤：过滤低曝光数据（在 CLIP 计算前过滤，节省计算资源）
            if self.min_impression_threshold > 0 and 'imp_num' in df.columns:
                before_count = len(df)
                df = df[df['imp_num'] >= self.min_impression_threshold].reset_index(drop=True)
                after_count = len(df)
                filtered_count = before_count - after_count
                logger.info(f"   曝光过滤 (>={self.min_impression_threshold}): {before_count:,} -> {after_count:,} 行 (过滤 {filtered_count:,} 行)")

                if len(df) == 0:
                    logger.warning(f"   文件 {file_path.name} 过滤后无数据，跳过")
                    continue

            # 生成批次
            batch_idx_in_file = 0
            for start_idx in range(0, len(df), self.batch_size):
                global_batch_id = file_info['start_batch'] + batch_idx_in_file
                
                # 检查是否需要跳过
                if file_idx == start_file_idx and batch_idx_in_file in completed_batches_in_current_file:
                    batch_idx_in_file += 1
                    continue
                
                end_idx = min(start_idx + self.batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx].copy()
                
                yield global_batch_id, batch_df, file_idx, batch_idx_in_file
                batch_idx_in_file += 1


class DataWriter:
    """数据写入器"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化特征规范
        self.feature_spec = None
        if FeatureSpec is not None:
            try:
                self.feature_spec = FeatureSpec()
                logging.info("✅ DataWriter使用特征规范进行命名")
            except Exception as e:
                logging.warning(f"特征规范初始化失败，使用默认命名: {e}")
                self.feature_spec = None
        
    def write_batch(self, batch_id: int, df: pd.DataFrame, features: Dict[str, np.ndarray]):
        """写入特征批次数据"""
        # 动态确定特征维度
        num_features = None
        for feature_key in features:
            if len(features[feature_key].shape) == 2:  # 2D特征数组
                num_features = features[feature_key].shape[1]
                break
        
        if num_features is None:
            raise ValueError("No valid feature arrays found in features dict")
        
        # 从基础数据开始
        result_df = df.copy()
        
        # 封面图特征（如果启用）
        if 'cover_image_features' in features:
            # 使用特征规范生成列名
            if self.feature_spec:
                column_names = self.feature_spec.generate_clip_names('cover_image', num_features)
            else:
                column_names = [f'cover_image_feat_{i}' for i in range(num_features)]
            
            cover_feat_df = pd.DataFrame(
                features['cover_image_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, cover_feat_df], axis=1)
        
        # 内页图特征（如果启用）
        if 'inner_images_features' in features:
            # 使用特征规范生成列名
            if self.feature_spec:
                column_names = self.feature_spec.generate_clip_names('inner_image', num_features)
            else:
                column_names = [f'inner_image_feat_{i}' for i in range(num_features)]
            
            inner_feat_df = pd.DataFrame(
                features['inner_images_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, inner_feat_df], axis=1)
            
            # 内页图数量（使用特征规范）
            if self.feature_spec:
                metadata_features = self.feature_spec.get_metadata_features()
                num_images_col = metadata_features.get('num_images', 'num_images')
            else:
                num_images_col = 'num_images'
            result_df[num_images_col] = features['num_images']
        
        # 标题文本特征
        if 'title_features' in features:
            # 使用特征规范生成列名
            if self.feature_spec:
                column_names = self.feature_spec.generate_clip_names('title', num_features)
            else:
                column_names = [f'title_feat_{i}' for i in range(num_features)]
            
            title_feat_df = pd.DataFrame(
                features['title_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, title_feat_df], axis=1)
        
        # 内容文本特征
        if 'content_features' in features:
            # 使用特征规范生成列名
            if self.feature_spec:
                column_names = self.feature_spec.generate_clip_names('content', num_features)
            else:
                column_names = [f'content_feat_{i}' for i in range(num_features)]
            
            content_feat_df = pd.DataFrame(
                features['content_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, content_feat_df], axis=1)
        
        # 标签文本特征
        if 'tag_features' in features:
            # 使用特征规范生成列名
            if self.feature_spec:
                column_names = self.feature_spec.generate_clip_names('tag', num_features)
            else:
                column_names = [f'tag_feat_{i}' for i in range(num_features)]
            
            tag_feat_df = pd.DataFrame(
                features['tag_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, tag_feat_df], axis=1)
        
        # 添加OCR结果（如果有）
        if 'cover_image_ocr_texts' in features:
            # 使用特征规范的OCR特征名
            if self.feature_spec:
                ocr_features = self.feature_spec.get_ocr_features()
                ocr_text_col = ocr_features.get('cover_image_ocr_text', 'cover_image_ocr_text')
                ocr_conf_col = ocr_features.get('cover_image_ocr_confidence', 'cover_image_ocr_confidence')
            else:
                ocr_text_col = 'cover_image_ocr_text'
                ocr_conf_col = 'cover_image_ocr_confidence'
            
            result_df[ocr_text_col] = features['cover_image_ocr_texts']
            result_df[ocr_conf_col] = features['cover_image_ocr_confidences']
            
        if 'inner_images_ocr_texts' in features:
            # 使用特征规范的OCR特征名
            if self.feature_spec:
                ocr_features = self.feature_spec.get_ocr_features()
                ocr_text_col = ocr_features.get('inner_images_ocr_text', 'inner_images_ocr_text')
                ocr_conf_col = ocr_features.get('inner_images_ocr_confidence', 'inner_images_ocr_confidence')
            else:
                ocr_text_col = 'inner_images_ocr_text'
                ocr_conf_col = 'inner_images_ocr_confidence'

            result_df[ocr_text_col] = features['inner_images_ocr_texts']
            result_df[ocr_conf_col] = features['inner_images_ocr_confidences']

        # OCR CLIP 特征（替代 LabelEncoder sparse 特征）
        if 'cover_ocr_features' in features:
            column_names = [f'cover_ocr_feat_{i}' for i in range(num_features)]
            cover_ocr_feat_df = pd.DataFrame(
                features['cover_ocr_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, cover_ocr_feat_df], axis=1)

        if 'inner_ocr_features' in features:
            column_names = [f'inner_ocr_feat_{i}' for i in range(num_features)]
            inner_ocr_feat_df = pd.DataFrame(
                features['inner_ocr_features'],
                columns=column_names,
                index=df.index
            )
            result_df = pd.concat([result_df, inner_ocr_feat_df], axis=1)

        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_path / f"multimodal_batch_{batch_id:06d}_{timestamp}.parquet"
        
        # 保存为parquet
        result_df.to_parquet(output_file, compression='snappy')
        logger.info(f"Saved batch {batch_id} to {output_file} ({result_df.shape[0]} rows, {result_df.shape[1]} columns)")
        
        return output_file


def extract_cover_image_url(row: pd.Series) -> Optional[str]:
    """提取封面图URL"""
    first_file_id = row.get('first_file_id', '')
    file_url_list = row.get('file_url_list', '')
    
    if not first_file_id or not file_url_list:
        return None
        
    try:
        file_url_list = str(file_url_list).strip()
        if file_url_list.startswith('[') and file_url_list.endswith(']'):
            try:
                urls = json.loads(file_url_list)
            except:
                file_url_list = file_url_list[1:-1]
                urls = [url.strip().strip('"\'') for url in file_url_list.split(',')]
        else:
            urls = [url.strip() for url in file_url_list.split(',')]
            
        for url in urls:
            if first_file_id in url:
                return url
                
        if urls:
            return urls[0]
    except Exception as e:
        logger.debug(f"Failed to extract cover URL: {e}")
        
    return None


def extract_inner_image_urls(row: pd.Series, max_images: int = 5) -> List[str]:
    """提取内页图片URLs（排除封面图）"""
    first_file_id = row.get('first_file_id', '')
    file_url_list = row.get('file_url_list', '')
    
    if not file_url_list:
        return []
        
    try:
        file_url_list = str(file_url_list).strip()
        if file_url_list.startswith('[') and file_url_list.endswith(']'):
            try:
                urls = json.loads(file_url_list)
            except:
                file_url_list = file_url_list[1:-1]
                urls = [url.strip().strip('"\'') for url in file_url_list.split(',')]
        else:
            urls = [url.strip() for url in file_url_list.split(',')]
            
        # 排除封面图
        inner_urls = []
        for url in urls:
            if first_file_id and first_file_id in url:
                continue  # 跳过封面图
            inner_urls.append(url)
            
        # 限制数量
        return inner_urls[:max_images]
        
    except Exception as e:
        logger.debug(f"Failed to extract inner URLs: {e}")
        
    return []


def extract_tag_names(tag_info_str: str) -> str:
    """从tag_info JSON中提取name字段"""
    if not tag_info_str:
        return ""
        
    try:
        # 清理字符串
        tag_info_str = str(tag_info_str).strip()
        
        # 尝试解析JSON
        if tag_info_str.startswith('[') and tag_info_str.endswith(']'):
            tag_list = json.loads(tag_info_str)
            tag_names = []
            
            for tag in tag_list:
                if isinstance(tag, dict) and 'name' in tag:
                    tag_names.append(tag['name'])
                    
            return ','.join(tag_names)
            
    except Exception as e:
        logger.debug(f"Failed to parse tag_info: {e}")
        
    return ""