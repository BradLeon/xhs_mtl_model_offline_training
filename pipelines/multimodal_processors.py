#!/usr/bin/env python3
"""
多模态特征处理器模块
从run_local_image_pipeline_chinese-clip_for_PLE.py迁移而来
包含所有核心处理器类
"""

import os
import asyncio
import logging
from typing import List, Optional, Tuple
from io import BytesIO

import numpy as np
import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name
from PIL import Image
import aiohttp
import easyocr
import gc

from .multimodal_config import MultimodalConfig

logger = logging.getLogger(__name__)


def cleanup_memory():
    """强制清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        # MPS内存管理
        torch.mps.empty_cache()
    logger.debug(f"Memory cleanup completed in process {os.getpid()}")


class ImageDownloader:
    """异步图片下载器"""
    
    def __init__(self, num_workers: int = 10, timeout: int = 30):
        self.num_workers = num_workers
        self.timeout = timeout
        self.session = None
        
    async def _download_image(self, url: str) -> Optional[bytes]:
        """下载单张图片"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            
            if proxy:
                logger.debug(f"Using proxy: {proxy}")
            self.session = aiohttp.ClientSession(timeout=timeout)
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            
            async with self.session.get(url, headers=headers, proxy=proxy) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.debug(f"Failed to download {url}: HTTP {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Failed to download {url}: Timeout after {self.timeout}s")
        except aiohttp.ClientError as e:
            logger.debug(f"Failed to download {url}: {type(e).__name__}: {e}")
        except Exception as e:
            logger.debug(f"Failed to download {url}: Unexpected error: {e}")
        return None
        
    async def download_batch(self, urls: List[str]) -> List[Optional[bytes]]:
        """批量下载图片"""
        tasks = [self._download_image(url) for url in urls]
        return await asyncio.gather(*tasks)
        
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()


class ChineseCLIPProcessor:
    """Chinese-CLIP特征提取器"""
    
    def __init__(self, model_name: str = "ViT-B-16", batch_size: int = 16, target_dim: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.target_dim = target_dim
        self.logger = logging.getLogger(__name__)
        self.device = self._get_device()
        self.model, self.preprocess = self._load_model()
        
    def _get_device(self):
        """获取最佳设备"""
        if torch.backends.mps.is_available():
            self.logger.info("Using Apple Silicon GPU (MPS)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            self.logger.info("Using NVIDIA GPU (CUDA)")
            return torch.device("cuda")
        else:
            self.logger.info("Using CPU")
            return torch.device("cpu")
            
    def _load_model(self):
        """加载Chinese-CLIP模型"""
        self.logger.info(f"Loading Chinese-CLIP model: {self.model_name}")
        
        model, preprocess = load_from_name(self.model_name, device=self.device, download_root='./cache')
        model.eval()
        
        self.logger.info(f"✅ Chinese-CLIP model loaded: {self.model_name}")
        return model, preprocess
        
    def process_cover_images(self, images: List[bytes]) -> np.ndarray:
        """处理封面图片"""
        image_tensors = []
        for img_bytes in images:
            if img_bytes:
                try:
                    img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    tensor = self.preprocess(img)
                    image_tensors.append(tensor)
                except Exception as e:
                    self.logger.debug(f"Failed to process cover image: {e}")
                    image_tensors.append(torch.zeros(3, 224, 224))
            else:
                image_tensors.append(torch.zeros(3, 224, 224))
                
        if image_tensors:
            image_batch = torch.stack(image_tensors).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy()
        else:
            return np.zeros((len(images), self.target_dim))
            
    def process_inner_images_batch(self, batch_inner_images: List[List[bytes]], 
                                 pooling_strategy: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
        """批量处理内页图片并池化
        
        Args:
            batch_inner_images: 每个样本的内页图片列表
            pooling_strategy: 池化策略 (mean/max)
            
        Returns:
            (pooled_features, num_images_array)
        """
        pooled_features = []
        num_images_array = []
        
        for inner_images in batch_inner_images:
            num_images = len([img for img in inner_images if img])
            num_images_array.append(num_images)
            
            if num_images == 0:
                # 没有内页图片，使用零向量
                pooled_features.append(np.zeros(self.target_dim))
                continue
                
            # 处理内页图片
            image_tensors = []
            for img_bytes in inner_images:
                if img_bytes:
                    try:
                        img = Image.open(BytesIO(img_bytes)).convert('RGB')
                        tensor = self.preprocess(img)
                        image_tensors.append(tensor)
                    except Exception as e:
                        self.logger.debug(f"Failed to process inner image: {e}")
                        continue
            
            if image_tensors:
                image_batch = torch.stack(image_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.model.encode_image(image_batch)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_np = features.cpu().numpy()
                    
                    # 池化
                    if pooling_strategy == "mean":
                        pooled_feat = np.mean(features_np, axis=0)
                    elif pooling_strategy == "max":
                        pooled_feat = np.max(features_np, axis=0)
                    else:
                        # 默认使用mean
                        pooled_feat = np.mean(features_np, axis=0)
                    
                    pooled_features.append(pooled_feat)
            else:
                pooled_features.append(np.zeros(self.target_dim))
        
        return np.array(pooled_features), np.array(num_images_array)
        
    def process_texts(self, texts: List[str]) -> np.ndarray:
        """处理文本（支持批量）"""
        if not texts:
            return np.zeros((0, self.target_dim))
            
        # 过滤空文本
        processed_texts = []
        for text in texts:
            text = str(text).strip() if text else ""
            if not text:
                text = " "  # 避免空字符串导致的问题
            processed_texts.append(text)
            
        # Chinese-CLIP文本编码
        text_tokens = clip.tokenize(processed_texts, context_length=52).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()
            
    def process_long_content(self, content_list: List[str], max_length: int = 200, 
                           chunk_size: int = 52, max_chunks: int = 4) -> np.ndarray:
        """处理长内容文本（分块+聚合）"""
        features_list = []
        
        for content in content_list:
            content = str(content).strip() if content else ""
            
            if not content:
                # 空内容
                features_list.append(np.zeros(self.target_dim))
                continue
                
            # 截断或分块
            if len(content) <= max_length:
                # 短内容，直接编码
                text_tokens = clip.tokenize([content], context_length=52).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_text(text_tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_list.append(features.cpu().numpy()[0])
            else:
                # 长内容，分块处理
                chunks = []
                for i in range(0, min(len(content), max_length), chunk_size):
                    chunk = content[i:i + chunk_size]
                    chunks.append(chunk)
                    if len(chunks) >= max_chunks:
                        break
                
                if chunks:
                    # 批量编码分块
                    text_tokens = clip.tokenize(chunks, context_length=52).to(self.device)
                    with torch.no_grad():
                        chunk_features = self.model.encode_text(text_tokens)
                        chunk_features = chunk_features / chunk_features.norm(dim=-1, keepdim=True)
                        # 平均池化
                        avg_features = torch.mean(chunk_features, dim=0)
                        features_list.append(avg_features.cpu().numpy())
                else:
                    features_list.append(np.zeros(self.target_dim))
        
        return np.array(features_list)


class OCRProcessor:
    """OCR处理器"""
    
    def __init__(self):
        try:
            use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
            self.enabled = True
            logger.info(f"EasyOCR initialized, GPU: {use_gpu}")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.reader = None
            self.enabled = False
            
    def extract_text(self, image_bytes: bytes) -> Tuple[str, float]:
        """从图片提取文本"""
        if not self.enabled or not image_bytes:
            return "", 0.0
            
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img_array = np.array(img)
            
            results = self.reader.readtext(img_array, detail=1)
            
            if results:
                texts = []
                confidences = []
                for (_, text, conf) in results:
                    if conf > 0.3:  # 置信度阈值
                        texts.append(text)
                        confidences.append(conf)
                        
                ocr_text = ' '.join(texts)[:200]  # 限制长度
                avg_confidence = np.mean(confidences) if confidences else 0.0
                return ocr_text, avg_confidence
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            
        return "", 0.0
    
    def extract_batch_texts(self, image_bytes_list: List[bytes]) -> Tuple[List[str], List[float]]:
        """批量提取OCR文本"""
        ocr_texts = []
        ocr_confidences = []
        
        if self.enabled:
            for img_bytes in image_bytes_list:
                text, conf = self.extract_text(img_bytes)
                ocr_texts.append(text)
                ocr_confidences.append(conf)
        else:
            ocr_texts = [''] * len(image_bytes_list)
            ocr_confidences = [0.0] * len(image_bytes_list)
            
        return ocr_texts, ocr_confidences
    
    def extract_inner_images_ocr(self, inner_images_batch: List[List[bytes]]) -> Tuple[List[str], List[float]]:
        """提取内页图片OCR文本（聚合多张图片的结果）"""
        aggregated_texts = []
        aggregated_confidences = []
        
        for inner_images in inner_images_batch:
            if not inner_images or not self.enabled:
                aggregated_texts.append("")
                aggregated_confidences.append(0.0)
                continue
                
            # 提取所有内页图的OCR
            all_texts = []
            all_confidences = []
            
            for img_bytes in inner_images:
                if img_bytes:
                    text, conf = self.extract_text(img_bytes)
                    if text:  # 只添加非空文本
                        all_texts.append(text)
                        all_confidences.append(conf)
            
            # 聚合结果
            if all_texts:
                combined_text = ' '.join(all_texts)[:500]  # 限制总长度
                max_confidence = max(all_confidences)  # 使用最高置信度
            else:
                combined_text = ""
                max_confidence = 0.0
                
            aggregated_texts.append(combined_text)
            aggregated_confidences.append(max_confidence)
        
        return aggregated_texts, aggregated_confidences


# 全局变量用于进程内CLIP模型缓存
_clip_processor_cache = None

def get_clip_processor(config: MultimodalConfig) -> ChineseCLIPProcessor:
    """获取或创建进程内缓存的CLIP处理器"""
    global _clip_processor_cache
    
    if _clip_processor_cache is None:
        logger.info(f"Creating CLIP processor in process {os.getpid()}")
        _clip_processor_cache = ChineseCLIPProcessor(
            model_name=config.model_name,
            batch_size=config.gpu_batch_size, 
            target_dim=config.target_dim
        )
        logger.info(f"✅ CLIP processor cached in process {os.getpid()}")
    
    return _clip_processor_cache