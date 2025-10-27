#!/usr/bin/env python3
"""
特征选择器

提供预设的特征配置和选择逻辑，支持单任务训练器的特征工程。

支持的特征配置：
- basic: 基础特征（无CLIP）
- clip_only: 仅CLIP特征
- no_image: 无图像特征
- no_text: 无文本特征
- all: 所有特征
- default: 默认配置
- custom: 自定义配置
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureSelector:
    """特征选择器
    
    提供预设的特征配置和特征选择逻辑，支持多种特征组合策略。
    """
    
    @staticmethod
    def get_feature_config(config_name: str = 'default') -> Dict:
        """获取预设的特征配置
        
        Args:
            config_name: 配置名称 (basic/clip_only/all/custom/default/no_image/no_text)
            
        Returns:
            特征配置字典
        """
        configs = {
            'basic': {
                'sparse': ['nickname', 'user_id', 'title_clean', 'ocr_text', 'type'],
                'dense': ['title_length'],
                'clip_image': False,
                'clip_text': False,
                'description': '基础特征（无CLIP）'
            },
            'clip_only': {
                'sparse': [],
                'dense': [],
                'clip_image': True,  # image_feat_0 到 image_feat_511
                'clip_text': True,   # text_feat_0 到 text_feat_511
                'description': '仅CLIP特征'
            },
            'no_image': {
                'sparse': ['nickname', 'user_id', 'title_clean', 'ocr_text', 'type'],
                'dense': ['title_length'],
                'clip_image': False,
                'clip_text': True,
                'description': '无图像特征'
            },
            'no_text': {
                'sparse': ['nickname', 'user_id', 'title_clean', 'ocr_text', 'type'],
                'dense': ['title_length'],
                'clip_image': True,
                'clip_text': False,
                'description': '无文本特征'
            },
            'default': {
                'sparse': ['nickname', 'user_id', 'title_clean', 'ocr_text', 'type'],
                'dense': ['title_length'],
                'clip_image': True,
                'clip_text': True,
                'description': '默认配置（基础+CLIP）'
            },
            'all': {
                'sparse': ['nickname', 'user_id', 'title_clean', 'ocr_text', 'type', 
                          'intention_lv1', 'taxonomy1', 'taxonomy2'],
                'dense': ['title_length'],
                'clip_image': True,
                'clip_text': True,
                'description': '所有特征'
            }
        }
        
        config = configs.get(config_name, configs['default'])
        logger.debug(f"Using feature config '{config_name}': {config.get('description', 'Unknown')}")
        
        return config
    
    @staticmethod
    def select_features(df: pd.DataFrame, config: Dict) -> Tuple[List[str], List[str], List[str]]:
        """根据配置选择特征
        
        Args:
            df: 数据框
            config: 特征配置
            
        Returns:
            (稀疏特征列表, 密集特征列表, CLIP特征列表)
        """
        sparse_features = []
        dense_features = []
        clip_features = []
        
        # 选择稀疏特征
        for feat in config.get('sparse', []):
            if feat in df.columns:
                sparse_features.append(feat)
            else:
                logger.warning(f"Sparse feature '{feat}' not found in dataframe")
        
        # 选择密集特征
        for feat in config.get('dense', []):
            if feat in df.columns:
                dense_features.append(feat)
            else:
                logger.warning(f"Dense feature '{feat}' not found in dataframe")
        
        # 选择CLIP特征
        if config.get('clip_image', False):
            image_feats = [col for col in df.columns if col.startswith('image_feat_')]
            clip_features.extend(image_feats)
            logger.debug(f"Selected {len(image_feats)} image CLIP features")
        
        if config.get('clip_text', False):
            text_feats = [col for col in df.columns if col.startswith('text_feat_')]
            clip_features.extend(text_feats)
            logger.debug(f"Selected {len(text_feats)} text CLIP features")
        
        # 记录特征选择结果
        logger.info(f"📊 Feature selection results:")
        logger.info(f"   - Sparse features: {len(sparse_features)} selected")
        logger.info(f"   - Dense features: {len(dense_features)} selected")
        logger.info(f"   - CLIP features: {len(clip_features)} selected")
        
        return sparse_features, dense_features, clip_features
    
    @staticmethod
    def create_custom_config(sparse_features: List[str] = None,
                           dense_features: List[str] = None,
                           use_clip_image: bool = False,
                           use_clip_text: bool = False) -> Dict:
        """创建自定义特征配置
        
        Args:
            sparse_features: 稀疏特征列表
            dense_features: 密集特征列表
            use_clip_image: 是否使用CLIP图像特征
            use_clip_text: 是否使用CLIP文本特征
            
        Returns:
            自定义特征配置字典
        """
        config = {
            'sparse': sparse_features or [],
            'dense': dense_features or [],
            'clip_image': use_clip_image,
            'clip_text': use_clip_text,
            'description': '自定义配置'
        }
        
        logger.info(f"Created custom feature config:")
        logger.info(f"   - Sparse: {config['sparse']}")
        logger.info(f"   - Dense: {config['dense']}")
        logger.info(f"   - CLIP Image: {config['clip_image']}")
        logger.info(f"   - CLIP Text: {config['clip_text']}")
        
        return config
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """获取所有可用的预设配置名称
        
        Returns:
            配置名称列表
        """
        return ['basic', 'clip_only', 'no_image', 'no_text', 'default', 'all', 'custom']
    
    @staticmethod
    def describe_config(config_name: str) -> str:
        """获取配置描述
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置描述字符串
        """
        config = FeatureSelector.get_feature_config(config_name)
        return config.get('description', f'Unknown config: {config_name}')
    
    @staticmethod
    def validate_config(config: Dict, df: pd.DataFrame) -> Dict[str, any]:
        """验证特征配置的有效性
        
        Args:
            config: 特征配置字典
            df: 数据框
            
        Returns:
            验证结果字典
        """
        validation_report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'missing_features': [],
            'available_features': {
                'sparse': 0,
                'dense': 0,
                'clip_image': 0,
                'clip_text': 0
            }
        }
        
        # 检查稀疏特征
        for feat in config.get('sparse', []):
            if feat not in df.columns:
                validation_report['missing_features'].append(('sparse', feat))
                validation_report['issues'].append(f"Sparse feature '{feat}' not found")
            else:
                validation_report['available_features']['sparse'] += 1
        
        # 检查密集特征
        for feat in config.get('dense', []):
            if feat not in df.columns:
                validation_report['missing_features'].append(('dense', feat))
                validation_report['issues'].append(f"Dense feature '{feat}' not found")
            else:
                validation_report['available_features']['dense'] += 1
        
        # 检查CLIP特征
        if config.get('clip_image', False):
            image_feats = [col for col in df.columns if col.startswith('image_feat_')]
            validation_report['available_features']['clip_image'] = len(image_feats)
            if not image_feats:
                validation_report['issues'].append("CLIP image features requested but not found")
        
        if config.get('clip_text', False):
            text_feats = [col for col in df.columns if col.startswith('text_feat_')]
            validation_report['available_features']['clip_text'] = len(text_feats)
            if not text_feats:
                validation_report['issues'].append("CLIP text features requested but not found")
        
        # 检查是否有任何特征可用
        total_available = sum(validation_report['available_features'].values())
        if total_available == 0:
            validation_report['valid'] = False
            validation_report['issues'].append("No features available with current configuration")
        
        # 警告
        if total_available < 5:
            validation_report['warnings'].append(f"Only {total_available} features available, may affect model performance")
        
        if validation_report['issues']:
            validation_report['valid'] = False
        
        return validation_report
    
    @staticmethod
    def auto_detect_features(df: pd.DataFrame) -> Dict:
        """自动检测数据框中可用的特征并生成配置
        
        Args:
            df: 数据框
            
        Returns:
            自动检测的特征配置
        """
        logger.info("🔍 Auto-detecting available features...")
        
        all_columns = set(df.columns)
        
        # 排除目标变量和ID列
        exclude_patterns = {
            'ctr', 'click', 'imp_num', 'impression', 'click_num',
            'note_id', 'user_id'
        }
        
        # 可能的稀疏特征
        sparse_candidates = [
            'nickname', 'title_clean', 'type', 'ocr_text',
            'intention_lv1', 'intention_lv2', 
            'taxonomy1', 'taxonomy2', 'taxonomy3'
        ]
        
        # 可能的密集特征
        dense_candidates = [
            'title_length', 'content_length', 'tag_count'
        ]
        
        # 检测可用特征
        available_sparse = [feat for feat in sparse_candidates if feat in all_columns]
        available_dense = [feat for feat in dense_candidates if feat in all_columns]
        
        # 检测CLIP特征
        has_image_feats = any(col.startswith('image_feat_') for col in all_columns)
        has_text_feats = any(col.startswith('text_feat_') for col in all_columns)
        
        image_feat_count = len([col for col in all_columns if col.startswith('image_feat_')])
        text_feat_count = len([col for col in all_columns if col.startswith('text_feat_')])
        
        auto_config = {
            'sparse': available_sparse,
            'dense': available_dense,
            'clip_image': has_image_feats,
            'clip_text': has_text_feats,
            'description': 'Auto-detected configuration'
        }
        
        logger.info("✅ Auto-detection results:")
        logger.info(f"   - Sparse features: {len(available_sparse)} found")
        logger.info(f"   - Dense features: {len(available_dense)} found")
        logger.info(f"   - CLIP image features: {image_feat_count} found")
        logger.info(f"   - CLIP text features: {text_feat_count} found")
        
        return auto_config
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """合并两个特征配置
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in ['sparse', 'dense']:
                # 对于列表类型，合并去重
                existing = set(merged.get(key, []))
                new_items = set(value) if isinstance(value, list) else set()
                merged[key] = list(existing | new_items)
            else:
                # 对于其他类型，直接覆盖
                merged[key] = value
        
        merged['description'] = f"Merged configuration ({base_config.get('description', 'Unknown')} + {override_config.get('description', 'Override')})"
        
        return merged