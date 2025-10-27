#!/usr/bin/env python3
"""
轻量级特征规范工具

提供简单的特征命名规范管理，确保pipeline和training使用一致的特征名。
只包含必要的功能，避免过度复杂化。
"""

import os
import yaml
import logging
from typing import Dict, List, Optional
from pathlib import Path


class FeatureSpec:
    """特征规范管理器
    
    从 config/features.yaml 读取特征命名规范，
    提供简单的特征名生成和验证功能。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化特征规范
        
        Args:
            config_path: 配置文件路径，默认为 config/features.yaml
        """
        if config_path is None:
            # 自动查找项目根目录下的配置文件
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "features.yaml"
        
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # 加载规范
        self._load_spec()
    
    def _load_spec(self):
        """加载特征规范配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.spec = yaml.safe_load(f)
            self.logger.info(f"✅ 加载特征规范: {self.config_path}")
        except FileNotFoundError:
            self.logger.error(f"❌ 特征规范文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"❌ 特征规范文件格式错误: {e}")
            raise
    
    def get_clip_pattern(self, feature_type: str) -> str:
        """获取CLIP特征命名模式
        
        Args:
            feature_type: 特征类型 ('cover_image', 'inner_image', 'title', 'content', 'tag')
            
        Returns:
            特征命名模式字符串
            
        Example:
            >>> spec.get_clip_pattern('cover_image')
            'cover_image_feat_'
        """
        if feature_type not in self.spec['clip_features']:
            available_types = list(self.spec['clip_features'].keys())
            raise ValueError(f"不支持的CLIP特征类型: {feature_type}. "
                           f"支持的类型: {available_types}")
        
        return self.spec['clip_features'][feature_type]
    
    def generate_clip_names(self, feature_type: str, num_dimensions: int) -> List[str]:
        """生成CLIP特征名列表
        
        Args:
            feature_type: 特征类型
            num_dimensions: 特征维度数
            
        Returns:
            特征名列表
            
        Example:
            >>> spec.generate_clip_names('cover_image', 3)
            ['cover_image_feat_0', 'cover_image_feat_1', 'cover_image_feat_2']
        """
        pattern = self.get_clip_pattern(feature_type)
        return [f"{pattern}{i}" for i in range(num_dimensions)]
    
    def get_all_clip_patterns(self) -> List[str]:
        """获取所有CLIP特征模式
        
        Returns:
            所有CLIP特征模式列表
        """
        return list(self.spec['clip_features'].values())
    
    def get_metadata_features(self) -> Dict[str, str]:
        """获取元数据特征规范"""
        return self.spec.get('metadata_features', {})
    
    def get_ocr_features(self) -> Dict[str, str]:
        """获取OCR特征规范"""
        return self.spec.get('ocr_features', {})
    
    def get_basic_features(self) -> Dict[str, List[str]]:
        """获取基础特征规范"""
        return self.spec.get('basic_features', {'sparse': [], 'dense': []})
    
    def is_clip_feature(self, feature_name: str) -> bool:
        """检查是否为CLIP特征
        
        Args:
            feature_name: 特征名
            
        Returns:
            是否为CLIP特征
        """
        for pattern in self.get_all_clip_patterns():
            if feature_name.startswith(pattern):
                return True
        return False
    
    def get_clip_feature_type(self, feature_name: str) -> Optional[str]:
        """获取CLIP特征类型
        
        Args:
            feature_name: 特征名
            
        Returns:
            特征类型，如果不是CLIP特征则返回None
        """
        for feature_type, pattern in self.spec['clip_features'].items():
            if feature_name.startswith(pattern):
                return feature_type
        return None
    
    def validate_feature_names(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """验证特征名是否符合规范
        
        Args:
            feature_names: 特征名列表
            
        Returns:
            验证结果字典，包含违规的特征名
        """
        result = {
            'deprecated_features': [],
            'unknown_clip_features': []
        }
        
        deprecated_patterns = self.spec.get('deprecated_patterns', [])
        
        for feature_name in feature_names:
            # 检查是否使用了废弃模式
            for pattern in deprecated_patterns:
                if feature_name.startswith(pattern):
                    result['deprecated_features'].append(feature_name)
                    break
            else:
                # 检查是否为无效的CLIP特征
                if any(pattern in feature_name for pattern in deprecated_patterns):
                    if not self.is_clip_feature(feature_name):
                        result['unknown_clip_features'].append(feature_name)
        
        return result
    
    def print_summary(self):
        """打印特征规范摘要"""
        print("📋 特征命名规范摘要:")
        print(f"  配置文件: {self.config_path}")
        
        print("\n🎯 CLIP特征模式:")
        for feature_type, pattern in self.spec['clip_features'].items():
            print(f"  - {feature_type}: {pattern}{{index}}")
        
        basic = self.get_basic_features()
        print(f"\n📊 基础特征: {len(basic['sparse'])} 稀疏 + {len(basic['dense'])} 密集")
        
        metadata = self.get_metadata_features()
        ocr = self.get_ocr_features()
        print(f"📁 其他特征: {len(metadata)} 元数据 + {len(ocr)} OCR")


# 全局实例（延迟初始化）
_feature_spec = None


def get_feature_spec() -> FeatureSpec:
    """获取全局特征规范实例
    
    Returns:
        FeatureSpec实例
    """
    global _feature_spec
    if _feature_spec is None:
        _feature_spec = FeatureSpec()
    return _feature_spec


# 便捷函数
def generate_clip_features(feature_type: str, num_dimensions: int) -> List[str]:
    """便捷函数：生成CLIP特征名"""
    return get_feature_spec().generate_clip_names(feature_type, num_dimensions)


def validate_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """便捷函数：验证特征名"""
    return get_feature_spec().validate_feature_names(feature_names)


def is_valid_feature_naming(feature_names: List[str]) -> bool:
    """便捷函数：检查特征命名是否有效"""
    validation = validate_features(feature_names)
    return len(validation['deprecated_features']) == 0


if __name__ == "__main__":
    # 测试代码
    spec = FeatureSpec()
    spec.print_summary()
    
    # 测试生成特征名
    cover_features = spec.generate_clip_names('cover_image', 5)
    print(f"\n🧪 测试生成封面图特征: {cover_features}")
    
    # 测试验证
    test_features = ['cover_image_feat_0', 'cover_image_embedding_1', 'title_feat_0']
    validation = spec.validate_feature_names(test_features)
    print(f"\n🔍 测试验证结果: {validation}")