#!/usr/bin/env python3
"""
统一模型工厂

提供所有DeepCTR模型的创建接口，支持单任务和多任务模型。

功能特性：
1. 单任务模型创建 (PNN, DeepFM, WDL, DCN, xDeepFM, AutoInt, FiBiNET)
2. 多任务模型创建 (PLE, MMOE, SharedBottom, ESMM)
3. 自动参数验证和默认值设置
4. 模型配置管理
5. 设备自适应 (CPU/GPU/MPS)
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import torch
import torch.nn as nn

# DeepCTR-Torch imports
from deepctr_torch.models import (
    PNN, DeepFM, WDL, DCN, xDeepFM, AutoInt, FiBiNET, NFM, AFM, CCPM
)
from deepctr_torch.models.multitask import (
    PLE, MMOE, SharedBottom, ESMM
)
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

from .base_config import BaseTrainingConfig
from .pnn_mmoe_model import PNN_MMOE


class SingleTaskModel(Enum):
    """单任务模型类型"""
    PNN = "PNN"
    DEEPFM = "DeepFM"
    WDL = "WDL"
    DCN = "DCN"
    XDEEPFM = "xDeepFM"
    AUTOINT = "AutoInt"
    FIBINET = "FiBiNET"
    NFM = "NFM"
    AFM = "AFM"
    CCPM = "CCPM"


class MultiTaskModel(Enum):
    """多任务模型类型"""
    PLE = "PLE"
    MMOE = "MMOE"
    PNN_MMOE = "PNN_MMOE"
    SHAREDBOTTOM = "SharedBottom"
    ESMM = "ESMM"


class ModelFactory:
    """统一模型工厂
    
    提供所有DeepCTR模型的创建接口，管理模型配置和初始化。
    """
    
    def __init__(self, config: Optional[BaseTrainingConfig] = None):
        """初始化模型工厂
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设备检测
        self.device = self._detect_device()
        self.logger.info(f"🖥️ 使用设备: {self.device}")
        
        # 单任务模型映射
        self.single_task_models = {
            SingleTaskModel.PNN: PNN,
            SingleTaskModel.DEEPFM: DeepFM,
            SingleTaskModel.WDL: WDL,
            SingleTaskModel.DCN: DCN,
            SingleTaskModel.XDEEPFM: xDeepFM,
            SingleTaskModel.AUTOINT: AutoInt,
            SingleTaskModel.FIBINET: FiBiNET,
            SingleTaskModel.NFM: NFM,
            SingleTaskModel.AFM: AFM,
            SingleTaskModel.CCPM: CCPM,
        }
        
        # 多任务模型映射
        self.multi_task_models = {
            MultiTaskModel.PLE: PLE,
            MultiTaskModel.MMOE: MMOE,
            MultiTaskModel.PNN_MMOE: PNN_MMOE,
            MultiTaskModel.SHAREDBOTTOM: SharedBottom,
            MultiTaskModel.ESMM: ESMM,
        }
    
    def _detect_device(self) -> str:
        """自动检测可用设备
        
        Returns:
            设备字符串 ('cuda', 'mps', 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def create_single_task_model(
        self, 
        model_type: Union[str, SingleTaskModel],
        feature_columns: List,
        **kwargs
    ) -> nn.Module:
        """创建单任务模型
        
        Args:
            model_type: 模型类型
            feature_columns: DeepCTR特征列定义
            **kwargs: 模型特定参数
            
        Returns:
            初始化的模型
        """
        # 类型转换
        if isinstance(model_type, str):
            try:
                model_type = SingleTaskModel(model_type.upper())
            except ValueError:
                raise ValueError(f"不支持的单任务模型类型: {model_type}. "
                               f"支持的类型: {[m.value for m in SingleTaskModel]}")
        
        # 获取模型类
        model_class = self.single_task_models[model_type]
        
        # 准备模型参数
        model_params = self._prepare_single_task_params(model_type, feature_columns, kwargs)
        
        # 创建模型
        self.logger.info(f"🏗️ 创建单任务模型: {model_type.value}")
        model = model_class(**model_params)
        
        # 移动到设备
        model = model.to(self.device)
        
        # 打印模型信息
        self._print_model_info(model, model_type.value)
        
        return model
    
    def create_multi_task_model(
        self,
        model_type: Union[str, MultiTaskModel],
        feature_columns: List,
        task_names: List[str],
        **kwargs
    ) -> nn.Module:
        """创建多任务模型
        
        Args:
            model_type: 模型类型
            feature_columns: DeepCTR特征列定义
            task_names: 任务名称列表
            **kwargs: 模型特定参数
            
        Returns:
            初始化的模型
        """
        # 类型转换
        if isinstance(model_type, str):
            try:
                model_type = MultiTaskModel(model_type.upper())
            except ValueError:
                raise ValueError(f"不支持的多任务模型类型: {model_type}. "
                               f"支持的类型: {[m.value for m in MultiTaskModel]}")
        
        # 获取模型类
        model_class = self.multi_task_models[model_type]
        
        # 准备模型参数
        model_params = self._prepare_multi_task_params(
            model_type, feature_columns, task_names, kwargs
        )
        
        # 创建模型
        self.logger.info(f"🏗️ 创建多任务模型: {model_type.value}")
        self.logger.info(f"📋 任务列表: {task_names}")
        model = model_class(**model_params)
        
        # 移动到设备
        model = model.to(self.device)
        
        # 打印模型信息
        self._print_model_info(model, f"{model_type.value} (Tasks: {len(task_names)})")
        
        return model
    
    def _prepare_single_task_params(
        self,
        model_type: SingleTaskModel,
        feature_columns: List,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备单任务模型参数
        
        Args:
            model_type: 模型类型
            feature_columns: 特征列定义
            kwargs: 用户提供的参数
            
        Returns:
            完整的模型参数字典
        """
        # 基础参数
        params = {
            'feature_columns': feature_columns,
            'task': 'regression',  # CTR是回归任务
            'device': self.device,
            'gpus': None,
            'seed': 1024,
        }
        
        # 默认DNN参数
        params.update({
            'dnn_hidden_units': kwargs.get('dnn_hidden_units', (256, 128, 64)),
            'dnn_activation': kwargs.get('dnn_activation', 'relu'),
            'dnn_dropout': kwargs.get('dnn_dropout', 0.1),
            'dnn_use_bn': kwargs.get('dnn_use_bn', True),
        })
        
        # 模型特定参数
        if model_type == SingleTaskModel.PNN:
            params['use_inner'] = kwargs.get('use_inner', True)
            params['use_outter'] = kwargs.get('use_outter', False)
            params['kernel_type'] = kwargs.get('kernel_type', 'mat')
        
        elif model_type == SingleTaskModel.WDL:
            params['dnn_hidden_units'] = kwargs.get('deep_hidden_units', (256, 128, 64))
            params['l2_reg_linear'] = kwargs.get('l2_reg_linear', 1e-5)
            params['l2_reg_dnn'] = kwargs.get('l2_reg_dnn', 0)
        
        elif model_type == SingleTaskModel.DCN:
            params['cross_num'] = kwargs.get('cross_num', 3)
            params['cross_parameterization'] = kwargs.get('cross_parameterization', 'vector')
        
        elif model_type == SingleTaskModel.XDEEPFM:
            params['cin_layer_size'] = kwargs.get('cin_layer_size', (128, 128))
            params['cin_split_half'] = kwargs.get('cin_split_half', True)
            params['cin_activation'] = kwargs.get('cin_activation', 'relu')
        
        elif model_type == SingleTaskModel.AUTOINT:
            params['att_layer_num'] = kwargs.get('att_layer_num', 3)
            params['att_head_num'] = kwargs.get('att_head_num', 2)
            params['att_res'] = kwargs.get('att_res', True)
        
        elif model_type == SingleTaskModel.FIBINET:
            params['reduction_ratio'] = kwargs.get('reduction_ratio', 3)
            params['bilinear_type'] = kwargs.get('bilinear_type', 'interaction')
        
        # 更新用户提供的参数
        params.update(kwargs)
        
        return params
    
    def _prepare_multi_task_params(
        self,
        model_type: MultiTaskModel,
        feature_columns: List,
        task_names: List[str],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备多任务模型参数
        
        Args:
            model_type: 模型类型
            feature_columns: 特征列定义
            task_names: 任务名称列表
            kwargs: 用户提供的参数
            
        Returns:
            完整的模型参数字典
        """
        num_tasks = len(task_names)
        
        # 基础参数
        if model_type == MultiTaskModel.PNN_MMOE:
            # PNN_MMOE 自定义模型参数
            params = {
                'dnn_feature_columns': feature_columns,
                'num_tasks': num_tasks,
                'task_types': ['regression'] * num_tasks,
                'task_names': task_names,
                'device': self.device,
                'seed': 1024,
            }
        else:
            # 标准DeepCTR多任务模型参数
            params = {
                'dnn_feature_columns': feature_columns,
                'task_types': ['regression'] * num_tasks,
                'task_names': task_names,
                'device': self.device,
                'gpus': None,
                'seed': 1024,
            }
        
        # 默认tower参数 - 根据模型类型设置不同格式
        if model_type == MultiTaskModel.PNN_MMOE:
            params['tower_dnn_hidden_units'] = kwargs.get('tower_dnn_hidden_units', (64, 32))
        elif model_type == MultiTaskModel.PLE:
            # PLE需要每个任务一个tuple
            params['tower_dnn_hidden_units'] = kwargs.get(
                'tower_dnn_hidden_units', 
                [(64, 32)] * num_tasks
            )
        else:
            # MMOE和其他标准模型使用单个tuple
            params['tower_dnn_hidden_units'] = kwargs.get('tower_dnn_hidden_units', (64,))
        
        # 模型特定参数
        if model_type == MultiTaskModel.PLE:
            params['shared_expert_num'] = kwargs.get('shared_expert_num', 3)
            params['specific_expert_num'] = kwargs.get('specific_expert_num', 2)
            params['num_levels'] = kwargs.get('num_levels', 2)
            params['expert_dnn_hidden_units'] = kwargs.get(
                'expert_dnn_hidden_units', (256, 128)
            )
            params['gate_dnn_hidden_units'] = kwargs.get(
                'gate_dnn_hidden_units', (64,)
            )
        
        elif model_type == MultiTaskModel.MMOE:
            params['num_experts'] = kwargs.get('num_experts', 3)
            params['expert_dnn_hidden_units'] = kwargs.get(
                'expert_dnn_hidden_units', (256, 128)
            )
            params['gate_dnn_hidden_units'] = kwargs.get(
                'gate_dnn_hidden_units', (64,)
            )
        
        elif model_type == MultiTaskModel.PNN_MMOE:
            params['num_experts'] = kwargs.get('num_experts', 3)
            params['expert_dnn_hidden_units'] = kwargs.get(
                'expert_dnn_hidden_units', (256, 128)
            )
            params['gate_dnn_hidden_units'] = kwargs.get(
                'gate_dnn_hidden_units', (64,)
            )
            params['use_inner_product'] = kwargs.get('use_inner_product', True)
            params['use_outter_product'] = kwargs.get('use_outter_product', False)
        
        elif model_type == MultiTaskModel.SHAREDBOTTOM:
            params['bottom_dnn_hidden_units'] = kwargs.get(
                'bottom_dnn_hidden_units', (256, 128)
            )
        
        elif model_type == MultiTaskModel.ESMM:
            # ESMM需要特殊处理，因为它只支持特定的任务配置
            if num_tasks != 2:
                raise ValueError(f"ESMM只支持2个任务，但提供了{num_tasks}个任务")
            params['tower_dnn_hidden_units'] = kwargs.get(
                'tower_dnn_hidden_units', [(64, 32), (64, 32)]
            )
        
        # 更新用户提供的参数
        params.update(kwargs)
        
        return params
    
    def _print_model_info(self, model: nn.Module, model_name: str):
        """打印模型信息
        
        Args:
            model: 模型实例
            model_name: 模型名称
        """
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 模型信息 - {model_name}:")
        self.logger.info(f"  - 总参数量: {total_params:,}")
        self.logger.info(f"  - 可训练参数量: {trainable_params:,}")
        self.logger.info(f"  - 设备: {self.device}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取所有可用的模型类型
        
        Returns:
            包含单任务和多任务模型列表的字典
        """
        return {
            'single_task': [m.value for m in SingleTaskModel],
            'multi_task': [m.value for m in MultiTaskModel]
        }
    
    def create_model_from_config(self, config: BaseTrainingConfig) -> nn.Module:
        """从配置直接创建模型
        
        Args:
            config: 训练配置
            
        Returns:
            初始化的模型
        """
        self.config = config
        
        # 根据配置类型创建模型
        if hasattr(config, 'multi_task_model'):
            # 多任务模型
            return self.create_multi_task_model(
                model_type=config.multi_task_model,
                feature_columns=config.feature_columns,
                task_names=config.task_names,
                **config.model_params
            )
        else:
            # 单任务模型
            return self.create_single_task_model(
                model_type=config.model_type,
                feature_columns=config.feature_columns,
                **config.model_params
            )


def create_model(
    model_type: str,
    feature_columns: List,
    task: str = 'single',
    task_names: Optional[List[str]] = None,
    **kwargs
) -> nn.Module:
    """便捷函数：创建模型
    
    Args:
        model_type: 模型类型
        feature_columns: 特征列定义
        task: 'single' 或 'multi'
        task_names: 任务名称列表（多任务时必需）
        **kwargs: 模型参数
        
    Returns:
        初始化的模型
        
    Example:
        >>> model = create_model(
        ...     'DeepFM', 
        ...     feature_columns,
        ...     task='single',
        ...     dnn_hidden_units=(256, 128)
        ... )
    """
    factory = ModelFactory()
    
    if task == 'single':
        return factory.create_single_task_model(model_type, feature_columns, **kwargs)
    elif task == 'multi':
        if task_names is None:
            raise ValueError("多任务模型必须提供task_names参数")
        return factory.create_multi_task_model(model_type, feature_columns, task_names, **kwargs)
    else:
        raise ValueError(f"不支持的任务类型: {task}. 应该是 'single' 或 'multi'")


# 导出主要接口
__all__ = [
    'ModelFactory',
    'SingleTaskModel',
    'MultiTaskModel',
    'create_model',
]