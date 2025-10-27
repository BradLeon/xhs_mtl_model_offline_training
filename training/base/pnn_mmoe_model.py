#!/usr/bin/env python3
"""
PNN-MMOE混合架构模型定义

结合PNN的特征交互能力和MMOE的多任务学习优势的自定义模型。

架构特点：
1. PNN Product Layer：对embedding进行内积/外积特征交互，增强特征表示
2. MMOE架构：多专家网络自动学习任务间的特征共享与独立性
3. 特征增强：将原始特征与product特征拼接，形成增强的输入
4. 多任务学习：支持多个回归任务同时训练

架构流程：
    [Sparse Features] → [Embedding] → [PNN Product Layer] ↘
                                                          → [Concat] → [MMOE] → [Tasks]
    [Dense Features] → [StandardScaler] ─────────────────↗
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepctr_torch.inputs import SparseFeat, DenseFeat, combined_dnn_input
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.layers import InnerProductLayer, OutterProductLayer, DNN, PredictionLayer, concat_fun

logger = logging.getLogger(__name__)


class PNN_MMOE(BaseModel):
    """自定义PNN-MMOE混合架构模型
    
    结合PNN的特征交互能力和MMOE的多任务学习架构。
    """
    
    def __init__(self,
                 dnn_feature_columns,
                 num_tasks,
                 task_types,
                 task_names=None,
                 num_experts=3,
                 expert_dnn_hidden_units=(128, 64),
                 gate_dnn_hidden_units=(64,),
                 tower_dnn_hidden_units=(64, 32),
                 use_inner_product=True,
                 use_outter_product=False,
                 l2_reg_embedding=1e-5,
                 l2_reg_dnn=0,
                 init_std=0.0001,
                 seed=1024,
                 dnn_dropout=0.1,
                 dnn_activation='relu',
                 device='cpu'):
        """初始化PNN-MMOE混合架构
        
        Args:
            dnn_feature_columns: 特征列定义
            num_tasks: 任务数量
            task_types: 任务类型列表
            task_names: 任务名称列表
            num_experts: 专家网络数量
            expert_dnn_hidden_units: 专家网络隐藏单元
            gate_dnn_hidden_units: 门控网络隐藏单元
            tower_dnn_hidden_units: 塔网络隐藏单元
            use_inner_product: 是否使用内积
            use_outter_product: 是否使用外积
            l2_reg_embedding: 嵌入层L2正则化
            l2_reg_dnn: DNN层L2正则化
            init_std: 初始化标准差
            seed: 随机种子
            dnn_dropout: Dropout率
            dnn_activation: 激活函数
            device: 设备
        """
        
        super(PNN_MMOE, self).__init__([], dnn_feature_columns,
                                       l2_reg_embedding=l2_reg_embedding,
                                       init_std=init_std, seed=seed, device=device)
        
        self.num_tasks = num_tasks
        self.task_types = task_types
        self.task_names = task_names if task_names else [f"task_{i}" for i in range(num_tasks)]
        self.num_experts = num_experts
        
        # 获取特征信息
        self.sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
        self.dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))
        
        # 计算embedding维度
        num_fields = len(self.sparse_feature_columns)
        if num_fields > 0:
            self.embed_dim = self.sparse_feature_columns[0].embedding_dim
        else:
            self.embed_dim = 0

        # PNN Product层设置
        self.use_inner = use_inner_product
        self.use_outter = use_outter_product
        
        if self.use_inner and num_fields > 0:
            self.innerproduct = InnerProductLayer(reduce_sum=True, device=device)
            logger.info("✅ Added InnerProductLayer for feature interaction")
        
        if self.use_outter and num_fields > 0:
            self.outterproduct = OutterProductLayer(
                field_size=num_fields,
                embedding_size=self.embed_dim,
                kernel_type='mat',
                seed=seed,
                device=device
            )
            logger.info("✅ Added OutterProductLayer for feature interaction")
        
        # 计算PNN增强后的特征维度
        linear_signal_dim = sum([feat.embedding_dim for feat in self.sparse_feature_columns])
        product_layer_dim = linear_signal_dim
        
        if self.use_inner and num_fields > 0:
            inner_product_dim = num_fields * (num_fields - 1) // 2
            product_layer_dim += inner_product_dim
            logger.info(f"Added inner product dimension: {inner_product_dim}")
        
        if self.use_outter and num_fields > 0:
            outter_product_dim = self.embed_dim
            product_layer_dim += outter_product_dim
            logger.info(f"Added outter product dimension: {outter_product_dim}")
        
        # 加上dense features的维度
        dense_feature_dim = len(self.dense_feature_columns)
        dnn_input_dim = product_layer_dim + dense_feature_dim
        
        logger.info(f"Enhanced input dimension after PNN: {dnn_input_dim}")
        
        # 创建专家网络
        self.expert_networks = nn.ModuleList([
            DNN(dnn_input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                init_std=init_std, device=device)
            for _ in range(num_experts)
        ])
        
        # 创建门控网络
        self.gate_networks = nn.ModuleList([
            DNN(dnn_input_dim, gate_dnn_hidden_units + (num_experts,),
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=False,
                init_std=init_std, device=device)
            for _ in range(num_tasks)
        ])
        
        # 创建任务塔
        self.tower_networks = nn.ModuleList([
            DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units,
                activation=dnn_activation, l2_reg=l2_reg_dnn,
                dropout_rate=dnn_dropout, use_bn=False,
                init_std=init_std, device=device)
            for _ in range(num_tasks)
        ])
        
        # 输出层
        self.out = nn.ModuleList([
            PredictionLayer(task_type) if tower_dnn_hidden_units[-1] == 1
            else nn.Sequential(
                nn.Linear(tower_dnn_hidden_units[-1], 1),
                PredictionLayer(task_type)
            )
            for task_type in task_types
        ])
        
        self.to(device)
    
    def forward(self, X):
        """前向传播"""
        # 获取稀疏和密集特征输入
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.dnn_feature_columns, self.embedding_dict
        )
        
        # PNN Product Layer处理
        if len(sparse_embedding_list) > 0 and (self.use_inner or self.use_outter):
            # 计算linear signal
            linear_signal = torch.flatten(concat_fun(sparse_embedding_list), start_dim=1)
            
            # 计算product components
            product_components = [linear_signal]
            
            if self.use_inner:
                inner_product = torch.flatten(self.innerproduct(sparse_embedding_list), start_dim=1)
                product_components.append(inner_product)
            
            if self.use_outter:
                outer_product = torch.flatten(self.outterproduct(sparse_embedding_list), start_dim=1)
                product_components.append(outer_product)
            
            # 组合product layer
            product_layer = torch.cat(product_components, dim=1)
            
            # 与dense features合并
            dnn_input = combined_dnn_input([product_layer], dense_value_list)
            
        else:
            # 使用标准方式
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        
        # MMOE处理
        expert_outputs = []
        for expert in self.expert_networks:
            expert_out = expert(dnn_input)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 每个任务的门控和输出
        task_outputs = []
        for i, (gate, tower) in enumerate(zip(self.gate_networks, self.tower_networks)):
            gate_out = gate(dnn_input)
            gate_out = F.softmax(gate_out, dim=-1).unsqueeze(1)
            
            weighted_expert = torch.matmul(gate_out, expert_outputs).squeeze(1)
            tower_out = tower(weighted_expert)
            task_out = self.out[i](tower_out)
            task_outputs.append(task_out)
        
        outputs = torch.cat(task_outputs, dim=1)
        
        return outputs