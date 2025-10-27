#!/usr/bin/env python3
"""
MMOE (Multi-gate Mixture-of-Experts) 多任务学习训练器

基于统一基础架构的轻量级MMOE训练器实现。

MMOE模型特点：
- 多门控专家混合架构
- 每个任务有独立的门控网络
- 专家网络在任务间共享，门控网络自动学习专家权重分配
- 平衡任务间的知识共享与冲突

使用示例：
----------
# 1. 基础训练
python training/multi_task/mmoe_trainer.py \
    --input-path /data/features \
    --epochs 30

# 2. 自定义MMOE架构
python training/multi_task/mmoe_trainer.py \
    --input-path /data/features \
    --num-experts 6 \
    --expert-dims 256,128 \
    --gate-dims 64 \
    --epochs 20

# 3. 训练特定任务子集
python training/multi_task/mmoe_trainer.py \
    --tasks ctr,like_rate,comment_rate \
    --epochs 25
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List
import torch.nn as nn

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入基础架构
from training.base.base_trainer import BaseMTLTrainer
from training.base.base_config import create_config_from_args

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MMOETrainer(BaseMTLTrainer):
    """MMOE多任务学习训练器
    
    继承BaseMTLTrainer，专注于MMOE模型特定的配置和创建逻辑。
    所有通用的训练流程（数据处理、特征工程、训练循环等）都由基类处理。
    """
    
    def create_model(self, feature_columns: List) -> nn.Module:
        """创建MMOE模型
        
        Args:
            feature_columns: DeepCTR特征列定义
            
        Returns:
            初始化的MMOE模型
        """
        logger.info("Creating MMOE (Multi-gate Mixture-of-Experts) model...")
        
        model = self.model_factory.create_multi_task_model(
            model_type='MMOE',
            feature_columns=feature_columns,
            task_names=self.config.tasks,
            # MMOE特定参数
            num_experts=self.config.num_experts,
            # 通用架构参数
            expert_dnn_hidden_units=tuple(self.config.expert_dims),
            gate_dnn_hidden_units=tuple(self.config.gate_dims),
            tower_dnn_hidden_units=tuple(self.config.tower_dims),
            # 正则化参数
            dnn_dropout=self.config.dropout,
            l2_reg_embedding=self.config.l2_reg_embedding,
            l2_reg_dnn=self.config.l2_reg_dnn
        )
        
        logger.info(f"✅ Created MMOE model with:")
        logger.info(f"   Number of experts: {self.config.num_experts}")
        logger.info(f"   Expert dims: {self.config.expert_dims}")
        logger.info(f"   Gate dims: {self.config.gate_dims}")
        logger.info(f"   Tower dims: {self.config.tower_dims}")
        logger.info(f"   Tasks: {', '.join(self.config.tasks)}")
        
        return model
    
    def get_model_config(self) -> dict:
        """获取MMOE模型配置"""
        config = super().get_model_config()
        config.update({
            'mmoe_config': {
                'num_experts': self.config.num_experts,
                'expert_dims': self.config.expert_dims,
                'gate_dims': self.config.gate_dims,
                'tower_dims': self.config.tower_dims
            }
        })
        return config


def parse_dimensions(dims_str: str) -> List[int]:
    """解析维度字符串"""
    return [int(d.strip()) for d in dims_str.split(',')]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MMOE多任务学习CTR模型训练')
    
    # ============ 基础参数 ============
    parser.add_argument('--input-path', '-i', required=True,
                        help='输入数据路径（文件或目录）')
    parser.add_argument('--output-path', '-o', default="models/mmoe_training",
                        help='输出目录')
    parser.add_argument('--sample-size', type=int, 
                        help='随机采样的文件数量（仅目录输入）')
    
    # ============ 任务配置 ============
    parser.add_argument('--tasks', 
                        help='要训练的任务，逗号分隔（如：ctr,like_rate,comment_rate）')
    parser.add_argument('--task-weights', 
                        help='任务权重，JSON格式（如：{"ctr":1.0,"like_rate":2.0}）')
    
    # ============ MMOE模型参数 ============
    parser.add_argument('--num-experts', type=int, default=3,
                        help='专家网络数量')
    parser.add_argument('--expert-dims', default='128,64',
                        help='专家网络维度，逗号分隔')
    parser.add_argument('--gate-dims', default='32',
                        help='门控网络维度，逗号分隔')
    parser.add_argument('--tower-dims', default='64,32',
                        help='任务塔维度，逗号分隔')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--l2-reg-embedding', type=float, default=1e-5,
                        help='Embedding层L2正则化')
    parser.add_argument('--l2-reg-dnn', type=float, default=0,
                        help='DNN层L2正则化')
    
    # ============ 数据预处理参数 ============
    parser.add_argument('--filter-zeros', action='store_true', default=True,
                        help='过滤全零CLIP特征的样本')
    parser.add_argument('--min-impression', type=int, default=5000,
                        help='最小曝光数阈值')
    parser.add_argument('--use-pca', action='store_true',
                        help='对CLIP特征使用PCA降维')
    parser.add_argument('--pca-components', type=int, default=256,
                        help='PCA维度')
    parser.add_argument('--label-normalization', default='standard',
                        choices=['none', 'standard', 'minmax', 'robust'],
                        help='标签归一化方法')
    
    # ============ 训练参数 ============
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='批大小')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='验证集比例')
    
    # ============ 早停参数 ============
    parser.add_argument('--use-early-stopping', action='store_true', default=True,
                        help='启用早停机制')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0001,
                        help='早停最小改善值')
    
    # ============ 其他参数 ============
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='训练设备')
    parser.add_argument('--verbose', type=int, default=1,
                        help='日志详细程度')
    
    args = parser.parse_args()
    
    # 解析复杂参数
    if args.expert_dims:
        args.expert_dims = parse_dimensions(args.expert_dims)
    if args.gate_dims:
        args.gate_dims = parse_dimensions(args.gate_dims)
    if args.tower_dims:
        args.tower_dims = parse_dimensions(args.tower_dims)
    
    # 解析任务权重
    if args.task_weights:
        import json
        args.task_weights = json.loads(args.task_weights)
    
    # 设置模型类型
    args.model_type = 'MMOE'
    
    # 创建配置
    config = create_config_from_args(args, config_type="multi")
    
    # 创建训练器
    trainer = MMOETrainer(config)
    
    # 运行训练
    results = trainer.run()
    
    # 打印最终结果摘要（由基类的evaluator处理）
    logger.info("🎉 MMOE training completed successfully!")
    
    return results


if __name__ == "__main__":
    main()