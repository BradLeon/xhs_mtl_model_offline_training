#!/usr/bin/env python3
"""
统一多任务学习训练入口脚本

支持所有MTL模型（PLE、MMOE、PNN-MMOE）的统一训练接口。
基于重构后的基础架构，提供一致的命令行参数和配置管理。

支持的模型：
- PLE (Progressive Layered Extraction)
- MMOE (Multi-gate Mixture-of-Experts)  
- PNN_MMOE (PNN-MMOE Hybrid Architecture)

使用示例：
----------
# 1. PLE模型训练
python scripts/train_mtl_model.py \
    --model-type PLE \
    --input-path /data/features \
    --shared-expert-num 3 \
    --specific-expert-num 2 \
    --epochs 30

# 2. MMOE模型训练
python scripts/train_mtl_model.py \
    --model-type MMOE \
    --input-path /data/features \
    --num-experts 6 \
    --epochs 30

# 3. PNN-MMOE混合架构训练
python scripts/train_mtl_model.py \
    --model-type PNN_MMOE \
    --input-path /data/features \
    --use-inner-product \
    --num-experts 4 \
    --epochs 30

# 4. 自定义任务和权重
python scripts/train_mtl_model.py \
    --model-type PLE \
    --input-path /data/features \
    --tasks ctr,like_rate,comment_rate \
    --task-weights '{"ctr":1.0,"like_rate":2.0,"comment_rate":3.0}' \
    --epochs 25
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入重构后的训练器
from training.multi_task.ple_trainer import PLETrainer
from training.multi_task.mmoe_trainer import MMOETrainer
from training.multi_task.pnn_mmoe_trainer import PNNMMOETrainer

# 导入配置管理
from training.base.base_config import create_config_from_args, MultiTaskConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedMTLTrainer:
    """统一MTL训练器
    
    根据模型类型动态选择对应的训练器，提供统一的训练接口。
    """
    
    def __init__(self):
        # 注册所有可用的训练器
        self.trainer_registry = {
            'PLE': PLETrainer,
            'MMOE': MMOETrainer,
            'PNN_MMOE': PNNMMOETrainer
        }
        
        # 支持的模型类型
        self.supported_models = list(self.trainer_registry.keys())
        
        logger.info(f"Initialized Unified MTL Trainer")
        logger.info(f"Supported models: {self.supported_models}")
    
    def create_trainer(self, config: MultiTaskConfig):
        """根据配置创建对应的训练器
        
        Args:
            config: 多任务训练配置
            
        Returns:
            对应的训练器实例
        """
        model_type = config.model_type
        
        if model_type not in self.trainer_registry:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported models: {self.supported_models}")
        
        trainer_class = self.trainer_registry[model_type]
        trainer = trainer_class(config)
        
        logger.info(f"✅ Created {model_type} trainer")
        return trainer
    
    def train(self, config: MultiTaskConfig) -> Dict[str, Any]:
        """统一训练接口
        
        Args:
            config: 训练配置
            
        Returns:
            训练结果
        """
        logger.info("="*80)
        logger.info("UNIFIED MTL TRAINING STARTED")
        logger.info("="*80)
        logger.info(f"Model Type: {config.model_type}")
        logger.info(f"Tasks: {config.tasks}")
        logger.info(f"Input Path: {config.input_path}")
        logger.info(f"Output Path: {config.output_path}")
        logger.info("="*80)
        
        # 创建训练器
        trainer = self.create_trainer(config)
        
        # 执行训练
        results = trainer.run()
        
        logger.info("="*80)
        logger.info("UNIFIED MTL TRAINING COMPLETED")
        logger.info("="*80)
        
        return results
    
    def get_model_info(self) -> Dict[str, str]:
        """获取所有支持模型的信息"""
        return {
            'PLE': 'Progressive Layered Extraction - 渐进式分层提取架构',
            'MMOE': 'Multi-gate Mixture-of-Experts - 多门控专家混合架构',
            'PNN_MMOE': 'PNN-MMOE Hybrid - PNN特征交互 + MMOE多任务架构'
        }


def parse_dimensions(dims_str: str) -> List[int]:
    """解析维度字符串"""
    return [int(d.strip()) for d in dims_str.split(',')]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='统一多任务学习MTL模型训练入口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的模型类型：
  PLE       Progressive Layered Extraction (渐进式分层提取)
  MMOE      Multi-gate Mixture-of-Experts (多门控专家混合)
  PNN_MMOE  PNN-MMOE Hybrid Architecture (PNN-MMOE混合架构)

使用示例：
  # PLE模型训练
  python scripts/train_mtl_model.py --model-type PLE --input-path /data/features --epochs 30
  
  # MMOE模型训练  
  python scripts/train_mtl_model.py --model-type MMOE --input-path /data/features --num-experts 6 --epochs 30
  
  # PNN-MMOE混合架构训练
  python scripts/train_mtl_model.py --model-type PNN_MMOE --input-path /data/features --use-inner-product --epochs 30
        """
    )
    
    # ============ 基础参数 ============
    parser.add_argument('--model-type', '-m', required=True,
                        choices=['PLE', 'MMOE', 'PNN_MMOE'],
                        help='模型类型：PLE/MMOE/PNN_MMOE')
    parser.add_argument('--input-path', '-i', required=True,
                        help='输入数据路径（文件或目录）')
    parser.add_argument('--output-path', '-o', default="models/unified_mtl_training",
                        help='输出目录')
    parser.add_argument('--sample-size', type=int, 
                        help='随机采样的文件数量（仅目录输入）')
    
    # ============ 任务配置 ============
    parser.add_argument('--tasks', 
                        help='要训练的任务，逗号分隔（如：ctr,like_rate,comment_rate）')
    parser.add_argument('--task-weights', 
                        help='任务权重，JSON格式（如：{"ctr":1.0,"like_rate":2.0}）')
    
    # ============ PLE专用参数 ============
    ple_group = parser.add_argument_group('PLE模型参数', 'Progressive Layered Extraction模型特定参数')
    ple_group.add_argument('--shared-expert-num', type=int, default=2,
                          help='共享专家数量（仅PLE）')
    ple_group.add_argument('--specific-expert-num', type=int, default=2,
                          help='任务特定专家数量（仅PLE）')
    ple_group.add_argument('--num-levels', type=int, default=2,
                          help='PLE层数（仅PLE）')
    
    # ============ MMOE/PNN-MMOE专用参数 ============
    mmoe_group = parser.add_argument_group('MMOE模型参数', 'MMOE和PNN-MMOE模型特定参数')
    mmoe_group.add_argument('--num-experts', type=int, default=3,
                           help='专家网络数量（MMOE/PNN_MMOE）')
    
    # ============ PNN-MMOE专用参数 ============
    pnn_group = parser.add_argument_group('PNN参数', 'PNN-MMOE混合架构的PNN特定参数')
    pnn_group.add_argument('--use-inner-product', action='store_true', default=True,
                          help='使用PNN内积特征交互（仅PNN_MMOE）')
    pnn_group.add_argument('--use-outter-product', action='store_true',
                          help='使用PNN外积特征交互（仅PNN_MMOE）')
    pnn_group.add_argument('--no-inner-product', action='store_true',
                          help='禁用PNN内积特征交互（仅PNN_MMOE）')
    
    # ============ 通用网络架构参数 ============
    arch_group = parser.add_argument_group('网络架构参数', '所有模型通用的网络架构参数')
    arch_group.add_argument('--expert-dims', default='128,64',
                           help='专家网络维度，逗号分隔')
    arch_group.add_argument('--gate-dims', default='32',
                           help='门控网络维度，逗号分隔')
    arch_group.add_argument('--tower-dims', default='64,32',
                           help='任务塔维度，逗号分隔')
    arch_group.add_argument('--dropout', type=float, default=0.1,
                           help='Dropout率')
    arch_group.add_argument('--l2-reg-embedding', type=float, default=1e-5,
                           help='Embedding层L2正则化')
    arch_group.add_argument('--l2-reg-dnn', type=float, default=0,
                           help='DNN层L2正则化')
    
    # ============ 数据预处理参数 ============
    data_group = parser.add_argument_group('数据预处理参数', '数据处理和特征工程参数')
    data_group.add_argument('--filter-zeros', action='store_true', default=True,
                           help='过滤全零CLIP特征的样本')
    data_group.add_argument('--min-impression', type=int, default=5000,
                           help='最小曝光数阈值')
    data_group.add_argument('--use-pca', action='store_true',
                           help='对CLIP特征使用PCA降维')
    data_group.add_argument('--pca-components', type=int, default=256,
                           help='PCA维度')
    data_group.add_argument('--label-normalization', default='standard',
                           choices=['none', 'standard', 'minmax', 'robust'],
                           help='标签归一化方法')
    
    # ============ 训练参数 ============
    train_group = parser.add_argument_group('训练参数', '模型训练相关参数')
    train_group.add_argument('--epochs', type=int, default=20,
                            help='训练轮数')
    train_group.add_argument('--batch-size', type=int, default=256,
                            help='批大小')
    train_group.add_argument('--learning-rate', type=float, default=0.001,
                            help='学习率')
    train_group.add_argument('--validation-split', type=float, default=0.2,
                            help='验证集比例')
    
    # ============ 早停参数 ============
    early_group = parser.add_argument_group('早停参数', '早停机制相关参数')
    early_group.add_argument('--use-early-stopping', action='store_true', default=True,
                            help='启用早停机制')
    early_group.add_argument('--early-stopping-patience', type=int, default=10,
                            help='早停耐心值')
    early_group.add_argument('--early-stopping-min-delta', type=float, default=0.0001,
                            help='早停最小改善值')
    
    # ============ 其他参数 ============
    misc_group = parser.add_argument_group('其他参数', '设备和日志相关参数')
    misc_group.add_argument('--device', default='auto',
                           choices=['auto', 'cpu', 'cuda', 'mps'],
                           help='训练设备')
    misc_group.add_argument('--verbose', type=int, default=1,
                           help='日志详细程度')
    misc_group.add_argument('--show-model-info', action='store_true',
                           help='显示支持的模型信息并退出')
    
    args = parser.parse_args()
    
    # 显示模型信息
    if args.show_model_info:
        trainer = UnifiedMTLTrainer()
        model_info = trainer.get_model_info()
        print("\n支持的MTL模型:")
        print("="*50)
        for model_type, description in model_info.items():
            print(f"{model_type:10} : {description}")
        print("="*50)
        return
    
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
        try:
            args.task_weights = json.loads(args.task_weights)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid task weights JSON format: {e}")
            return
    
    # 处理PNN参数
    if args.no_inner_product:
        args.use_inner_product = False
    
    # 创建配置
    try:
        config = create_config_from_args(args, config_type="multi")
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return
    
    # 验证模型特定参数
    if config.model_type == 'PLE':
        logger.info(f"PLE配置: shared_experts={config.shared_expert_num}, "
                   f"specific_experts={config.specific_expert_num}, levels={config.num_levels}")
    elif config.model_type == 'MMOE':
        logger.info(f"MMOE配置: num_experts={config.num_experts}")
    elif config.model_type == 'PNN_MMOE':
        logger.info(f"PNN-MMOE配置: num_experts={config.num_experts}, "
                   f"inner_product={config.use_inner_product}, outer_product={config.use_outter_product}")
    
    # 创建统一训练器并执行训练
    try:
        unified_trainer = UnifiedMTLTrainer()
        results = unified_trainer.train(config)
        
        logger.info("🎉 Unified MTL training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()