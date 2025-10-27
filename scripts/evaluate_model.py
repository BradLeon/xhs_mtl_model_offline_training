#!/usr/bin/env python3
"""
小红书CTR预估模型 - 统一模型评估入口
支持CLIP特征评估和模型性能评估
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.clip_evaluator import evaluate_clip_features


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    if log_file is None:
        log_file = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 确保logs目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_file
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return str(log_file)


def auto_detect_input_path() -> str:
    """自动检测输入路径"""
    logger = logging.getLogger(__name__)
    
    # 查找最新的特征输出
    base_paths = [
        "/Volumes/home/raw_data/image_features_ple_parquet",
        "/Volumes/home/raw_data/image_features_chinese_clip_parquet",
        "/Volumes/home/raw_data/merged_features_parquet",
        "/Volumes/home/raw_data/text_features_parquet"
    ]
    
    for base_path in base_paths:
        if os.path.exists(base_path):
            # 找到最新的输出目录
            subdirs = []
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    subdirs.append(item_path)
            
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                logger.info(f"Auto-detected input path: {latest_dir}")
                return latest_dir
    
    # 如果自动检测失败，返回默认路径
    default_path = "/Volumes/home/raw_data/image_features_ple_parquet"
    logger.warning(f"Could not auto-detect input path, using default: {default_path}")
    return default_path


def main():
    parser = argparse.ArgumentParser(description='小红书CTR预估模型 - 统一评估入口')
    
    # 基本参数
    parser.add_argument('--type', choices=['clip', 'model'], default='clip',
                       help='评估类型: clip=CLIP特征评估, model=模型性能评估')
    parser.add_argument('--input', help='输入数据路径 (如不指定则自动检测)')
    parser.add_argument('--output', default='evaluation_results',
                       help='评估结果输出目录')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--log-file', help='日志文件名')
    
    # CLIP评估参数
    clip_group = parser.add_argument_group('CLIP特征评估参数')
    clip_group.add_argument('--sample-size', type=int, default=10000,
                           help='采样大小')
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # 自动检测输入路径
    if not args.input:
        args.input = auto_detect_input_path()
    
    logger.info("="*80)
    logger.info("🎯 小红书CTR预估模型 - 模型评估")
    logger.info("="*80)
    logger.info(f"评估类型: {args.type}")
    logger.info(f"输入路径: {args.input}")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*80)
    
    try:
        # 验证输入路径
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input path not found: {args.input}")
        
        # 根据评估类型运行相应的评估
        if args.type == 'clip':
            results = evaluate_clip_features(
                input_path=args.input,
                output_dir=args.output,
                sample_size=args.sample_size
            )
            
            logger.info("="*80)
            logger.info("🎉 CLIP特征评估完成!")
            logger.info(f"📊 评估报告: {results['report_path']}")
            logger.info(f"📁 结果目录: {args.output}")
            
        elif args.type == 'model':
            logger.info("模型性能评估功能开发中...")
            # TODO: 实现模型性能评估
            
        else:
            raise ValueError(f"Unknown evaluation type: {args.type}")
        
        logger.info(f"完成时间: {datetime.now()}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()