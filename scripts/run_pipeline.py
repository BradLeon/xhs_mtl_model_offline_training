#!/usr/bin/env python3
"""
小红书CTR预估模型 - 统一特征提取管道入口
支持文本特征提取和多模态特征提取的统一接口
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.text_pipeline import run_text_pipeline, run_text_pipeline_from_args
from pipelines.multimodal_pipeline import run_multimodal_pipeline
from pipelines.incremental_multimodal_pipeline import run_incremental_multimodal_pipeline


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    if log_file is None:
        log_file = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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


def run_text_stage(args) -> Dict[str, Any]:
    """运行文本特征提取阶段"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting text feature extraction stage")
    
    # 使用新的模块化text pipeline接口
    return run_text_pipeline_from_args(args)


def run_multimodal_stage(args) -> Dict[str, Any]:
    """运行多模态特征提取阶段"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting multimodal feature extraction stage")
    
    # 确定输入路径
    if args.input:
        input_path = args.input
    else:
        # 自动查找text pipeline的输出
        text_output_base = "/Volumes/home/raw_data/text_features_parquet"
        if os.path.exists(text_output_base):
            # 找到最新的批次输出
            batch_dirs = []
            for item in os.listdir(text_output_base):
                if item.startswith(f"batch_{args.batch_start:05d}"):
                    batch_dirs.append(os.path.join(text_output_base, item))
            
            if batch_dirs:
                # 选择最新的目录
                input_path = max(batch_dirs, key=os.path.getmtime)
                logger.info(f"Auto-detected text features input: {input_path}")
            else:
                raise ValueError(f"No text features found for batch {args.batch_start}-{args.batch_end}")
        else:
            raise ValueError("Text features directory not found. Run text stage first.")
    
    # 构建参数字典，适配新的MultimodalConfig结构
    kwargs = {}
    
    # 基本参数
    if hasattr(args, 'batch_size'):
        kwargs['batch_size'] = args.batch_size
    if hasattr(args, 'resume'):
        kwargs['resume'] = args.resume
    if hasattr(args, 'model_name'):
        kwargs['model_name'] = args.model_name
    if hasattr(args, 'gpu_batch_size'):
        kwargs['gpu_batch_size'] = args.gpu_batch_size
    if hasattr(args, 'num_downloaders'):
        kwargs['num_downloaders'] = args.num_downloaders
    if hasattr(args, 'checkpoint_interval'):
        kwargs['checkpoint_interval'] = args.checkpoint_interval
    if hasattr(args, 'min_impression_threshold'):
        kwargs['min_impression_threshold'] = args.min_impression_threshold
    if hasattr(args, 'max_workers'):
        kwargs['max_workers'] = args.max_workers
    
    # 特征启用配置
    if hasattr(args, 'enable_all_features') and args.enable_all_features:
        kwargs['enable_cover_image'] = True
        kwargs['enable_inner_images'] = True
        kwargs['enable_title_text'] = True
        kwargs['enable_content_text'] = True
        kwargs['enable_tag_text'] = True
    else:
        # 单独的特征配置 - 只有在用户明确提供参数时才覆盖默认值
        # 这样可以保持 MultimodalConfig 中的默认 True 值
        pass  # 让 MultimodalConfig 使用其默认值
    
    # 特征处理参数
    if hasattr(args, 'pooling_strategy'):
        kwargs['pooling_strategy'] = args.pooling_strategy
    if hasattr(args, 'max_inner_images'):
        kwargs['max_inner_images'] = args.max_inner_images
    if hasattr(args, 'max_content_length'):
        kwargs['max_content_length'] = args.max_content_length
    
    return run_multimodal_pipeline(
        input_path=input_path,
        output_path=args.output,
        **kwargs
    )


def run_incremental_multimodal_stage(args) -> Dict[str, Any]:
    """运行增量多模态特征提取阶段

    从已有的 image_features_parquet 提取缺失特征，避免重复 CLIP 计算
    与 run_multimodal_stage 保持一致的参数处理方式
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting incremental multimodal feature extraction stage")

    # 确定输入路径（必须是已有 image_feat_* 和 text_feat_* 的 parquet）
    if not args.input:
        raise ValueError("--input is required for incremental-multimodal stage")

    input_path = args.input

    # 构建参数字典（与 run_multimodal_stage 保持一致）
    kwargs = {}

    # 基本参数
    if hasattr(args, 'batch_size') and args.batch_size:
        kwargs['batch_size'] = args.batch_size
    if hasattr(args, 'resume') and args.resume:
        kwargs['resume'] = args.resume
    if hasattr(args, 'model_name') and args.model_name:
        kwargs['model_name'] = args.model_name
    if hasattr(args, 'gpu_batch_size') and args.gpu_batch_size:
        kwargs['gpu_batch_size'] = args.gpu_batch_size
    if hasattr(args, 'num_downloaders') and args.num_downloaders:
        kwargs['num_downloaders'] = args.num_downloaders
    if hasattr(args, 'checkpoint_interval') and args.checkpoint_interval:
        kwargs['checkpoint_interval'] = args.checkpoint_interval
    if hasattr(args, 'max_workers') and args.max_workers:
        kwargs['max_workers'] = args.max_workers

    # 特征启用配置（与 run_multimodal_stage 保持一致的逻辑）
    # --enable-all-features 会启用所有特征
    if hasattr(args, 'enable_all_features') and args.enable_all_features:
        kwargs['enable_inner_images'] = True
        kwargs['enable_content_text'] = True
        kwargs['enable_tag_text'] = True
        kwargs['enable_cover_ocr_clip'] = True
        kwargs['enable_inner_ocr_clip'] = True
    # 否则让 IncrementalMultimodalConfig 使用其默认值 (全部为 True)

    # 特征处理参数
    if hasattr(args, 'pooling_strategy') and args.pooling_strategy:
        kwargs['pooling_strategy'] = args.pooling_strategy
    if hasattr(args, 'max_inner_images') and args.max_inner_images is not None:
        kwargs['max_inner_images'] = args.max_inner_images
    if hasattr(args, 'max_content_length') and args.max_content_length:
        kwargs['max_content_length'] = args.max_content_length

    return run_incremental_multimodal_pipeline(
        input_path=input_path,
        output_path=args.output,
        **kwargs
    )


def run_full_pipeline(args) -> Dict[str, Any]:
    """运行完整管道"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting full pipeline: text → multimodal")
    
    results = {}
    
    # 第一阶段：文本特征提取
    if not args.skip_text:
        logger.info("="*60)
        logger.info("STAGE 1: TEXT FEATURE EXTRACTION")
        logger.info("="*60)
        
        text_results = run_text_stage(args)
        results['text_stage'] = text_results
        
        logger.info("✅ Text feature extraction completed")
    else:
        logger.info("⏭️ Skipping text feature extraction stage")
    
    # 第二阶段：多模态特征提取
    if not args.skip_multimodal:
        logger.info("="*60)
        logger.info("STAGE 2: MULTIMODAL FEATURE EXTRACTION")
        logger.info("="*60)
        
        multimodal_results = run_multimodal_stage(args)
        results['multimodal_stage'] = multimodal_results
        
        logger.info("✅ Multimodal feature extraction completed")
    else:
        logger.info("⏭️ Skipping multimodal feature extraction stage")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='小红书CTR预估模型 - 统一特征提取管道')

    # 基本参数
    parser.add_argument('--stage', choices=['text', 'multimodal', 'incremental-multimodal', 'all'], default='all',
                       help='运行阶段: text=文本特征, multimodal=多模态特征, incremental-multimodal=增量多模态特征, all=完整管道')
    parser.add_argument('--input', help='输入数据路径 (multimodal阶段使用)')
    parser.add_argument('--output', help='输出路径')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--log-file', help='日志文件名')
    
    # 文本管道参数
    text_group = parser.add_argument_group('文本特征提取参数')
    text_group.add_argument('--batch-start', type=int, default=0,
                           help='起始批次号')
    text_group.add_argument('--batch-end', type=int, default=100,
                           help='结束批次号')
    text_group.add_argument('--min-impression', type=int, default=500,
                           help='最小曝光阈值')
    
    # 多模态管道参数
    multimodal_group = parser.add_argument_group('多模态特征提取参数')
    multimodal_group.add_argument('--batch-size', type=int, default=3000,
                                 help='批次大小')
    multimodal_group.add_argument('--enable-all-features', action='store_true',
                                 help='启用所有特征类型')
    multimodal_group.add_argument('--resume', action='store_true',
                                 help='从checkpoint恢复处理')
    multimodal_group.add_argument('--model-name', default='ViT-B-16',
                                 help='Chinese-CLIP模型名称')
    multimodal_group.add_argument('--gpu-batch-size', type=int, default=8,
                                 help='GPU批次大小')
    multimodal_group.add_argument('--num-downloaders', type=int, default=8,
                                 help='图片下载并发数')
    multimodal_group.add_argument('--checkpoint-interval', type=int, default=5,
                                 help='checkpoint保存间隔')
    multimodal_group.add_argument('--min-impression-threshold', type=int, default=5000,
                                 help='最小曝光阈值')
    multimodal_group.add_argument('--max-workers', type=int, default=2,
                                 help='多进程worker数量')
    
    # 特征配置
    feature_group = parser.add_argument_group('特征配置')
    feature_group.add_argument('--enable-cover-image', action='store_true',
                              help='启用封面图特征')
    feature_group.add_argument('--enable-inner-images', action='store_true', 
                              help='启用内页图特征')
    feature_group.add_argument('--enable-title-text', action='store_true',
                              help='启用标题文本特征')
    feature_group.add_argument('--enable-content-text', action='store_true',
                              help='启用内容文本特征')
    feature_group.add_argument('--enable-tag-text', action='store_true',
                              help='启用标签文本特征')
    feature_group.add_argument('--pooling-strategy', choices=['mean', 'max'], default='mean',
                              help='内页图池化策略')
    feature_group.add_argument('--max-inner-images', type=int, default=5,
                              help='最大内页图片数量')
    feature_group.add_argument('--max-content-length', type=int, default=200,
                              help='内容文本最大长度')
    
    # 跳过选项
    skip_group = parser.add_argument_group('跳过选项')
    skip_group.add_argument('--skip-text', action='store_true',
                           help='跳过文本特征提取阶段')
    skip_group.add_argument('--skip-multimodal', action='store_true',
                           help='跳过多模态特征提取阶段')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.stage == 'text' and args.skip_text:
        parser.error("Cannot skip text stage when stage is 'text'")
    if args.stage == 'multimodal' and args.skip_multimodal:
        parser.error("Cannot skip multimodal stage when stage is 'multimodal'")
    if args.batch_start < 0:
        parser.error("Batch start must be non-negative")
    if args.batch_end < args.batch_start:
        parser.error("Batch end must be >= batch start")
    
    # 设置日志
    log_file = setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("🎯 小红书CTR预估模型 - 特征提取管道")
    logger.info("="*80)
    logger.info(f"阶段: {args.stage}")
    logger.info(f"批次范围: {args.batch_start} - {args.batch_end}")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info(f"日志文件: {log_file}")
    logger.info("="*80)
    
    try:
        # 根据阶段运行相应的管道
        if args.stage == 'text':
            results = run_text_stage(args)
        elif args.stage == 'multimodal':
            results = run_multimodal_stage(args)
        elif args.stage == 'incremental-multimodal':
            results = run_incremental_multimodal_stage(args)
        elif args.stage == 'all':
            results = run_full_pipeline(args)
        else:
            raise ValueError(f"Unknown stage: {args.stage}")
        
        # 输出结果摘要
        logger.info("="*80)
        logger.info("🎉 管道执行完成!")
        logger.info("="*80)
        
        if isinstance(results, dict):
            for stage_name, stage_results in results.items():
                if isinstance(stage_results, dict):
                    logger.info(f"{stage_name.upper()}:")
                    for key, value in stage_results.items():
                        if key in ['rows_processed', 'total_records', 'total_batches']:
                            logger.info(f"  {key}: {value:,}")
                        elif key in ['duration_hours', 'training_time']:
                            logger.info(f"  {key}: {value:.2f}")
                        elif key in ['output_path']:
                            logger.info(f"  {key}: {value}")
        
        logger.info(f"完成时间: {datetime.now()}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ 管道执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()