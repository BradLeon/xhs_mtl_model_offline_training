#!/usr/bin/env python3
"""
标签体系知识库提取脚本

从小红书CTR预估数据中提取平台标签体系，生成可参考的知识库文件。
支持大规模数据处理，包含数据采样、质量控制和统计分析功能。

标签字段：
- intention_lv1: 意图分类一级
- intention_lv2is_mcn: 意图分类二级+MCN信息
- taxonomy1: 分类标签一级
- taxonomy2: 分类标签二级
- taxonomy3: 分类标签三级

使用示例：
    # 基础使用
    python tools/extract_taxonomy_knowledge.py \
        --input /Volumes/home/raw_data/image_features_ple_parquet/ \
        --output taxonomy_knowledge.csv

    # 大数据采样处理
    python tools/extract_taxonomy_knowledge.py \
        --input /Volumes/home/raw_data/image_features_ple_parquet/ \
        --output taxonomy_knowledge.csv \
        --sample-size 1000 \
        --exclude-defaults \
        --verbose

    # 生成详细报告
    python tools/extract_taxonomy_knowledge.py \
        --input /Volumes/home/raw_data/image_features_ple_parquet/ \
        --output taxonomy_knowledge.csv \
        --sample-size 500 \
        --generate-report \
        --verbose
"""

import argparse
import logging
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TaxonomyKnowledgeExtractor:
    """标签体系知识库提取器
    
    从parquet文件中提取标签组合，生成结构化的知识库。
    支持大规模数据处理、采样、质量控制等功能。
    """
    
    # 目标标签字段
    TARGET_COLUMNS = [
        'intention_lv1', 'intention_lv2is_mcn', 
        'taxonomy1', 'taxonomy2', 'taxonomy3'
    ]
    
    # 需要过滤的默认值
    DEFAULT_VALUES = {
        '其他', 'None', 'null', 'nan', '', 'unknown', '未知', '暂无'
    }
    
    def __init__(self, 
                 input_path: str,
                 output_path: str = 'taxonomy_knowledge.csv',
                 sample_size: Optional[int] = None,
                 exclude_defaults: bool = True,
                 generate_report: bool = False,
                 verbose: bool = False):
        """初始化提取器
        
        Args:
            input_path: 输入目录路径
            output_path: 输出CSV文件路径
            sample_size: 随机采样文件数量
            exclude_defaults: 是否排除默认值
            generate_report: 是否生成详细报告
            verbose: 是否详细输出
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.sample_size = sample_size
        self.exclude_defaults = exclude_defaults
        self.generate_report = generate_report
        self.verbose = verbose
        
        # 设置日志级别
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_rows': 0,
            'unique_combinations': 0,
            'filtered_combinations': 0,
            'processing_time': 0.0
        }
        
        # 字段值统计
        self.field_stats = {col: Counter() for col in self.TARGET_COLUMNS}
        
        logger.info(f"🎯 Initialized TaxonomyKnowledgeExtractor")
        logger.info(f"   Input: {self.input_path}")
        logger.info(f"   Output: {self.output_path}")
        logger.info(f"   Sample size: {self.sample_size or 'All files'}")
        logger.info(f"   Exclude defaults: {self.exclude_defaults}")
    
    def extract_knowledge(self) -> Dict[str, List[str]]:
        """提取标签体系知识库
        
        Returns:
            包含各列unique值的字典
        """
        start_time = time.time()
        logger.info("🚀 开始提取标签体系知识库...")
        
        # 1. 发现并采样文件
        parquet_files = self._discover_files()
        
        # 2. 分批处理文件
        all_combinations = self._process_files_in_batches(parquet_files)
        
        # 3. 提取各列unique值
        unique_values = self._extract_unique_values(all_combinations)
        
        # 4. 保存结果
        self._save_results(unique_values)
        
        # 5. 生成报告
        if self.generate_report:
            self._generate_detailed_report(unique_values)
        
        # 记录总时间
        self.stats['processing_time'] = time.time() - start_time
        self._print_summary()
        
        return unique_values
    
    def _discover_files(self) -> List[Path]:
        """发现并采样parquet文件"""
        logger.info("📁 扫描parquet文件...")
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"输入路径不存在: {self.input_path}")
        
        if not self.input_path.is_dir():
            raise ValueError(f"输入路径必须是目录: {self.input_path}")
        
        # 递归查找所有parquet文件
        parquet_files = list(self.input_path.glob("**/*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"在 {self.input_path} 中未找到parquet文件")
        
        self.stats['total_files'] = len(parquet_files)
        logger.info(f"   发现 {len(parquet_files)} 个parquet文件")
        
        # 随机采样
        if self.sample_size and len(parquet_files) > self.sample_size:
            logger.info(f"🎲 随机采样 {self.sample_size} 个文件 (共{len(parquet_files)}个)")
            parquet_files = random.sample(parquet_files, self.sample_size)
        
        self.stats['processed_files'] = len(parquet_files)
        logger.info(f"✅ 将处理 {len(parquet_files)} 个文件")
        
        return parquet_files
    
    def _process_files_in_batches(self, parquet_files: List[Path]) -> List[pd.DataFrame]:
        """分批处理文件以避免内存溢出"""
        logger.info("📊 开始分批处理文件...")
        
        batch_size = 50  # 每批处理50个文件
        all_combinations = []
        available_columns = None  # 用于存储实际可用的列
        
        for i in range(0, len(parquet_files), batch_size):
            batch_files = parquet_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(parquet_files) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 处理批次 {batch_num}/{total_batches} ({len(batch_files)} 文件)")
            
            batch_dfs = []
            
            for j, file_path in enumerate(batch_files):
                try:
                    if self.verbose:
                        logger.debug(f"   加载文件: {file_path.name}")
                    
                    # 先读取一个文件检查可用列（只在第一次）
                    if available_columns is None:
                        temp_df = pd.read_parquet(file_path)
                        available_columns = [col for col in self.TARGET_COLUMNS if col in temp_df.columns]
                        
                        if not available_columns:
                            logger.error(f"❌ 在文件中未找到任何目标列: {self.TARGET_COLUMNS}")
                            logger.info(f"   文件 {file_path.name} 包含的列: {list(temp_df.columns)}")
                            raise ValueError("未找到目标列")
                        
                        logger.info(f"✅ 找到可用列: {available_columns}")
                        
                        # 使用第一个文件的数据
                        df = temp_df[available_columns].copy()
                    else:
                        # 后续文件只读取可用列
                        df = pd.read_parquet(file_path, columns=available_columns)
                    
                    if len(df) > 0:
                        # 统计字段值
                        for col in available_columns:
                            if col in df.columns:
                                self.field_stats[col].update(df[col].dropna().astype(str))
                        
                        # 确保所有目标列都存在（缺失的用NaN填充）
                        for col in self.TARGET_COLUMNS:
                            if col not in df.columns:
                                df[col] = pd.NA
                        
                        # 只保留目标列并按顺序排列
                        df_subset = df[self.TARGET_COLUMNS].copy()
                        batch_dfs.append(df_subset)
                        self.stats['total_rows'] += len(df)
                    
                except Exception as e:
                    logger.warning(f"   ⚠️ 跳过文件 {file_path.name}: {e}")
                    continue
            
            # 合并当前批次的数据
            if batch_dfs:
                batch_combined = pd.concat(batch_dfs, ignore_index=True)
                all_combinations.append(batch_combined)
                
                if self.verbose:
                    logger.debug(f"   批次 {batch_num} 完成: {len(batch_combined):,} 行")
        
        logger.info(f"✅ 文件处理完成，共收集 {self.stats['total_rows']:,} 行数据")
        return all_combinations
    
    def _extract_unique_values(self, all_combinations: List[pd.DataFrame]) -> Dict[str, List[str]]:
        """提取每列的unique值"""
        logger.info("🧹 开始提取各列unique值...")
        
        # 合并所有数据
        logger.info("   合并所有批次数据...")
        combined_df = pd.concat(all_combinations, ignore_index=True)
        logger.info(f"   合并后总行数: {len(combined_df):,}")
        
        # 为每列提取unique值
        unique_values = {}
        
        for col in self.TARGET_COLUMNS:
            if col in combined_df.columns:
                logger.info(f"   处理列: {col}")
                
                # 获取非空值
                col_values = combined_df[col].dropna().astype(str).str.strip()
                
                # 质量过滤
                if self.exclude_defaults:
                    col_values = col_values[~col_values.isin(self.DEFAULT_VALUES)]
                
                # 去重并排序
                unique_vals = sorted(col_values.unique())
                unique_values[col] = unique_vals
                
                logger.info(f"     找到 {len(unique_vals)} 个unique值")
                
                # 更新统计
                self.field_stats[col] = Counter(col_values)
            else:
                logger.warning(f"   ⚠️ 列 {col} 在数据中不存在")
                unique_values[col] = []
        
        # 更新统计信息
        total_unique = sum(len(vals) for vals in unique_values.values())
        self.stats['unique_combinations'] = total_unique
        self.stats['filtered_combinations'] = total_unique
        
        logger.info(f"✅ 数据处理完成，总计 {total_unique} 个unique值")
        return unique_values
    
    def _save_results(self, unique_values: Dict[str, List[str]]) -> None:
        """保存结果到CSV文件"""
        logger.info("💾 保存结果...")
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 找到最长的列
        max_length = max(len(vals) for vals in unique_values.values()) if unique_values else 0
        
        # 创建DataFrame，用NaN填充短列
        result_data = {}
        for col in self.TARGET_COLUMNS:
            values = unique_values.get(col, [])
            # 用NaN填充到最大长度
            padded_values = values + [pd.NA] * (max_length - len(values))
            result_data[col] = padded_values
        
        result_df = pd.DataFrame(result_data)
        
        # 保存主要结果
        result_df.to_csv(self.output_path, index=False, encoding='utf-8')
        logger.info(f"✅ 知识库已保存: {self.output_path}")
        logger.info(f"   文件大小: {self.output_path.stat().st_size / 1024:.2f} KB")
        
        # 输出每列的统计信息
        logger.info(f"📊 各列unique值数量:")
        for col in self.TARGET_COLUMNS:
            count = len(unique_values.get(col, []))
            logger.info(f"   - {col}: {count} 个值")
    
    def _generate_detailed_report(self, unique_values: Dict[str, List[str]]) -> None:
        """生成详细的分析报告"""
        logger.info("📊 生成详细分析报告...")
        
        report_path = self.output_path.with_suffix('.report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("小红书标签体系知识库分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基础统计
            f.write("📊 基础统计信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"处理文件数: {self.stats['processed_files']:,}\n")
            f.write(f"总数据行数: {self.stats['total_rows']:,}\n")
            f.write(f"总unique值数: {self.stats['unique_combinations']:,}\n")
            f.write(f"处理耗时: {self.stats['processing_time']:.2f}秒\n\n")
            
            # 各字段unique值数量
            f.write("📈 各字段unique值统计\n")
            f.write("-" * 40 + "\n")
            for col in self.TARGET_COLUMNS:
                count = len(unique_values.get(col, []))
                f.write(f"{col}: {count} 个unique值\n")
            
            # 各字段值分布
            f.write("\n🏷️ 各字段值分布 (Top 20)\n")
            f.write("-" * 40 + "\n")
            
            for col in self.TARGET_COLUMNS:
                f.write(f"\n{col}:\n")
                if col in self.field_stats:
                    for value, count in self.field_stats[col].most_common(20):
                        f.write(f"  {value}: {count:,}\n")
            
            # 各字段所有unique值
            f.write(f"\n🎯 各字段所有unique值\n")
            f.write("-" * 40 + "\n")
            for col in self.TARGET_COLUMNS:
                values = unique_values.get(col, [])
                f.write(f"\n{col} ({len(values)} 个值):\n")
                for i, value in enumerate(values, 1):
                    f.write(f"  {i:3d}. {value}\n")
            
        logger.info(f"📊 详细报告已保存: {report_path}")
    
    def _print_summary(self) -> None:
        """打印执行摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("🎉 标签体系知识库提取完成!")
        logger.info("=" * 80)
        logger.info(f"📁 输入目录: {self.input_path}")
        logger.info(f"💾 输出文件: {self.output_path}")
        logger.info(f"📊 统计信息:")
        logger.info(f"   - 发现文件: {self.stats['total_files']:,}")
        logger.info(f"   - 处理文件: {self.stats['processed_files']:,}")
        logger.info(f"   - 总数据行: {self.stats['total_rows']:,}")
        logger.info(f"   - 总unique值: {self.stats['unique_combinations']:,}")
        logger.info(f"   - 最终unique值: {self.stats.get('filtered_combinations', self.stats['unique_combinations']):,}")
        logger.info(f"⏱️  处理耗时: {self.stats['processing_time']:.2f}秒")
        logger.info("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='提取小红书平台标签体系知识库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础提取
  python tools/extract_taxonomy_knowledge.py \\
      --input /Volumes/home/raw_data/image_features_ple_parquet/ \\
      --output taxonomy_knowledge.csv

  # 大数据采样处理
  python tools/extract_taxonomy_knowledge.py \\
      --input /Volumes/home/raw_data/image_features_ple_parquet/ \\
      --output taxonomy_knowledge.csv \\
      --sample-size 1000 \\
      --exclude-defaults \\
      --verbose

  # 生成详细报告
  python tools/extract_taxonomy_knowledge.py \\
      --input /Volumes/home/raw_data/image_features_ple_parquet/ \\
      --output taxonomy_knowledge.csv \\
      --sample-size 500 \\
      --generate-report \\
      --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入parquet文件目录路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='taxonomy_knowledge.csv',
        help='输出CSV文件路径 (默认: taxonomy_knowledge.csv)'
    )
    
    parser.add_argument(
        '--sample-size', '-s',
        type=int,
        help='随机采样文件数量 (用于大数据集)'
    )
    
    parser.add_argument(
        '--exclude-defaults',
        action='store_true',
        default=True,
        help='排除默认值如"其他"等 (默认: True)'
    )
    
    parser.add_argument(
        '--no-exclude-defaults',
        action='store_false',
        dest='exclude_defaults',
        help='包含所有值，不过滤默认值'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='生成详细的分析报告'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细输出处理过程'
    )
    
    args = parser.parse_args()
    
    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"❌ 输入路径不存在: {input_path}")
        sys.exit(1)
    
    try:
        # 创建提取器并执行
        extractor = TaxonomyKnowledgeExtractor(
            input_path=args.input,
            output_path=args.output,
            sample_size=args.sample_size,
            exclude_defaults=args.exclude_defaults,
            generate_report=args.generate_report,
            verbose=args.verbose
        )
        
        # 执行提取
        unique_values = extractor.extract_knowledge()
        
        # 输出样本
        if unique_values:
            logger.info(f"\n📋 各列unique值样本 (前5个):")
            for col in extractor.TARGET_COLUMNS:
                values = unique_values.get(col, [])
                if values:
                    sample_values = values[:5]
                    logger.info(f"  {col}: {', '.join(sample_values)}...")
                    if len(values) > 5:
                        logger.info(f"    (共 {len(values)} 个值)")
                else:
                    logger.info(f"  {col}: (无数据)")
        
        logger.info(f"\n✅ 成功生成标签体系知识库: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ 提取过程中发生错误: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()