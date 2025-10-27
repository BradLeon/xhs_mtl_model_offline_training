#!/usr/bin/env python3
"""
测试PLE图像特征提取pipeline输出内容的脚本
基于tools/clip_evaluation/inspect_parquet.py开发

用于检查run_local_image_pipeline_chinese-clip_for_PLE.py的产出内容

使用方法:
  python test_ple_pipeline_output.py /path/to/ple_output.parquet
  python test_ple_pipeline_output.py /Volumes/home/raw_data/image_features_ple_parquet/ --scan-all
  python test_ple_pipeline_output.py /path/to/ple_output.parquet --show-features --show-values
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def get_ple_parquet_files(path: str) -> List[Path]:
    """获取PLE输出目录中的所有parquet文件"""
    path = Path(path)
    
    if path.is_file():
        return [path]
    
    # 查找PLE输出文件
    parquet_files = []
    for pattern in ["ple_batch_*.parquet", "**/*.parquet"]:
        parquet_files.extend(path.glob(pattern))
    
    return sorted(parquet_files, key=lambda x: x.name)


def analyze_ple_features(df: pd.DataFrame) -> Dict[str, Any]:
    """分析PLE特征的详细信息"""
    analysis = {
        'cover_image': {'columns': [], 'stats': {}},
        'inner_images': {'columns': [], 'stats': {}},
        'title_text': {'columns': [], 'stats': {}},
        'content_text': {'columns': [], 'stats': {}},
        'tag_text': {'columns': [], 'stats': {}},
        'auxiliary': {'columns': [], 'stats': {}},
        'metadata': {'columns': [], 'stats': {}}
    }
    
    # 分类列
    for col in df.columns:
        if col.startswith('cover_image_feat_'):
            analysis['cover_image']['columns'].append(col)
        elif col.startswith('inner_image_feat_'):
            analysis['inner_images']['columns'].append(col)
        elif col.startswith('title_feat_'):
            analysis['title_text']['columns'].append(col)
        elif col.startswith('content_feat_'):
            analysis['content_text']['columns'].append(col)
        elif col.startswith('tag_feat_'):
            analysis['tag_text']['columns'].append(col)
        elif col in ['num_images', 'cover_image_ocr_text', 'inner_images_ocr_text', 'cover_image_ocr_confidence', 'inner_images_ocr_confidence']:
            analysis['auxiliary']['columns'].append(col)
        elif col in ['note_id', 'user_id', 'title', 'nickname', 'file_url_list', 'first_file_id', 'content', 'tag_info']:
            analysis['metadata']['columns'].append(col)
    
    # 计算各类特征的统计信息
    for category, info in analysis.items():
        if info['columns']:
            feature_cols = info['columns']
            
            if category in ['cover_image', 'inner_images', 'title_text', 'content_text', 'tag_text']:
                # 向量特征
                feature_matrix = df[feature_cols].values
                info['stats'] = {
                    'dimensions': len(feature_cols),
                    'zero_rows': np.all(feature_matrix == 0, axis=1).sum(),
                    'total_rows': len(df),
                    'zero_percentage': np.all(feature_matrix == 0, axis=1).sum() / len(df) * 100,
                    'mean_norm': np.linalg.norm(feature_matrix, axis=1).mean(),
                    'std_norm': np.linalg.norm(feature_matrix, axis=1).std(),
                    'mean_values': feature_matrix.mean(axis=0)[:5].tolist(),  # 前5个维度的均值
                    'std_values': feature_matrix.std(axis=0)[:5].tolist()     # 前5个维度的标准差
                }
            elif category == 'auxiliary':
                # 辅助特征
                info['stats'] = {}
                for col in feature_cols:
                    if col == 'num_images' and col in df.columns:
                        info['stats']['num_images'] = {
                            'mean': df[col].mean(),
                            'std': df[col].std(),
                            'min': df[col].min(),
                            'max': df[col].max(),
                            'zero_count': (df[col] == 0).sum()
                        }
                    elif col == 'cover_image_ocr_text' and col in df.columns:
                        non_empty = (df[col].notna() & (df[col] != '')).sum()
                        info['stats']['cover_image_ocr_text'] = {
                            'non_empty': non_empty,
                            'percentage': non_empty / len(df) * 100
                        }
                    elif col == 'cover_image_ocr_confidence' and col in df.columns:
                        valid_conf = df[df[col] > 0][col]
                        info['stats']['cover_image_ocr_confidence'] = {
                            'mean': valid_conf.mean() if len(valid_conf) > 0 else 0,
                            'std': valid_conf.std() if len(valid_conf) > 0 else 0,
                            'valid_count': len(valid_conf)
                        }
    
    return analysis


def inspect_ple_parquet(file_path: str, num_rows: int = 10, show_features: bool = False, 
                       show_feature_values: bool = False, detailed_analysis: bool = True):
    """检查PLE pipeline输出的parquet文件内容"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print("="*80)
    print(f"📁 PLE Pipeline输出文件: {file_path.name}")
    print(f"📂 路径: {file_path}")
    print(f"📏 大小: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print("="*80)
    
    # 读取parquet文件
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 基本信息
    print("\n📊 基本信息:")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # PLE特征分析
    if detailed_analysis:
        print("\n🔍 PLE特征分析:")
        analysis = analyze_ple_features(df)
        
        # 封面图特征
        cover_info = analysis['cover_image']
        if cover_info['columns']:
            stats = cover_info['stats']
            print(f"\n  📷 封面图特征:")
            print(f"    维度: {stats['dimensions']}")
            print(f"    零特征行: {stats['zero_rows']}/{stats['total_rows']} ({stats['zero_percentage']:.1f}%)")
            print(f"    特征向量L2范数: mean={stats['mean_norm']:.4f}, std={stats['std_norm']:.4f}")
            if show_features:
                print(f"    前5维均值: {[f'{x:.4f}' for x in stats['mean_values']]}")
                print(f"    前5维标准差: {[f'{x:.4f}' for x in stats['std_values']]}")
        
        # 内页图特征
        inner_info = analysis['inner_images']
        if inner_info['columns']:
            stats = inner_info['stats']
            print(f"\n  🖼️ 内页图特征:")
            print(f"    维度: {stats['dimensions']}")
            print(f"    零特征行: {stats['zero_rows']}/{stats['total_rows']} ({stats['zero_percentage']:.1f}%)")
            print(f"    特征向量L2范数: mean={stats['mean_norm']:.4f}, std={stats['std_norm']:.4f}")
            if show_features:
                print(f"    前5维均值: {[f'{x:.4f}' for x in stats['mean_values']]}")
        
        # 标题文本特征
        title_info = analysis['title_text']
        if title_info['columns']:
            stats = title_info['stats']
            print(f"\n  📝 标题文本特征:")
            print(f"    维度: {stats['dimensions']}")
            print(f"    零特征行: {stats['zero_rows']}/{stats['total_rows']} ({stats['zero_percentage']:.1f}%)")
            print(f"    特征向量L2范数: mean={stats['mean_norm']:.4f}, std={stats['std_norm']:.4f}")
        
        # 内容文本特征
        content_info = analysis['content_text']
        if content_info['columns']:
            stats = content_info['stats']
            print(f"\n  📄 内容文本特征:")
            print(f"    维度: {stats['dimensions']}")
            print(f"    零特征行: {stats['zero_rows']}/{stats['total_rows']} ({stats['zero_percentage']:.1f}%)")
            print(f"    特征向量L2范数: mean={stats['mean_norm']:.4f}, std={stats['std_norm']:.4f}")
        
        # 标签文本特征
        tag_info = analysis['tag_text']
        if tag_info['columns']:
            stats = tag_info['stats']
            print(f"\n  🏷️ 标签文本特征:")
            print(f"    维度: {stats['dimensions']}")
            print(f"    零特征行: {stats['zero_rows']}/{stats['total_rows']} ({stats['zero_percentage']:.1f}%)")
            print(f"    特征向量L2范数: mean={stats['mean_norm']:.4f}, std={stats['std_norm']:.4f}")
        
        # 辅助特征
        aux_info = analysis['auxiliary']
        if aux_info['columns']:
            print(f"\n  🔧 辅助特征:")
            for col in aux_info['columns']:
                if col in aux_info['stats']:
                    stat = aux_info['stats'][col]
                    if col == 'num_images':
                        print(f"    {col}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, "
                              f"range=[{stat['min']}-{stat['max']}], 零值={stat['zero_count']}")
                    elif col == 'ocr_text':
                        print(f"    {col}: 非空={stat['non_empty']}/{len(df)} ({stat['percentage']:.1f}%)")
                    elif col == 'ocr_confidence':
                        print(f"    {col}: mean={stat['mean']:.4f}, 有效样本={stat['valid_count']}")
    
    # 列详情
    print(f"\n📋 列详细信息:")
    meta_cols = []
    feature_cols = []
    aux_cols = []
    
    for col in df.columns:
        if any(col.startswith(prefix) for prefix in ['cover_image_feat_', 'inner_image_feat_', 'title_feat_', 'content_feat_', 'tag_feat_']):
            feature_cols.append(col)
        elif col in ['num_images', 'ocr_text', 'ocr_confidence']:
            aux_cols.append(col)
        else:
            meta_cols.append(col)
    
    print(f"  元数据列: {len(meta_cols)}")
    for col in meta_cols:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"    - {col}: {dtype} (非空: {non_null}/{len(df)})")
    
    print(f"  特征列: {len(feature_cols)}")
    print(f"  辅助列: {len(aux_cols)}")
    
    # 特征向量值示例
    if show_feature_values and feature_cols:
        print(f"\n📊 特征向量示例 (第1行):")
        
        # 分类显示特征值
        for prefix, desc in [('cover_image_feat_', '封面图'), ('inner_image_feat_', '内页图'), 
                            ('title_feat_', '标题'), ('content_feat_', '内容'), ('tag_feat_', '标签')]:
            cols = [col for col in feature_cols if col.startswith(prefix)]
            if cols:
                values = df[cols].iloc[0].values
                print(f"  {desc}特征 ({len(cols)}维): {values[:10]}... (显示前10维)")
    
    # 样本数据
    print(f"\n📝 样本数据 (前{num_rows}行):")
    print("-"*80)
    
    display_cols = meta_cols + aux_cols
    display_cols = [col for col in display_cols if col in df.columns]
    
    for i in range(min(num_rows, len(df))):
        print(f"\n行 {i+1}:")
        for col in display_cols:
            value = df.iloc[i][col]
            # 截断长字符串
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."  

            print(f"  {col}: {value}")
        
        # 显示各类特征统计
        feature_categories = [
            ('cover_image_feat_', '封面图'),
            ('inner_image_feat_', '内页图'),
            ('title_feat_', '标题'),
            ('content_feat_', '内容'),
            ('tag_feat_', '标签')
        ]
        
        for prefix, desc in feature_categories:
            cols = [col for col in df.columns if col.startswith(prefix)]
            if cols:
                feat_vec = df[cols].iloc[i].values
                norm = np.linalg.norm(feat_vec)
                nonzero = np.count_nonzero(feat_vec)
                print(f"  {desc}特征: dim={len(feat_vec)}, norm={norm:.4f}, 非零={nonzero}/{len(feat_vec)}")
    
    print("\n" + "="*80)


def scan_ple_output_directory(directory: str):
    """扫描PLE输出目录，提供概览"""
    files = get_ple_parquet_files(directory)
    
    if not files:
        print(f"❌ 在 {directory} 中未找到PLE输出文件")
        return
    
    print("="*80)
    print(f"📂 PLE输出目录扫描: {directory}")
    print("="*80)
    
    print(f"\n📄 找到 {len(files)} 个文件:")
    
    total_size = 0
    total_rows = 0
    
    for i, file in enumerate(files):
        try:
            size_mb = file.stat().st_size / (1024*1024)
            total_size += size_mb
            
            # 快速读取元数据
            df = pd.read_parquet(file)
            rows = len(df)
            total_rows += rows
            cols = len(df.columns)
            
            print(f"  {i+1:2d}. {file.name}")
            print(f"      大小: {size_mb:.2f} MB, 行数: {rows:,}, 列数: {cols}")
            
            # 检查特征类型
            feature_types = []
            if any(col.startswith('cover_image_feat_') for col in df.columns):
                feature_types.append('封面图')
            if any(col.startswith('inner_image_feat_') for col in df.columns):
                feature_types.append('内页图')
            if any(col.startswith('title_feat_') for col in df.columns):
                feature_types.append('标题')
            if any(col.startswith('content_feat_') for col in df.columns):
                feature_types.append('内容')
            if any(col.startswith('tag_feat_') for col in df.columns):
                feature_types.append('标签')
            
            print(f"      特征类型: {', '.join(feature_types) if feature_types else '无'}")
            
        except Exception as e:
            print(f"  {i+1:2d}. {file.name} - ❌ 读取失败: {e}")
    
    print(f"\n📊 总计:")
    print(f"  文件数: {len(files)}")
    print(f"  总大小: {total_size:.2f} MB")
    print(f"  总行数: {total_rows:,}")
    
    if files:
        print(f"\n💡 要检查具体文件，使用:")
        print(f"  python {sys.argv[0]} {files[0]}")
        print(f"  python {sys.argv[0]} {files[0]} --show-features --show-values")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试PLE图像特征提取pipeline输出内容')
    parser.add_argument('path', help='PLE输出parquet文件路径或目录')
    parser.add_argument('--rows', '-n', type=int, default=3,
                        help='显示的样本行数 (默认: 3)')
    parser.add_argument('--show-features', '-f', action='store_true',
                        help='显示特征向量的详细统计信息')
    parser.add_argument('--show-values', '-v', action='store_true',
                        help='显示特征向量的实际值')
    parser.add_argument('--scan-all', '-a', action='store_true',
                        help='扫描目录中的所有文件（概览模式）')
    parser.add_argument('--no-analysis', action='store_true',
                        help='跳过详细的PLE特征分析')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if args.scan_all or path.is_dir():
        scan_ple_output_directory(args.path)
    else:
        inspect_ple_parquet(
            args.path, 
            args.rows, 
            args.show_features, 
            args.show_values,
            not args.no_analysis
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  # 检查单个文件")
        print("  python test_ple_pipeline_output.py /path/to/ple_batch_000001.parquet")
        print("  python test_ple_pipeline_output.py /path/to/ple_batch_000001.parquet --show-features")
        print("  python test_ple_pipeline_output.py /path/to/ple_batch_000001.parquet --show-values")
        print("")
        print("  # 扫描整个输出目录")
        print("  python test_ple_pipeline_output.py /Volumes/home/raw_data/image_features_ple_parquet/")
        print("  python test_ple_pipeline_output.py /path/to/output/dir/ --scan-all")
        print("")
        print("  # 详细分析")
        print("  python test_ple_pipeline_output.py /path/to/file.parquet -f -v --rows 5")
        sys.exit(1)
    
    main()