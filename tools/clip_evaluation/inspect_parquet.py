#!/usr/bin/env python3
"""
检查parquet文件内容的工具脚本
用于查看图像特征提取pipeline的输出

使用方法:
  python inspect_parquet.py /path/to/file.parquet
  python inspect_parquet.py /Volumes/home/raw_data/image_features_parquet/batch_000012_20250928_160332.parquet
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def inspect_parquet(file_path: str, num_rows: int = 10, show_features: bool = False, show_feature_values: bool = False):
    """检查parquet文件内容
    
    Args:
        file_path: parquet文件路径
        num_rows: 显示的行数
        show_features: 是否显示特征向量的统计信息
        show_feature_values: 是否显示特征向量的实际值
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
        
    if not file_path.suffix in ['.parquet', '.snappy']:
        print(f"❌ 不是parquet文件: {file_path}")
        return
    
    print("="*80)
    print(f"📁 文件: {file_path}")
    print(f"📏 大小: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print("="*80)
    
    # 读取parquet文件
    df = pd.read_parquet(file_path)
    
    # 基本信息
    print("\n📊 基本信息:")
    print(f"  行数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")
    print(f"  内存占用: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    # 列信息
    print("\n📋 列信息:")
    
    # 分类显示列
    meta_cols = []
    image_feat_cols = []
    text_feat_cols = []
    other_cols = []
    
    for col in df.columns:
        if col.startswith('image_feat_'):
            image_feat_cols.append(col)
        elif col.startswith('text_feat_'):
            text_feat_cols.append(col)
        elif col in ['note_id', 'user_id', 'title', 'nickname', 'file_url_list', 'first_file_id']:
            meta_cols.append(col)
        else:
            other_cols.append(col)
    
    print(f"\n  元数据列 ({len(meta_cols)}):")
    for col in meta_cols:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"    - {col}: {dtype} (非空: {non_null}/{len(df)})")
    
    print(f"\n  图像特征列: {len(image_feat_cols)} 个维度")
    if image_feat_cols and show_features:
        # 显示前几个特征的统计
        sample_cols = image_feat_cols[:5]
        for col in sample_cols:
            print(f"    - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    
    if image_feat_cols and show_feature_values:
        # 显示完整的特征值（第一行）
        print(f"\n  图像特征值 (第1行):")
        image_values = df[image_feat_cols].iloc[0].values
        print(f"    {image_values}")
    
    print(f"\n  文本特征列: {len(text_feat_cols)} 个维度")
    if text_feat_cols and show_features:
        # 显示前几个特征的统计
        sample_cols = text_feat_cols[:5]
        for col in sample_cols:
            print(f"    - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    
    if text_feat_cols and show_feature_values:
        # 显示完整的特征值（第一行）
        print(f"\n  文本特征值 (第1行):")
        text_values = df[text_feat_cols].iloc[0].values
        print(f"    {text_values}")
    
    if other_cols:
        print(f"\n  其他列 ({len(other_cols)}):")
        for col in other_cols:
            dtype = df[col].dtype
            if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                print(f"    - {col}: {dtype} (mean={df[col].mean():.4f})")
            else:
                non_null = df[col].notna().sum()
                print(f"    - {col}: {dtype} (非空: {non_null}/{len(df)})")
    
    # 特征质量检查
    print("\n🔍 特征质量检查:")
    
    # 检查图像特征
    if image_feat_cols:
        image_feat_matrix = df[image_feat_cols].values
        zero_rows = np.all(image_feat_matrix == 0, axis=1).sum()
        print(f"  图像特征全零行: {zero_rows}/{len(df)} ({zero_rows/len(df)*100:.1f}%)")
        
        # 计算特征向量的L2范数
        norms = np.linalg.norm(image_feat_matrix, axis=1)
        print(f"  图像特征L2范数: mean={norms.mean():.4f}, std={norms.std():.4f}")
    
    # 检查文本特征
    if text_feat_cols:
        text_feat_matrix = df[text_feat_cols].values
        zero_rows = np.all(text_feat_matrix == 0, axis=1).sum()
        print(f"  文本特征全零行: {zero_rows}/{len(df)} ({zero_rows/len(df)*100:.1f}%)")
        
        # 计算特征向量的L2范数
        norms = np.linalg.norm(text_feat_matrix, axis=1)
        print(f"  文本特征L2范数: mean={norms.mean():.4f}, std={norms.std():.4f}")
    
    # OCR相关
    if 'ocr_text' in df.columns:
        non_empty_ocr = (df['ocr_text'].notna() & (df['ocr_text'] != '')).sum()
        print(f"  有OCR文本: {non_empty_ocr}/{len(df)} ({non_empty_ocr/len(df)*100:.1f}%)")
        
        if 'ocr_confidence' in df.columns:
            valid_conf = df[df['ocr_confidence'] > 0]['ocr_confidence']
            if len(valid_conf) > 0:
                print(f"  OCR置信度: mean={valid_conf.mean():.4f}, std={valid_conf.std():.4f}")
    
    # 显示样本数据
    print(f"\n📝 样本数据 (前{num_rows}行):")
    print("-"*80)
    
    # 选择要显示的列（不包括特征列）
    display_cols = meta_cols + other_cols
    display_cols = [col for col in display_cols if col in df.columns]
    
    for i in range(min(num_rows, len(df))):
        print(f"\n行 {i+1}:")
        for col in display_cols:
            value = df.iloc[i][col]
            # 截断长字符串
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"  {col}: {value}")
        
        # 显示特征统计
        if image_feat_cols:
            image_feat = df[image_feat_cols].iloc[i].values
            print(f"  图像特征: dim={len(image_feat)}, "
                  f"norm={np.linalg.norm(image_feat):.4f}, "
                  f"非零={np.count_nonzero(image_feat)}/{len(image_feat)}")
        
        if text_feat_cols:
            text_feat = df[text_feat_cols].iloc[i].values
            print(f"  文本特征: dim={len(text_feat)}, "
                  f"norm={np.linalg.norm(text_feat):.4f}, "
                  f"非零={np.count_nonzero(text_feat)}/{len(text_feat)}")
    
    print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查parquet文件内容')
    parser.add_argument('file', help='parquet文件路径')
    parser.add_argument('--rows', '-n', type=int, default=5,
                        help='显示的行数 (默认: 5)')
    parser.add_argument('--show-features', '-f', action='store_true',
                        help='显示特征向量的统计信息')
    parser.add_argument('--show-values', '-v', action='store_true',
                        help='显示特征向量的实际值')
    
    args = parser.parse_args()
    
    inspect_parquet(args.file, args.rows, args.show_features, args.show_values)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python inspect_parquet.py /path/to/file.parquet")
        print("  python inspect_parquet.py /path/to/file.parquet --rows 10")
        print("  python inspect_parquet.py /path/to/file.parquet --show-features")
        print("  python inspect_parquet.py /path/to/file.parquet --show-values")
        print("  python inspect_parquet.py /path/to/file.parquet -v  # 显示特征向量实际值")
        sys.exit(1)
    
    main()