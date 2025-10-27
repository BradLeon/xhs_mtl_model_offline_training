# CLIP特征评估工具套件

这是一个完整的CLIP特征评估工具套件，用于评估小红书CTR预测模型中CLIP图像和文本特征的质量和有效性。

## 目录结构

```
tools/clip_evaluation/
├── README.md                    # 使用说明（本文件）
├── METRICS_GUIDE.md            # 指标说明和基线值
├── evaluate_clip_features.py   # 主评估脚本（统一版本）
├── inspect_parquet.py         # 数据检查工具
├── CLIP_EVALUATION_EXPLAINED.md # 详细的指标解释文档
└── evaluation_results/        # 评估结果输出目录
    ├── evaluation_report.html  # HTML格式的评估报告
    ├── evaluation_plots.html   # 交互式可视化图表
    ├── evaluation_results.json # JSON格式的详细结果
    └── problem_analysis.md     # 问题分析和改进建议
```

## 核心功能

### 1. 三层评估体系

#### 视觉理解层 (Visual Understanding Layer)
- **聚类质量评估**: 使用Silhouette Score和Davies-Bouldin指数评估特征的聚类效果
- **图文语义一致性**: 计算同一note的图文特征相似度 vs 随机配对相似度，使用Cohen's d测量效应大小

#### 表征质量层 (Feature Quality Layer)  
- **方差分析**: 检查特征维度的方差分布，识别零方差和低方差维度
- **维度相关性**: 分析特征维度间的冗余性
- **PCA分析**: 评估降维潜力，确定保留多少维度可以维持95%/99%方差

#### 任务适配层 (Task Adaptation Layer)
- **CTR相关性分析**: 计算每个特征维度与CTR的Pearson相关系数
- **预测能力评估**: 使用逻辑回归(AUC)和随机森林(R²)评估预测性能
- **特征重要性**: 基于随机森林的特征重要性分析
- **消融实验**: 系统性地移除不同特征组合来量化各特征类型的贡献

### 2. 内置问题诊断

评估完成后，系统会自动分析结果并生成问题诊断报告，包括：
- 根本原因分析（如L2归一化的影响）
- 具体改进方案（短期、中期、长期）
- 验证步骤建议

### 3. 消融实验

量化评估不同特征组合的贡献：
- `baseline_all`: 所有特征
- `no_image`: 移除图像特征
- `no_text`: 移除文本特征  
- `no_clip`: 移除所有CLIP特征
- `only_image`: 仅图像特征
- `only_text`: 仅文本特征
- `only_clip`: 仅CLIP特征

## 使用方法

### 基本评估

```bash
# 评估指定目录下的特征数据
python evaluate_clip_features.py --input /path/to/parquet/files --output ./results

# 限制样本数量（加速评估）
python evaluate_clip_features.py --input /path/to/data --max-samples 10000

# 评估单个parquet文件
python evaluate_clip_features.py --input /path/to/file.parquet
```

### 仅分析现有结果

```bash
# 分析已有的评估结果，生成问题诊断报告
python evaluate_clip_features.py --analyze-only --results ./results/evaluation_results.json
```

### 数据检查

```bash
# 检查parquet文件内容
python inspect_parquet.py /path/to/file.parquet

# 显示特征统计信息
python inspect_parquet.py /path/to/file.parquet --show-features

# 显示特征值（用于调试）
python inspect_parquet.py /path/to/file.parquet --show-values
```

## 输出结果解读

### 1. HTML报告 (evaluation_report.html)

包含：
- 三层评估的详细结果
- 彩色指标展示（绿色=好，黄色=警告，红色=问题）
- 消融实验结果表格
- 自动生成的优化建议
- 集成的问题分析报告

### 2. 交互式图表 (evaluation_plots.html)

9个子图展示：
- 聚类质量曲线
- 语义一致性分布
- 特征方差分布
- PCA累积方差
- 特征-CTR相关性
- 预测性能指标
- 特征L2范数分布
- 维度方差散点图
- 特征重要性对比

### 3. JSON结果 (evaluation_results.json)

包含所有数值结果的机器可读格式，方便：
- 程序化分析
- 结果对比
- 自动化流程集成

### 4. 问题分析报告 (problem_analysis.md)

Markdown格式的详细分析，包括：
- 具体问题识别
- 根本原因分析  
- 分层改进方案
- 验证步骤指导

## 重要指标说明

### 聚类质量指标

| 指标 | 优秀 | 良好 | 可接受 | 问题 |
|------|------|------|--------|------|
| Silhouette Score | >0.7 | 0.5-0.7 | 0.3-0.5 | <0.3 |
| Clustering Purity | >0.7 | 0.5-0.7 | 0.4-0.5 | <0.4 |

**注意**: Silhouette Score过高(>0.95)可能表示异常，需要检查数据质量。

### CTR预测性能

| 指标 | 优秀 | 良好 | 基础 | 问题 |
|------|------|------|------|------|
| AUC | >0.75 | 0.70-0.75 | 0.65-0.70 | <0.65 |
| R² | >0.20 | 0.10-0.20 | 0.05-0.10 | <0.05 |
| Spearman | >0.40 | 0.30-0.40 | 0.20-0.30 | <0.20 |

### 语义一致性

| Cohen's d | 解释 |
|-----------|------|
| >1.0 | 强效应，图文对齐很好 |
| 0.5-1.0 | 中等效应，对齐可接受 |
| 0.2-0.5 | 弱效应，对齐较差 |
| <0.2 | 几乎无效应，需要改进 |

## 常见问题和解决方案

### 1. 低方差问题

**现象**: 99%+的维度是低方差
**原因**: L2归一化导致特征被压缩到单位球面
**解决**: 这是正常的，重点关注角度分布和余弦相似度

### 2. 预测性能差

**现象**: AUC<0.6, R²<0.05
**可能原因**:
- CLIP特征不适合CTR任务
- 图片下载失败率高
- 缺乏领域适配

**解决方案**:
- 运行消融实验确认特征价值
- 检查图片URL有效性
- 考虑fine-tune CLIP模型

### 3. 语义对齐弱

**现象**: Cohen's d < 0.5
**原因**: 
- 通用CLIP模型不适合小红书内容
- 图文内容本身关联性弱
- 数据质量问题

**解决方案**:
- 在领域数据上fine-tune
- 检查数据质量
- 尝试其他多模态模型

## 扩展使用

### 集成到训练流程

```python
from evaluate_clip_features import CLIPFeatureEvaluator

# 在模型训练前评估特征质量
evaluator = CLIPFeatureEvaluator(output_dir="./feature_analysis")
results = evaluator.run_evaluation(data_path="./features.parquet")

# 基于评估结果决定特征处理策略
if results['task_adaptation']['ablation']['contributions']['image_features']['is_positive']:
    print("保留图像特征")
else:
    print("移除图像特征")
```

### 批量数据评估

```bash
#!/bin/bash
# 批量评估不同数据批次
for file in /data/batches/*.parquet; do
    echo "Evaluating $file"
    python evaluate_clip_features.py --input "$file" --output "./results/$(basename $file .parquet)"
done
```

## 更新日志

- **v2.0**: 合并analyze_clip_issues.py，增加内置问题诊断
- **v1.5**: 添加消融实验功能
- **v1.0**: 基础三层评估体系

## 贡献指南

欢迎提交Issue和Pull Request来改进这个工具套件。

## 许可证

内部工具，仅供研发团队使用。