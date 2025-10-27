# 小红书CTR预估模型 (重构版)

基于深度学习的小红书笔记CTR预估系统，支持多模态特征提取和多任务学习。

## 🎯 项目概述

本项目实现了一个完整的CTR预估pipeline，包括：
- **多模态特征提取**: 基于Chinese-CLIP的图像和文本特征
- **多任务学习**: 支持CTR、点赞率、收藏率等10个任务的联合优化
- ✨ **统一训练架构**: 全新base class系统，减少80%+代码重复

## 🆕 最新架构重构 (2025.10.25)

### 🏆 重构成果
- **代码减少**: 多任务训练从3581行减少到710行 (80.2%减少)
- **代码减少**: 单任务训练从1355行减少到350行 (74%减少)  
- **统一入口**: 所有训练现在使用 `scripts/train_model.py`
- **基类系统**: 完整的base class继承架构，消除重复代码
- **增强功能**: 添加了数据采样、特征过滤、PCA降维、早停机制等功能

### 🎯 主要改进
- ✅ **单一入口点**: `scripts/train_model.py` 支持单任务和多任务训练
- ✅ **轻量级trainers**: 子类只需实现模型特定逻辑 (90%通用代码复用)
- ✅ **自动配置**: 智能参数路由和配置管理
- ✅ **代码清理**: 移除7个重复/遗留文件

## 📁 项目结构

```
xhs_ctr_model/
├── pipelines/                    # 四层架构的统一pipeline系统
│   ├── base_config.py           # 第一层：配置基类+混入类(BatchMixin, SparkMixin, MLMixin)
│   ├── base_system.py           # 第二层：系统管理基类(资源监控、错误处理)
│   ├── base_data.py             # 第三层：数据处理基类(IO抽象、特征处理)
│   ├── base_pipeline.py         # 第四层：Pipeline执行基类(模板方法模式)
│   ├── text_config.py           # 文本pipeline配置(继承多个混入类)
│   ├── text_system.py           # 文本系统管理(Spark Session管理)
│   ├── text_data.py             # 文本数据处理(读写器、处理器)
│   ├── text_pipeline.py         # 文本特征提取实现
│   ├── multimodal_config.py     # 多模态配置(支持CLIP特征选项)
│   ├── multimodal_data.py       # 多模态数据处理(异步处理)
│   ├── multimodal_pipeline.py   # 多模态特征提取实现
│   └── multimodal_processors.py # CLIP特征处理器
├── training/                    # ✨ 统一训练架构 (减少80%重复代码)
│   ├── base/                   # 统一基类系统
│   │   ├── single_task_trainer.py      # 单任务训练基类 (8步统一流程)
│   │   ├── single_task_data_processor.py   # 单任务数据处理基类
│   │   ├── single_task_feature_processor.py # 单任务特征处理基类  
│   │   ├── single_task_evaluator.py    # 单任务评估基类
│   │   ├── base_trainer.py             # MTL训练基类
│   │   ├── data_processor.py           # MTL数据处理基类
│   │   ├── feature_processor.py        # MTL特征处理基类
│   │   ├── evaluator.py                # MTL评估基类
│   │   ├── model_factory.py            # 统一模型工厂
│   │   └── feature_selector.py         # 特征选择器
│   ├── single_task/            # 轻量级单任务训练器 (350行 vs 1355行原始)
│   │   └── ctr_trainer.py      # 仅包含模型创建逻辑
│   └── multi_task/             # 轻量级多任务训练器 (710行 vs 3581行原始) 
│       ├── ple_trainer.py      # PLE实现 (仅模型特定逻辑)
│       ├── mmoe_trainer.py     # MMOE实现
│       └── pnn_mmoe_trainer.py # PNN+MMOE实现
├── scripts/                    # ✨ 统一入口脚本
│   ├── run_pipeline.py         # 特征提取pipeline入口
│   ├── train_model.py          # ✨ 统一训练入口 (单任务+多任务)
│   ├── train_mtl_model.py      # 专用多任务训练入口
│   └── evaluate_model.py       # 模型评估入口
├── models/                # 模型定义
├── config/               # 配置管理
├── data/                 # 数据存储
├── logs/                 # 日志文件
└── tests/               # 单元测试
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
```

### 2. 特征提取（重构后的统一入口）

```bash
# 运行文本特征提取（新架构）
python scripts/run_pipeline.py --stage text --batch-start 0 --batch-end 50

# 运行多模态特征提取（新架构）
python scripts/run_pipeline.py --stage multimodal \
    --input "/Volumes/home/raw_data/text_features_parquet/batch_00000_00010/dt=20250926_145314" \
    --output "/tmp/test_multimodal_output" --batch-size 1000

# 使用自定义配置
python scripts/run_pipeline.py --stage text --batch-start 0 --batch-end 100 \
    --driver-memory 32 --executor-cores 8
```

### 3. ✨ 统一模型训练 (新架构)

**单一入口点**: 所有训练现在使用 `scripts/train_model.py` 统一接口

#### 🎯 单任务训练 (支持8种DeepCTR模型)
```bash
# 基础训练
python scripts/train_model.py --task single --model DeepFM --epochs 20

# 训练所有模型并对比
python scripts/train_model.py --task single --all-models --epochs 20

# 高级特征工程训练
python scripts/train_model.py --task single --model xDeepFM \
    --sample-size 10000 --filter-zeros --min-impression 5000 \
    --use-pca --pca-components 128 --use-early-stopping \
    --label-normalization standard --epochs 30
```

#### 🎯 多任务训练 (PLE, MMOE, PNN-MMOE)
```bash
# 基础PLE训练
python scripts/train_model.py --task multi --multi-task-model PLE --epochs 30

# 高级PLE配置
python scripts/train_model.py --task multi --multi-task-model PLE \
    --tasks ctr,like_rate,comment_rate --epochs 50 \
    --shared-expert-num 4 --specific-expert-num 2 \
    --use-early-stopping --early-stopping-patience 15

# MMOE训练  
python scripts/train_model.py --task multi --multi-task-model MMOE \
    --num-experts 6 --use-pca --min-impression 10000
```

#### 🎯 自定义特征配置
```bash
# 自定义特征选择
python scripts/train_model.py --task single --model DeepFM \
    --feature-config custom --use-clip-image --use-clip-text \
    --sparse-features nickname,type,category \
    --dense-features like_num,fav_num,cmt_num
```

### 4. 模型评估

```bash
# CLIP特征评估
python scripts/evaluate_model.py --type clip --input /path/to/features

# 模型性能评估 (开发中)
python scripts/evaluate_model.py --type model --input /path/to/features
```

## 🔧 核心功能

### 重构后的统一架构系统

#### 四层基类架构
1. **配置层** (`base_config.py`): 使用混入类(Mixin)模式实现模块化配置
   - `BatchProcessingMixin`: 批处理参数管理
   - `SparkConfigMixin`: Spark集群配置
   - `MLConfigMixin`: 机器学习参数配置

2. **系统层** (`base_system.py`): 统一的资源管理和错误处理
   - Spark Session管理和优化
   - 系统资源监控和内存管理
   - 统一的错误处理和恢复机制

3. **数据层** (`base_data.py`): 抽象的数据处理接口
   - 统一的数据读写器抽象
   - 异步批处理支持
   - 数据验证和质量控制

4. **执行层** (`base_pipeline.py`): 模板方法模式的Pipeline执行框架
   - 统一的执行流程和生命周期管理
   - 可插拔的组件设计
   - 自动化的检查点和恢复机制

#### 文本特征管道
- 基于重构架构的Spark分布式处理
- 支持批处理范围指定和增量处理
- 自动内存和资源优化配置
- 统一的配置参数传递和验证

#### 多模态特征管道
- **图像特征**: 封面图和内页图的Chinese-CLIP特征提取
- **文本特征**: 标题、内容、标签的语义特征
- **特征融合**: 多种池化策略和特征对齐
- **异步处理**: 支持大规模数据的并行特征提取

### 模型训练

#### 单任务模型
支持的模型类型：
- **PNN**: Product-based Neural Network
- **DeepFM**: Deep Factorization Machine
- **WDL**: Wide & Deep Learning
- **DCN**: Deep & Cross Network
- **xDeepFM**: eXtreme Deep Factorization Machine
- **AutoInt**: Automatic Feature Interaction
- **FiBiNET**: Feature Importance and Bilinear feature Interaction

#### 🆕 新训练架构特性
- **sample-size**: 数据采样，支持大数据集的快速实验
- **filter-zeros**: 过滤零值特征，提升模型质量
- **min-impression**: 最小曝光数过滤，确保数据质量
- **label-normalization**: 标签归一化 (standard/minmax/robust/none)
- **use-pca**: CLIP特征PCA降维，减少计算复杂度
- **early-stopping**: 早停机制，防止过拟合
- **feature-config**: 预设特征配置 (all/basic/clip_only/no_image/no_text/custom)

#### 多任务模型
- **PLE**: Progressive Layered Extraction (推荐)
- **MMOE**: Multi-gate Mixture-of-Experts
- **PNN+MMOE**: 自定义PNN与MMOE结合

支持的任务：
- **CTR预测**: click_num / imp_num
- **点赞率**: like_num / click_num
- **收藏率**: fav_num / click_num
- **评论率**: cmt_num / click_num
- **分享率**: share_num / click_num
- **加粉率**: follow_from_discovery_num / click_num
- **互动率**: (like+fav+cmt+share+follow) / click_num
- **CES评分率**: (1×like + 1×fav + 4×cmt + 4×share + 4×follow) / click_num
- **曝光数预测**: log(imp_num)
- **排序分数**: sort_score2 column

### 评估系统

#### CLIP特征评估
- **特征质量**: 稀疏性、聚类分析、维度相关性
- **任务相关性**: 随机森林重要性、相关性分析
- **多模态对齐**: Cosine相似度、典型相关分析

## 📊 使用示例

### 端到端训练流程

```bash
# 1. 特征提取
python scripts/run_pipeline.py --stage all \
    --batch-start 0 --batch-end 200 \
    --enable-all-features

# 2. 模型训练
python scripts/train_model.py --task multi \
    --multi-task-model PLE \
    --tasks ctr,like_rate,comment_rate \
    --epochs 30 \
    --use-pca --pca-components 128

# 3. 特征评估
python scripts/evaluate_model.py --type clip \
    --sample-size 10000
```

### 自定义训练配置

```bash
# 使用PCA降维和零特征过滤
python scripts/train_model.py --task single \
    --model DeepFM \
    --use-pca --pca-components 128 \
    --filter-zeros \
    --min-impression 5000

# 自定义PLE架构
python scripts/train_model.py --task multi \
    --multi-task-model PLE \
    --shared-expert-num 6 \
    --specific-expert-num 3 \
    --num-levels 3 \
    --epochs 50
```

## 📈 性能指标

### 单任务CTR模型性能 (11个样本测试)
- **FNN**: Test MSE=0.014, Spearman=-1.0000 (完美排序，反向相关)
- **DNN**: Test MSE=0.015, Spearman=0.5000 (中等正相关)
- **PNN**: Test MSE=0.033, Spearman=-0.5000 (中等反向相关)

### 多任务学习优势
- 任务间信息共享，提升数据稀疏任务的性能
- PLE模型在复杂任务关系中表现更优
- 支持端到端的多目标优化

## 🛠️ 开发说明

### 🎯 特征命名标准

**重要：** 项目使用统一的特征命名标准，确保pipeline → training → inference全链路一致性。

#### CLIP特征命名格式
```python
# ✅ 正确的特征命名模式
cover_image_feat_0, cover_image_feat_1, ..., cover_image_feat_511
title_feat_0, title_feat_1, ..., title_feat_511
content_feat_0, content_feat_1, ..., content_feat_511
inner_image_feat_0, inner_image_feat_1, ..., inner_image_feat_511
tag_feat_0, tag_feat_1, ..., tag_feat_511

# ❌ 废弃的命名模式（避免使用）
cover_image_embedding_0, title_embedding_0, content_embedding_0
inner_images_embedding_0  # 注意：复数形式也是错误的
```

#### 使用特征命名标准
```python
from src.features.feature_names import (
    FeatureNames, generate_clip_features, validate_features
)

# 生成CLIP特征名
cover_features = generate_clip_features('cover_image', 512)
title_features = generate_clip_features('title', 512)

# 验证特征命名一致性
is_valid, report = validate_features(df.columns.tolist())
if not is_valid:
    print(f"特征命名问题：\n{report}")

# 检查特征类型
if FeatureNames.is_clip_feature('cover_image_feat_0'):
    feature_type = FeatureNames.get_feature_type('cover_image_feat_0')
    print(f"特征类型: {feature_type}")  # FeatureType.COVER_IMAGE
```

### 基于重构架构添加新的特征提取器

1. **创建配置类**: 在 `pipelines/` 目录下创建 `{name}_config.py`
   - 继承 `BasePipelineConfig` 和所需的混入类
   - 实现抽象方法: `validate()`, `get_output_path()`, `setup_environment()`

2. **创建系统管理类**: 创建 `{name}_system.py`
   - 继承 `BaseSystemManager`
   - 实现资源管理和错误处理逻辑

3. **创建数据处理类**: 创建 `{name}_data.py`
   - 继承 `BaseDataProcessor`
   - 实现数据读写器和处理器

4. **创建Pipeline类**: 创建 `{name}_pipeline.py`
   - 继承 `BasePipeline`
   - 实现 `run_stage()` 方法
   - 在 `scripts/run_pipeline.py` 中注册新的stage

### 配置系统扩展

重构后的架构使用混入类模式，可以灵活组合不同的配置功能：

```python
# 示例：创建支持批处理和Spark的新配置
@dataclass  
class NewConfig(BasePipelineConfig, BatchProcessingMixin, SparkConfigMixin):
    # 新的配置字段
    special_param: str = "default_value"
    
    def validate(self) -> None:
        # 使用统一的验证器
        ConfigValidator.validate_paths(self)
        ConfigValidator.validate_batch_config(self)
        ConfigValidator.validate_spark_config(self)
```

### 添加新的训练器

1. 在 `training/` 相应目录下创建训练器类
2. 实现标准的训练和评估接口
3. 在 `scripts/train_model.py` 中添加支持

### 运行测试

```bash
# 运行API测试
python tests/test_api.py

# 运行pipeline测试
python tests/test_text_pipeline.py
```

## 🔄 从旧版本迁移

如果您使用的是旧版本的代码结构，请参考以下迁移指南：

### 脚本迁移映射
- `run_local_text_pipeline_batch_ssd.py` → `python scripts/run_pipeline.py --stage text`
- `run_local_image_pipeline_chinese-clip_for_PLE.py` → `python scripts/run_pipeline.py --stage multimodal`
- `run_local_training_model.py` → `python scripts/train_model.py --task single`
- `run_local_training_model_ple.py` → `python scripts/train_model.py --task multi --multi-task-model PLE`

### 架构变化
- **新四层架构**: 配置 → 系统 → 数据 → 执行的清晰分层
- **混入类设计**: 配置功能模块化，支持灵活组合
- **统一入口**: `scripts/run_pipeline.py` 替代多个独立脚本
- **自动配置**: 系统资源自动检测和优化

### 配置参数变化
- **批处理参数**: `--batch-start`, `--batch-end` 替代文件名硬编码
- **系统参数**: `--driver-memory`, `--executor-cores` 支持动态调整
- **路径参数**: `--input`, `--output` 支持自定义路径
- **特征控制**: 通过配置文件而非命令行参数控制特征选项

### 数据路径
新架构自动管理数据路径，支持时间戳后缀和批次信息，无需手动指定复杂路径。

## 📝 配置说明

项目使用 `.env` 文件进行配置管理：

```bash
# 数据路径配置
DATA_BASE_PATH=/Volumes/home/raw_data
TEXT_FEATURES_PATH=${DATA_BASE_PATH}/text_features_parquet
IMAGE_FEATURES_PATH=${DATA_BASE_PATH}/image_features_ple_parquet

# 模型配置
MODEL_OUTPUT_PATH=models
LOG_LEVEL=INFO

# 训练参数
DEFAULT_EPOCHS=20
DEFAULT_BATCH_SIZE=256
MIN_IMPRESSION_THRESHOLD=5000
```

## 🤝 贡献指南

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 详情请查看 [LICENSE](LICENSE) 文件。

## 🆘 问题反馈

如遇到问题，请通过以下方式反馈：
1. 查看 `logs/` 目录下的日志文件
2. 在GitHub Issues中提交问题
3. 包含完整的错误信息和运行环境

## 🎉 重构成果总结

### 代码质量提升
- **代码减少**: 总计减少80%+重复代码 (>4000行 → <1100行)
- **架构清理**: 移除7个重复/遗留文件
- **统一接口**: 单一训练入口，支持所有模型类型
- **增强功能**: 添加了10+个新的训练功能和参数

### 文件清理记录 (2025.10.25)
**✅ 移除的重复文件**:
- `train_model.py` (根目录旧版本)
- `run_local_training_model*.py` (5个遗留训练脚本)
- `simplified_multi_task_config.py` (未使用配置)
- `task_specific_feature_masking.py` (遗留特征掩码)
- `text_pipeline_backup.py`, `text_pipeline_optimized.py` (管道备份文件)

**✅ 保留的核心文件**:
- `scripts/train_model.py` - ✨ 统一训练入口
- `training/base/` - ✨ 统一基类架构
- `training/single_task/ctr_trainer.py` - 轻量级单任务训练器
- `training/multi_task/*.py` - 轻量级多任务训练器

## 📚 相关资源

- [CLAUDE.md](CLAUDE.md) - Claude Code的详细项目说明和架构文档
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 详细架构说明
- [DeepCTR文档](https://deepctr-doc.readthedocs.io/) - DeepCTR库官方文档
- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) - 中文CLIP模型