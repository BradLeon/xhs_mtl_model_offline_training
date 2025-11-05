<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recent Critical Fixes (2025-11-05)

### Fixed Online-Offline Prediction Inconsistency

**Problem**: Online inference service (`xhs_mtl_model_online_reasoning`) was producing incorrect predictions with values of 0.0 for ctr, sort_score, and comment_rate due to model architecture mismatch.

**Root Cause**:
- Online service used `task_types=['regression']` while the model was trained with `task_types=['binary']`
- This caused output layer activation function mismatch (linear vs sigmoid)
- Raw predictions were out of range (e.g., -1.763, 20.473) and got clipped to 0.0

**Solution Applied** (`app/services/model_inference.py`):
1. Changed `task_types=['regression']` → `task_types=['binary']` (Line 361)
2. Added missing `dnn_dropout` parameter (Line 369)
3. Reordered parameters to match offline training code
4. Fixed import path: `from offline_training.training.base.pnn_mmoe_model import PNN_MMOE`

**Files Modified**:
- `/Users/liuchao/AI/xhs-ctr-project/xhs_mtl_model_online_reasoning/app/services/model_inference.py`

**Verification**:
- Online predictions now match offline predictions
- All rate values naturally fall within [0, 1] range without clipping
- ctr, sort_score, comment_rate now show correct non-zero values

## Project Overview

This is a comprehensive CTR (Click-Through Rate) prediction system for Xiaohongshu (小红书) notes. The repository contains THREE implementations:

### 1. Initial Implementation (Root Directory)
Complete CTR prediction system with CLIP multimodal features and PNN/FNN/DNN models for regression prediction. Uses real data from 11 Xiaohongshu notes with web-based prediction service.

### 2. New Refactored Architecture (Recommended)
Clean, layered architecture with unified execution interface:
```
config/ → src/pipelines/ → local/emr execution → run_pipeline.py
```
- **Unified Entry**: Single `run_pipeline.py` for all scenarios
- **Clear Separation**: config / source code / execution scripts
- **Code Reuse**: BasePipeline abstract class for all pipelines
- **Environment Adaptive**: Auto-detects local vs EMR mode

### 3. Production Pipeline (`xhs_ctr_production/`)
Enterprise-grade, distributed data processing pipeline implementing:
```
OSS → Spark Text Features → CLIP Image Features → Feature Merge → PetaStorm → DeepCTR Training
```
Supports TB-scale data processing with 6 different deep learning architectures.
**Note**: This directory was partially deleted but has been restored using `restore_xhs_production.sh`

## Core Architecture

### Data Pipeline
- **Real Data Source**: `data/xhs_note_stats_data_demo.json` contains 11 real Xiaohongshu notes with statistics
- **Data Fields**: title, desc, image_list (URLs), nickname, impression, click, like, collect, comment, engage
- **CTR Calculation**: target = click / impression (regression task, not binary classification)
- **Feature Engineering**: Extract cover image + title + user nickname + like count

### CLIP Feature Extraction Pipeline
- **Image Features**: Extract 512-dim features from cover image (first URL in image_list) using CLIP ViT-B/32
- **Text Features**: Extract 512-dim features from title text using CLIP
- **Fallback Mechanism**: Use zero vectors when image URLs are inaccessible
- **Testing**: `test_feature_extraction.py` provides comprehensive testing of CLIP extraction

### ML Model Architecture
- **Model Types**: Three neural networks implemented and compared
  - **PNN (Product-based Neural Network)**: Inner product for feature interactions
  - **FNN (Factorization-supported Neural Network)**: FM layer + DNN
  - **DNN (Deep Neural Network)**: Standard feedforward network
- **Input Features**:
  - CLIP image features (512 dims)
  - CLIP text features (512 dims) 
  - Nickname one-hot encoding (11 dims for 11 unique users)
  - Like count (standardized)
  - Total input: 1036 dimensions
- **Training**: PyTorch implementation with Adam optimizer, ReduceLROnPlateau scheduler
- **Evaluation**: MSE/MAE/R² for regression + Spearman/Kendall Tau for ranking ability

### Model Performance Results (11 real samples)
```
FNN: Test MSE=0.014, Spearman=-1.0000, Kendall=-1.0000 (Perfect reverse ranking)
DNN: Test MSE=0.015, Spearman=0.5000, Kendall=0.3333 (Medium positive ranking) 
PNN: Test MSE=0.033, Spearman=-0.5000, Kendall=-0.3333 (Medium reverse ranking)
```
**Key Finding**: FNN has perfect ranking ability but with reverse correlation (negative Spearman/Kendall)

### API Service Architecture
- **Framework**: FastAPI with CORS middleware for cross-origin requests
- **Current Model**: Uses PNN model (can be switched to any trained model)
- **Endpoints**:
  - `POST /predict`: Full prediction with image upload + title + nickname + like_count
  - `POST /predict_simple`: Simplified prediction with just title + nickname + like_count
  - `GET /health`: Service health check
  - `GET /model_info`: Model information and features
  - `POST /train`: Trigger model retraining
- **Response Format**: Simple `{pctr: float, status: string, message: string}` without ranking adjustments

### Web Frontend
- **File**: `frontend.html` - Self-contained HTML with embedded CSS/JavaScript
- **Features**:
  - Responsive design with gradient styling
  - Image upload with preview
  - Real-time progress indication during prediction
  - Smart suggestions based on predicted CTR values
  - Error handling and API connection detection
- **Integration**: Communicates with FastAPI backend via fetch API

## Key Commands

### Using New Refactored Architecture (Recommended)

```bash
# Setup
pip install -r requirements.txt

# Run individual pipelines with the new base class architecture
python scripts/run_pipeline.py --stage text \
    --input "/data/note_info_parquet" \
    --output "/data/text_features" \
    --batch-start 0 --batch-end 10

python scripts/run_pipeline.py --stage multimodal \
    --input "/data/text_features_parquet/batch_00000_00010" \
    --output "/data/multimodal_features" \
    --batch-size 1000

# Full pipeline execution (text → multimodal)
python scripts/run_pipeline.py --stage full \
    --input "/data/note_info_parquet" \
    --output "/data/final_features" \
    --batch-start 0 --batch-end 10

# Advanced options
python scripts/run_pipeline.py --stage multimodal \
    --input "/data/text_features" \
    --output "/data/multimodal_features" \
    --enable-all-features \
    --gpu-batch-size 16 \
    --max-workers 4
```

### Using Initial Implementation

```bash
# Model Training (Train all models and compare)
python train_model.py

# Start API Server (PNN model)
python api_server.py

# API Testing
python test_api_new.py

# Frontend Usage
Double-click `frontend.html` to open in browser
```

### Restore xhs_ctr_production (if deleted)

```bash
chmod +x restore_xhs_production.sh
./restore_xhs_production.sh
```

## Important Data Flow

1. **Real Data Input**: JSON with 11 real Xiaohongshu notes including impression/click/engagement metrics
2. **Feature Engineering**: 
   - Extract CTR = click/impression as regression target
   - Extract cover image (first URL) + title + nickname + like_count
3. **CLIP Processing**: Download images from URLs, extract joint embeddings with fallback to zero vectors
4. **Model Training**: Compare PNN/FNN/DNN with ranking evaluation (Spearman/Kendall Tau)
5. **API Prediction**: FastAPI serves PNN model with simple CTR prediction output
6. **Web Interface**: HTML frontend provides user-friendly input and result display

## Model Features Structure

### Input Features (Total: 1036 dimensions)
- **CLIP Image Features**: 512 dims from cover image
- **CLIP Text Features**: 512 dims from title text
- **Nickname One-hot**: 11 dims (one for each unique nickname in dataset)  
- **Like Count**: 1 dim (standardized numerical feature)

### No Sparse Categorical Features
Unlike the original design, we don't use traditional categorical features like category, has_video, post_hour, etc. The focus is on multimodal CLIP features + user identity (nickname) + engagement signal (like_count).

## API Usage Patterns

### Current Implementation (PNN Model)
```python
# Simple prediction
response = requests.post("http://localhost:8000/predict_simple", json={
    "title": "救命！终于有人把国考和省考说清楚了😭",
    "nickname": "高途考公", 
    "like_count": 7233
})
# Returns: {"pctr": 0.1234, "status": "success", "message": "..."}

# Full prediction with image
files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
data = {"title": "标题", "nickname": "昵称", "like_count": 1000}
response = requests.post("http://localhost:8000/predict", files=files, data=data)
```

## Development Notes

- **Small Dataset**: Currently uses 11 real samples for proof-of-concept and model comparison
- **CLIP Model**: Downloads ~340MB on first run (openai/clip-vit-base-patch32 → ViT-B/32)
- **Training Strategy**: Regression task using MSE loss, not binary classification
- **Model Storage**: Saves to `models/` directory with separate files per model type
- **Ranking Interpretation**: Negative Spearman/Kendall indicates reverse correlation but perfect ranking ability
- **Frontend Integration**: Complete web interface requires no additional setup beyond opening HTML file

## Data Requirements and Format

Real data should be in JSON format with these fields:
- `title`: Note title text
- `image_list`: Comma-separated image URLs (uses first as cover)
- `nickname`: Author username  
- `impression`, `click`: For CTR calculation
- `like`: Engagement metric used as feature
- Other fields (desc, collect, comment, engage) available but not currently used

## Architecture Decisions

### Why PNN in API vs FNN Best Performance?
- **FNN has perfect ranking** (Spearman=-1.0) but with reverse correlation
- **PNN provides simpler interpretation** with direct CTR values
- **API design favors simplicity** over complex ranking adjustments
- Easy to switch models by changing `model_type='PNN'` to `model_type='FNN'` in api_server.py

### Why Regression vs Classification?
- **Real-world CTR values** are continuous (0.1315 ± 0.0369 in our dataset)
- **Ranking matters more than binary decisions** for content recommendation
- **MSE loss captures prediction accuracy** while Spearman/Kendall measure ranking quality

### Why CLIP Features vs Traditional Features?
- **Multimodal understanding** of image+text semantics
- **Transfers better** to diverse content types vs manually engineered features  
- **Captures visual-textual alignment** important for social media engagement
- **Scalable** - no need for category taxonomies or manual feature engineering

## New Refactored Architecture (Recommended)

### Directory Structure
```
pipelines/                    # Core pipeline implementations
├── base_config.py           # Configuration base classes and mixins
├── base_system.py           # System management base classes
├── base_data.py             # Data processing base classes
├── base_pipeline.py         # Pipeline execution base classes
├── text_config.py           # Text pipeline configuration
├── text_system.py           # Text pipeline system management
├── text_data.py             # Text pipeline data processing
├── text_pipeline.py         # Text pipeline implementation
├── multimodal_config.py     # Multimodal pipeline configuration
├── multimodal_data.py       # Multimodal pipeline data processing
├── multimodal_pipeline.py   # Multimodal pipeline implementation
└── multimodal_processors.py # Multimodal feature processors

scripts/                     # Execution entry points
├── run_pipeline.py          # Unified pipeline execution entry
└── other execution scripts

src/models/                  # Model implementations
├── ctr_models.py           # PNN/FNN/DNN models
└── deepctr_model.py        # DeepCTR integration

xhs_ctr_production/         # Production pipeline (restored)
├── config/                 # Production configuration
├── data_pipeline/          # Production data processing
├── model/                  # Production model training
└── scripts/                # Production execution scripts
```

### Key Design Principles

1. **Layered Abstraction**: Four-layer base class hierarchy for maximum code reuse
   - `base_config.py`: Configuration abstraction with mixins
   - `base_system.py`: System monitoring and resource management
   - `base_data.py`: Data processing components
   - `base_pipeline.py`: Pipeline execution framework
2. **Multiple Inheritance**: Mixin classes for feature composition
3. **Template Method Pattern**: Unified execution flow with customizable hooks
4. **Factory Pattern**: Dynamic component creation based on configuration
5. **Unified Interface**: Consistent API across all pipeline types

### Architecture Advantages

- **Maximum Code Reuse**: Common functionality abstracted into base classes
- **Type Safety**: Strong typing with abstract base classes
- **Extensibility**: Easy to add new pipeline types by inheriting base classes
- **Maintainability**: Clear separation of concerns and consistent patterns
- **Backward Compatibility**: Existing code continues to work with alias classes

### Base Class Architecture Details

#### Configuration Layer (`base_config.py`)
```python
# Abstract base configuration
BasePipelineConfig(ABC)
├── input_path, output_path (required)
├── batch_size, checkpoint_interval (common)
└── Abstract methods: validate(), get_output_path(), setup_environment()

# Mixin classes for feature composition
BatchProcessingMixin    # batch_start, batch_end, get_batch_info()
SparkConfigMixin       # Spark-specific configuration
MLConfigMixin          # ML model configuration

# Concrete implementations
TextConfig(BasePipelineConfig, BatchProcessingMixin, SparkConfigMixin)
MultimodalConfig(BasePipelineConfig, BatchProcessingMixin, MLConfigMixin)
```

#### System Management Layer (`base_system.py`)
```python
# Abstract system monitoring
BaseSystemMonitor(ABC)
├── Common monitoring: memory, CPU, disk, progress tracking
├── Background thread with periodic status reports
└── Abstract method: _print_custom_metrics()

# Specialized monitors
SparkSystemMonitor(BaseSystemMonitor)      # Spark-specific metrics
MultimodalSystemMonitor(BaseSystemMonitor) # ML-specific metrics

# Resource management
BaseResourceManager
├── Memory cleanup, temp file management
├── Disk space checking, file descriptor limits
└── System information collection
```

#### Data Processing Layer (`base_data.py`)
```python
# Abstract data components
BaseCheckpointManager   # Progress saving/loading with versioning
BaseDataReader(ABC)     # Data reading abstraction
BaseDataWriter(ABC)     # Data writing abstraction
DataProcessor(ABC)      # Data processing abstraction

# Utility classes
DataPipelineStats      # Processing statistics and metrics
ErrorHandler          # Retry logic and error recovery
FilePatternMatcher     # File discovery and batch patterns
BatchDataMixin         # Batch processing utilities
```

#### Pipeline Execution Layer (`base_pipeline.py`)
```python
# Template method pattern for unified execution flow
BasePipeline(ABC)
├── run(): 1.Initialize → 2.PreCheck → 3.Execute → 4.Finalize → 5.Cleanup
├── Component management: monitor, resource_manager, stats, error_handler
└── Abstract method: _execute_pipeline()

# Specialized pipeline bases
SparkPipelineBase(BasePipeline)     # Spark session management
├── Abstract method: _create_spark_session()
└── Automatic Spark cleanup

AsyncPipelineBase(BasePipeline)     # Async processing support
├── Async event loop management
└── Abstract method: _execute_async_pipeline()

# Concrete implementations
TextPipeline(SparkPipelineBase)     # Text feature extraction
MultimodalPipeline(AsyncPipelineBase) # Multimodal feature extraction
```

### Configuration Examples

#### Text Pipeline Configuration
```python
config = TextConfig(
    input_path="/data/note_info_parquet",
    output_path="/data/text_features",
    batch_start=0, batch_end=10,           # BatchProcessingMixin
    driver_memory_gb=32, executor_cores=8, # SparkConfigMixin
    min_impression=500                     # Text-specific
)
```

#### Multimodal Pipeline Configuration
```python
config = MultimodalConfig(
    input_path="/data/note_info_parquet",
    output_path="/data/multimodal_features",
    batch_start=0, batch_end=10,           # BatchProcessingMixin
    model_name="ViT-B-16", gpu_batch_size=8, # MLConfigMixin
    enable_cover_image=True,               # Multimodal-specific
    enable_inner_images=True
)
```

## Production Pipeline Architecture (`xhs_ctr_production/`)

### Data Pipeline Components
- **OSS Connector** (`data_pipeline/oss_connector.py`): Aliyun OSS integration with Parquet support
- **Text Feature Pipeline** (`data_pipeline/text_feature_pipeline.py`): Spark-based text processing (length, sentiment, categories, temporal features)
- **Image Feature Pipeline** (`data_pipeline/image_feature_pipeline.py`): CLIP ViT-B/32 extracts 512-dim embeddings
- **Feature Merger** (`data_pipeline/feature_merger.py`): Spark Join operations with missing value handling

### Model Training Architecture  
- **PetaStorm Loader** (`model/data_loader.py`): Converts Parquet to TensorFlow Dataset for distributed training
- **DeepCTR Models** (`model/deepctr_model.py`): Production-ready implementations of:
  - **DeepFM**: FM + DNN for sparse-dense feature interaction
  - **PNN**: Product-based Neural Network with inner/outer products
  - **FNN**: Factorization-supported Neural Network
  - **Wide&Deep**: Google's memory + generalization architecture
  - **DCN**: Deep & Cross Network with automatic feature crossing
  - **xDeepFM**: Enhanced DeepFM with CIN layers

### Feature Engineering
- **Sparse Features**: category, hour_of_day, day_of_week, has_emoji, has_hashtag (label encoded)
- **Dense Features**: text lengths, engagement metrics, content richness (standardized)
- **CLIP Features**: 512-dim multimodal embeddings with special projection layers

### Execution Framework (`scripts/`)
- **Full Pipeline** (`run_full_pipeline.py`): Complete orchestration with step control
  - Support for `--skip-text`, `--skip-image`, `--skip-merge`, `--skip-training` flags
  - `--only` flag for running specific components
- **Individual Scripts**: `run_text_pipeline.py`, `run_image_pipeline.py`, `run_feature_merge.py`, `run_training.py`
- **Error Recovery**: Comprehensive logging and checkpoint management

### Configuration Management (`config/`)
- **Data Config** (`data_config.py`): OSS paths, processing thresholds, data schemas, PetaStorm settings
- **Model Config** (`model_config.py`): Architecture selection, hyperparameters, feature specifications
- **Spark Config** (`spark_config.py`): Cluster resources, optimization settings, executor configuration

### Production Pipeline Commands

**Note**: The xhs_ctr_production directory was accidentally deleted but has been restored.
Use `restore_xhs_production.sh` if you need to restore it again.

```bash
# Restore if needed
chmod +x restore_xhs_production.sh
./restore_xhs_production.sh

# Full pipeline execution
cd xhs_ctr_production/scripts/
python run_full_pipeline.py

# Step-by-step execution
python run_text_pipeline.py      # Spark text feature processing
python run_image_pipeline.py     # CLIP image feature extraction  
python run_feature_merge.py      # Feature joining and enhancement
python run_training.py           # DeepCTR model training

# Selective execution
python run_full_pipeline.py --skip-image      # Skip image processing
python run_full_pipeline.py --only training   # Run only model training
```

### Production Pipeline Features
- **Scalability**: Handles TB-scale datasets with Spark distributed processing
- **Fault Tolerance**: Automatic retry, graceful degradation, checkpoint recovery
- **Monitoring**: Comprehensive logging, metrics tracking, training reports
- **Flexibility**: Configurable model architectures, feature engineering pipelines
- **Production Ready**: Complete testing suite, documentation, deployment scripts

## Architecture Update (Latest - 2025.10.23)

### Complete Base Class Refactoring
The project has undergone a comprehensive refactoring to implement a unified base class architecture:

1. **Four-Layer Base Class Hierarchy**: 
   - `base_config.py` - Configuration abstraction with mixin classes
   - `base_system.py` - System monitoring and resource management
   - `base_data.py` - Data processing components and utilities
   - `base_pipeline.py` - Pipeline execution framework with template method pattern

2. **Unified Pipeline Implementations**:
   - `TextPipeline(SparkPipelineBase)` - Inherits from Spark-specific base class
   - `MultimodalPipeline(AsyncPipelineBase)` - Inherits from async-specific base class
   - All pipelines follow the same execution flow: Initialize → PreCheck → Execute → Finalize → Cleanup

3. **Configuration System Improvements**:
   - Multiple inheritance with mixins (`BatchProcessingMixin`, `SparkConfigMixin`, `MLConfigMixin`)
   - Smart parameter validation based on configuration class capabilities
   - Automatic environment detection and dynamic resource allocation

4. **Enhanced Reliability**:
   - Fixed import path issues and configuration parameter passing
   - Comprehensive error handling and retry mechanisms
   - Improved logging and progress tracking
   - Complete backward compatibility through alias classes

### 🆕 Training Architecture Refactoring (Latest Update - 2025.10.25)

**Major Addition**: Complete unified training base class architecture that eliminates 80%+ code duplication:

#### Unified Training System (`training/base/` + `scripts/train_model.py`)

**Key Achievement**: Reduced 3581 lines of MTL code to 710 lines (80.2% reduction) and 1355 lines of single-task code to ~350 lines (74% reduction) through unified base class architecture.

1. **Single Entry Point** (`scripts/train_model.py`):
   - ✨ **Unified Interface**: Both single-task and multi-task training through one script
   - ✨ **Smart Parameter Handling**: Automatic configuration routing based on task type
   - ✨ **Enhanced Features**: All advanced training capabilities in one place

2. **Unified Training Base Architecture** (`training/base/`):
   - **Single-Task Components**: `BaseSingleTaskTrainer`, `BaseSingleTaskDataProcessor`, `BaseSingleTaskFeatureProcessor`, `BaseSingleTaskEvaluator`
   - **Multi-Task Components**: `BaseMTLTrainer`, `BaseMTLDataProcessor`, `BaseMTLFeatureProcessor`, `BaseMTLEvaluator`
   - **Shared Utilities**: `FeatureSelector`, `ModelFactory` with automatic model creation
   - **Lightweight Trainers**: Child classes only implement model-specific logic (90% code reduction)

3. **Enhanced Training Features**:
   - 🆕 **sample-size**: Data sampling support for large datasets
   - 🆕 **filter-zeros**: Zero feature filtering capability  
   - 🆕 **min-impression**: Minimum impression threshold filtering
   - 🆕 **label-normalization**: Standard/MinMax/Robust scaling options
   - 🆕 **use-pca**: CLIP feature PCA dimensionality reduction
   - 🆕 **early-stopping**: Comprehensive early stopping mechanism
   - 🆕 **feature-config**: Feature configuration presets (all/basic/clip_only/no_image/no_text/custom)

2. **Unified Feature Processing** (`feature_processor.py`):
   - **Feature Naming Standardization**: Fixed all `*_embedding_` → `*_feat_` naming inconsistencies
   - **Automatic Validation**: Uses `src/features/feature_names.py` for consistency checks
   - **Label Normalization**: Standard/MinMax/Robust scaling with inverse transform support
   - **Grouped PCA**: Feature-type-specific PCA (cover_image, title, content, etc.)
   - **DeepCTR Integration**: Automatic feature column generation for all models

3. **Unified Data Loading** (`data_loader.py`):
   - **Memory Optimization**: Automatic dtype conversion and memory reduction
   - **Sample-Size Support**: Intelligent sampling for large datasets
   - **Quality Filtering**: Data validation and anomaly detection
   - **Multi-File Processing**: Robust error recovery for batch processing

4. **Unified Model Evaluation** (`evaluator.py`):
   - **Comprehensive Metrics**: MSE, MAE, R², Spearman, Kendall Tau for regression tasks
   - **Multi-Task Support**: Evaluation across multiple tasks simultaneously
   - **Visualization**: Automatic plot generation for model performance
   - **Model Comparison**: Side-by-side comparison utilities

#### 🎯 Feature Naming Standardization (Critical Fix)

**Problem Solved**: Fixed feature naming inconsistencies between pipeline output and training input:

**Correct Patterns** (Pipeline Output):
```python
# ✅ Now standardized across entire codebase
'cover_image_feat_0', 'cover_image_feat_1', ...    # Cover image CLIP features
'title_feat_0', 'title_feat_1', ...                # Title text CLIP features  
'content_feat_0', 'content_feat_1', ...            # Content text CLIP features
'inner_image_feat_0', 'inner_image_feat_1', ...    # Inner image CLIP features (singular form)
'tag_feat_0', 'tag_feat_1', ...                    # Tag CLIP features
'num_images'                                        # Image count metadata
```

**Fixed Deprecated Patterns**:
```python
# ❌ Old patterns (now completely removed from training code)
'cover_image_embedding_0', 'title_embedding_0', 'content_embedding_0'
'inner_images_embedding_0'  # Plural form was also incorrect
```

**Unified Feature Standard** (`src/features/feature_names.py`):
- Centralized feature naming constants and validation functions
- Automatic consistency checking with detailed error reporting
- Feature type identification and grouping utilities
- Conversion tools for legacy patterns

#### 🎯 New Unified Training Commands

**Single Entry Point**: All training now uses `scripts/train_model.py` with automatic task routing.

**Single-Task Training** (All DeepCTR Models Supported):
```bash
# Basic single-task training
python scripts/train_model.py --task single --model DeepFM --epochs 20

# Train all models and compare
python scripts/train_model.py --task single --all-models --epochs 20

# Advanced data processing and feature engineering
python scripts/train_model.py --task single --model xDeepFM \
    --sample-size 10000 --filter-zeros --min-impression 5000 \
    --use-pca --pca-components 128 --use-early-stopping

# Custom feature selection
python scripts/train_model.py --task single --model DeepFM \
    --feature-config custom --use-clip-image --use-clip-text \
    --sparse-features nickname,type,category \
    --dense-features like_num,fav_num,cmt_num
```

**Multi-Task Training** (PLE, MMOE, PNN-MMOE):
```bash
# Basic multi-task training
python scripts/train_model.py --task multi --multi-task-model PLE --epochs 30

# Advanced multi-task with custom tasks
python scripts/train_model.py --task multi --multi-task-model PLE \
    --tasks ctr,like_rate,comment_rate --epochs 50 \
    --shared-expert-num 4 --specific-expert-num 2

# MMOE with feature engineering
python scripts/train_model.py --task multi --multi-task-model MMOE \
    --num-experts 6 --use-pca --pca-components 256 \
    --min-impression 10000 --min-click 500
```

**Advanced Training Options**:
```bash
# Comprehensive training with all options
python scripts/train_model.py --task single --model DeepFM \
    --input /path/to/data --output models/my_experiment \
    --epochs 50 --batch-size 512 --learning-rate 0.001 \
    --sample-size 50000 --filter-zeros --min-impression 5000 \
    --use-pca --pca-components 128 --use-fp16 \
    --early-stopping-patience 15 --save-best-model \
    --label-normalization standard --device auto
```

## Project File Recovery

If files are accidentally deleted:

1. **Restore xhs_ctr_production**: Run `./restore_xhs_production.sh`
2. **Check new architecture**: All core functionality exists in `src/` and related directories
3. **Use new architecture**: Recommend using `run_pipeline.py` for cleaner execution

## Important Files

### 🎯 Main Entry Points
- **scripts/train_model.py**: ✨ **Unified training entry point** (single & multi-task)
- **scripts/run_pipeline.py**: Feature extraction pipeline entry
- **scripts/train_mtl_model.py**: Dedicated multi-task training entry

### 📁 Core Architecture 
- **training/base/**: ✨ **Unified base class architecture** (eliminates 80%+ duplication)
- **training/single_task/ctr_trainer.py**: Lightweight single-task trainer (350 lines vs 1355 original)
- **training/multi_task/**: Lightweight MTL trainers (710 lines vs 3581 original)
- **pipelines/**: Four-layer pipeline base class system

### 📋 Documentation & Recovery
- **PROJECT_STRUCTURE.md**: Detailed architecture documentation
- **.env.example**: Environment variable template
- **restore_xhs_production.sh**: Recovery script for deleted files

### 🧹 Code Cleanup Completed (2025.10.25)
**Files Removed During Architecture Cleanup**:
- ❌ Legacy training scripts: `train_model.py`, `run_local_training_model*.py` (5 files)
- ❌ Unused models: `simplified_multi_task_config.py`, `task_specific_feature_masking.py`
- ❌ Backup files: `text_pipeline_backup.py`, `text_pipeline_optimized.py`
- ✅ **Result**: Cleaner codebase with 80%+ code duplication eliminated

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.