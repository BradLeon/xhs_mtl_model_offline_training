#!/bin/bash
#
# 本地开发环境运行入口
#

set -e

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 设置Python路径
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 设置环境
export ENVIRONMENT="development"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Local Development Mode ===${NC}"

# 检查环境
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found, copying from .env.example${NC}"
    cp .env.example .env
fi

# 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 选择要运行的pipeline
echo "Select pipeline to run:"
echo "1) Text Feature Pipeline"
echo "2) Image Feature Pipeline"
echo "3) Feature Merge"
echo "4) Full Pipeline"
echo "5) Model Training"

read -p "Choice [1-5]: " choice

case $choice in
    1)
        echo -e "${GREEN}Running Text Feature Pipeline...${NC}"
        python local/scripts/run_text_pipeline.py
        ;;
    2)
        echo -e "${GREEN}Running Image Feature Pipeline...${NC}"
        python local/scripts/run_image_pipeline.py
        ;;
    3)
        echo -e "${GREEN}Running Feature Merge...${NC}"
        python local/scripts/run_feature_merge.py
        ;;
    4)
        echo -e "${GREEN}Running Full Pipeline...${NC}"
        python local/scripts/run_text_pipeline.py && \
        python local/scripts/run_image_pipeline.py && \
        python local/scripts/run_feature_merge.py
        ;;
    5)
        echo -e "${GREEN}Running Model Training...${NC}"
        python local/scripts/run_model_training.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"