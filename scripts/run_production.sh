#!/bin/bash
#
# 生产环境运行入口
#

set -e

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Production Mode (EMR) ===${NC}"

# 检查环境变量
if [ -z "$EMR_MASTER_IP" ]; then
    echo -e "${RED}Error: EMR_MASTER_IP not set${NC}"
    echo "Please run: export EMR_MASTER_IP=your-master-ip"
    exit 1
fi

# 加载环境变量
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

export ENVIRONMENT="production"

# 检查OSS凭证
if [ -z "$OSS_ACCESS_KEY_ID" ] || [ -z "$OSS_ACCESS_KEY_SECRET" ]; then
    echo -e "${YELLOW}Warning: OSS credentials not set${NC}"
    echo "Please check your .env file"
fi

# 选择操作
echo "Select operation:"
echo "1) Deploy code to EMR"
echo "2) Run full pipeline (parallel)"
echo "3) Run single pipeline"
echo "4) Check job status"
echo "5) View logs"

read -p "Choice [1-5]: " choice

case $choice in
    1)
        echo -e "${GREEN}Deploying code to EMR...${NC}"
        ./emr/deploy/upload_code.sh
        ;;
    2)
        echo -e "${GREEN}Running full pipeline...${NC}"
        python emr/submit/orchestrator.py --master-ip $EMR_MASTER_IP
        ;;
    3)
        echo -e "${GREEN}Select pipeline:${NC}"
        echo "1) Text pipeline"
        echo "2) Image pipeline"
        echo "3) Merge pipeline"
        read -p "Choice: " pipeline
        
        case $pipeline in
            1) ./emr/submit/submit_tools.sh text ;;
            2) ./emr/submit/submit_tools.sh image ;;
            3) ./emr/submit/submit_tools.sh merge ;;
            *) echo "Invalid choice" ;;
        esac
        ;;
    4)
        ./emr/submit/submit_tools.sh status
        ;;
    5)
        ./emr/submit/submit_tools.sh logs
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"