"""
OSS Helper - 封装Spark对OSS的读写操作
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

try:
    import oss2
except ImportError:
    print("Warning: oss2 not installed. Installing...")
    os.system("pip install oss2")
    import oss2

from pyspark.sql import SparkSession, DataFrame
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)


class OSSHelper:
    """简化的OSS助手类 - 专注于验证和配置"""
    
    def __init__(self):
        """初始化OSS客户端"""
        # 获取OSS凭证
        self.access_key_id = os.getenv("OSS_ACCESS_KEY_ID") or os.getenv("XHS_CTR_OSS_ACCESS_KEY_ID", "")
        self.access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET") or os.getenv("XHS_CTR_OSS_ACCESS_KEY_SECRET", "")
        
        # 获取endpoint
        endpoint_raw = os.getenv("OSS_ENDPOINT") or os.getenv("XHS_CTR_OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com")
        # oss2需要完整URL (带https://)
        self.endpoint = endpoint_raw if endpoint_raw.startswith("http") else f"https://{endpoint_raw}"
        self.bucket_name = os.getenv("OSS_BUCKET_NAME") or os.getenv("XHS_CTR_OSS_BUCKET_NAME", "xhs-ctr-model")
        
        # 验证配置
        if not self.access_key_id or not self.access_key_secret:
            raise ValueError("OSS credentials not configured!")
        
        # 创建OSS客户端（仅用于验证和文件列表）
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
        
        logger.info(f"OSS Helper initialized: bucket={self.bucket_name}, endpoint={self.endpoint}")
    
    def list_files(self, prefix: str = "") -> List[str]:
        """列出指定前缀的所有文件"""
        files = []
        for obj in oss2.ObjectIterator(self.bucket, prefix=prefix):
            if not obj.key.endswith('/'):  # 排除目录
                files.append(obj.key)
        return files
    
    def file_exists(self, key: str) -> bool:
        """检查文件是否存在"""
        try:
            self.bucket.head_object(key)
            return True
        except oss2.exceptions.NoSuchKey:
            return False
    
    def normalize_path(self, path: str) -> str:
        """
        统一处理各种路径格式，返回标准的oss://格式
        避免魔法数字和重复的路径解析代码
        """
        # 如果已经是完整的oss://路径
        if path.startswith("oss://"):
            return path
        
        # 如果是以bucket名开头的路径
        if path.startswith(self.bucket_name):
            path = path[len(self.bucket_name):].lstrip('/')
        
        # 去掉开头的斜杠
        path = path.lstrip('/')
        
        # 构建完整的OSS路径
        return f"oss://{self.bucket_name}/{path}"
    
    def parse_oss_path(self, path: str) -> tuple:
        """
        解析OSS路径，返回(bucket_name, key)
        用于text_pipeline等需要提取key的场景
        """
        if path.startswith("oss://"):
            # oss://bucket/key/path
            path_without_prefix = path[6:]  # 去掉"oss://"
            parts = path_without_prefix.split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            return bucket, key
        else:
            # 假定是key
            return self.bucket_name, path
    
    def read_parquet(self, spark: SparkSession, path: str) -> DataFrame:
        """
        使用Spark直接读取OSS上的parquet文件
        封装读取逻辑，统一处理路径
        """
        oss_path = self.normalize_path(path)
        logger.info(f"Reading parquet from: {oss_path}")
        
        try:
            df = spark.read.parquet(oss_path)
            # 不使用cache()和count()，让Spark延迟执行以提高性能
            # 对于TB级数据，cache会占用大量内存，count会触发不必要的全表扫描
            logger.info(f"Successfully loaded parquet from {oss_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {oss_path}: {e}")
            raise
    
    def write_parquet(self, df: DataFrame, path: str, 
                     num_partitions: Optional[int] = None,
                     mode: str = "overwrite"):
        """
        使用Spark直接写入parquet文件到OSS
        不需要临时文件，直接写入
        """
        oss_path = self.normalize_path(path)
        logger.info(f"Writing parquet to: {oss_path}")
        
        # 如果指定了分区数，重新分区
        if num_partitions:
            df = df.repartition(num_partitions)
            logger.info(f"Repartitioned to {num_partitions} partitions")
        
        try:
            df.write \
              .mode(mode) \
              .option("compression", "snappy") \
              .parquet(oss_path)
            
            logger.info(f"Successfully wrote data to {oss_path}")
        except Exception as e:
            logger.error(f"Failed to write to {oss_path}: {e}")
            raise
    
    def check_parquet_files(self, prefix: str) -> List[str]:
        """检查指定前缀下的parquet文件"""
        if not prefix.endswith('/'):
            prefix += '/'
        
        parquet_files = [
            f for f in self.list_files(prefix) 
            if f.endswith('.parquet') and not f.split('/')[-1].startswith('.')
        ]
        
        logger.info(f"Found {len(parquet_files)} parquet files in {prefix}")
        return parquet_files
    
    def validate_connection(self) -> bool:
        """验证OSS连接"""
        try:
            # 尝试列出根目录
            list(oss2.ObjectIterator(self.bucket, max_keys=1))
            logger.info("OSS connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"OSS connection validation failed: {e}")
            return False




if __name__ == "__main__":
    # 测试OSS连接
    helper = OSSHelper()
    
    print("Testing OSS connection...")
    if helper.validate_connection():
        print("✅ OSS connection successful!")
        
        # 列出根目录文件
        print("\nListing files in root directory:")
        files = helper.list_files()[:10]  # 只显示前10个
        for f in files:
            print(f"  - {f}")
        
        # 检查note_info_parquet目录
        print("\nChecking note_info_parquet directory:")
        note_files = helper.list_files("note_info_parquet/")[:5]
        if note_files:
            for f in note_files:
                print(f"  - {f}")
        else:
            print("  No files found in note_info_parquet/")
    else:
        print("❌ OSS connection failed!")