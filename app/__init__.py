import yaml
from loguru import logger
from .api import app
from .core.pool import Pool
from .module.file_downloader import FileDownloader

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

# 初始化 各个模块
pool = Pool(**config.get("pool", {}))
logger.info(f"init pool: {config.get('pool', {})}")
