# -*- coding: utf-8 -*-
"""
Settings for WebResearcher Web UI
"""
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用基础配置
    app_name: str = "WebResearcher Web UI"
    app_version: str = "0.2.0"
    debug: bool = False
    
    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 数据目录
    data_dir: Optional[Path] = None
    
    # CORS 配置
    allow_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
