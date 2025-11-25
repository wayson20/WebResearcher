# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Config for WebResearcher
Centralized configuration management for all environment variables.
"""
import os
from pathlib import Path

# ==================== API Keys (Sensitive) ====================
# LLM api key
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
# google search api key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
# url page reader api key
JINA_API_KEY = os.getenv("JINA_API_KEY")

# ==================== Sandbox Configuration ====================
# code execution sandbox endpoints
SANDBOX_FUSION_ENDPOINTS = [
    endpoint.strip() 
    for endpoint in os.getenv('SANDBOX_FUSION_ENDPOINTS', '').split(',') 
    if endpoint.strip()
]

# ==================== Agent Configuration ====================
MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))
AGENT_TIMEOUT = int(os.getenv('AGENT_TIMEOUT', 1800))
FILE_DIR = os.getenv('FILE_DIR', './files')

# ==================== Visit Tool Configuration ====================
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
VISIT_SERVER_MAX_RETRIES = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "gpt-4o-mini")

# ==================== Video Analysis Configuration ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE", "")
VIDEO_MODEL_NAME = os.getenv('VIDEO_MODEL_NAME', 'qwen-omni-turbo')
VIDEO_ANALYSIS_MODEL_NAME = os.getenv('VIDEO_ANALYSIS_MODEL_NAME', 'qwen-plus-latest')
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', os.getcwd()))

# ==================== Logging Configuration ====================
LOG_LEVEL = os.getenv("WEBRESEARCHER_LOG_LEVEL", "INFO").upper()

# ==================== Constants ====================
OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'
