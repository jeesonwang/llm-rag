#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: conf.py
Time: 20240902
"""

import os

import pytz
from dotenv import load_dotenv

# [读取.env文件，转化为环境变量]
load_dotenv()

# [BASE]
TZ = pytz.timezone(os.getenv("TZ", "Asia/Shanghai"))
LOG_DIR = os.getenv("LOG_DIR", "./logs")
# [EMBEDDING MODEL]
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")

# [LLM SERVER]
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

