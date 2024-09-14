#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wjs
File: log.py
Time: 20240913
"""

import os
import re
import sys
import datetime
import logging

from loguru import logger

from .conf import LOG_DIR


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def clear_timeout_logs(log_dir: str, keep_day: int = 15):
    """
    删除最大超过keep_day的日志
    由于loguru本身不具备在删除既往日志的功能(只支持同一进程的日志旋转),添加本方法进行处理
    强依赖数据文件格式包含 yyyy-mm-dd格式
    :param log_dir: 日志路径
    :param keep_day: 最多保留日
    :return:
    """
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    for filename in os.listdir(log_dir):
        search = pattern.search(filename)
        today = datetime.date.today()
        if search:
            filepath = os.path.join(log_dir, filename)
            log_day = datetime.datetime.strptime(search.group(), "%Y-%m-%d").date()
            if (today - log_day).days > keep_day:
                try:
                    os.remove(filepath)
                except Exception as err:
                    logging.info(f"删除超时日志失败: {filepath}: {err}")
                else:
                    logging.info(f"删除超时日志成功: {filepath}")


def init():
    clear_timeout_logs(LOG_DIR, keep_day=15)


LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))

intercept_handler = InterceptHandler()
logging.root.setLevel(LOG_LEVEL)

seen = set()
for name in [*logging.root.manager.loggerDict.keys()]:
    if name not in seen:
        seen.add(name.split(".")[0])
        logging.getLogger(name).handlers = [intercept_handler]

logger.configure(handlers=[{"sink": sys.stderr, "level": LOG_LEVEL}])

# [定义日志路径]
os.makedirs(LOG_DIR, exist_ok=True)

logger.add(
    os.path.join(LOG_DIR, "info_{time:%Y-%m-%d}.log"),
    level="INFO",
    colorize=False,
    rotation="1 days",
    retention="7 days",
    backtrace=False,
    diagnose=False,
    encoding="utf-8",
    format="{time} {level} {message} | PID:{process} | TID: {thread}",
    catch=False,
)
logger.add(
    os.path.join(LOG_DIR, "error_{time:%Y-%m-%d}.log"),
    level="ERROR",
    colorize=False,
    rotation="1 days",
    retention="15 days",
    backtrace=False,
    diagnose=False,
    encoding="utf-8",
    format="{time} {level} {message} | PID:{process} | TID: {thread}",
    catch=False,
)