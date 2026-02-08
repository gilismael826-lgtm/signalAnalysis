#!/usr/bin/env python3
"""
配置模块
集中管理所有配置参数和全局变量
"""

import os
import logging
from collections import deque

# 系统配置
SERIAL_PORT = "COM5"
BAUDRATE = 921600
SAVE_DATA = True
DATA_DIR = "data"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# 数据缓冲区
emg_buffer = deque(maxlen=10000)
imu_buffer = deque(maxlen=10000)

# 缩放参数
x_scale = 5000
emg_y_min = -50000
emg_y_max = 50000
imu_y_min = -5
imu_y_max = 5

# 运行状态
running = False
ser = None

# 缩放预设
zoom_presets = {
    "默认": (5000, -50000, 50000, -5, 5),
    "EMG精细": (2000, -500, 500, -5, 5),
    "EMG宽范围": (1000, -500000, 500000, -5, 5),
    "IMU精细": (5000, -50000, 50000, -1, 1),
    "IMU宽范围": (5000, -50000, 50000, -20, 20),
    "全屏": (10000, -50000, 50000, -5, 5)
}

# 日志配置
logging.basicConfig(
    filename=os.path.join('logs', 'system.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('system')
