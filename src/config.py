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

# 预处理信号的独立缩放参数
processed_emg_y_min = -1.5
processed_emg_y_max = 1.5
processed_imu_y_min = -1.5
processed_imu_y_max = 1.5

# 运行状态
running = False
ser = None

# 信号预处理配置
PREPROCESSING_ENABLED = False
PREPROCESSING_FILTER_TYPE = 'bandpass'
PREPROCESSING_NORMALIZE = False
PREPROCESSING_DISPLAY_MODE = 'raw'  # 'raw' 或 'processed'

# IMU预处理配置（用于手势识别）
IMU_LOWPASS_ENABLED = True  # 启用低通滤波
IMU_LOWPASS_CUTOFF = 10.0  # 低通滤波截止频率（Hz）
IMU_LOWPASS_ORDER = 4  # 低通滤波阶数
IMU_MOVING_AVG_ENABLED = True  # 启用移动平均滤波
IMU_MOVING_AVG_WINDOW = 5  # 移动平均窗口大小

# 预处理后的数据缓冲区（由signal_processor管理）
processed_emg_buffer = []
processed_imu_buffer = []

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
