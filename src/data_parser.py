#!/usr/bin/env python3
"""
数据解析模块
负责解析接收到的数据包
"""

import time
import src.config as config
from src.config import logger


def parse_packet(data):
    """解析数据包
    
    Args:
        data: 接收到的数据包（bytes）
    
    Returns:
        tuple: (数据类型, 时间戳, 数据) 或 (None, None, None)
    """
    if len(data) < 29:
        logger.warning(f"数据包长度不足 29 字节: {len(data)}")
        return None, None, None
    
    if data[0:3] != b'\xd2\xd2\xd2':
        logger.warning(f"数据包起始标志错误: {data[0:3]}")
        return None, None, None
    
    pkt_type = data[3]
    ts = time.time()
    
    if pkt_type == 0xAA:
        emg = []
        for i in range(8):
            b0, b1, b2 = data[5 + i*3 : 8 + i*3]
            val = (b0 << 16) | (b1 << 8) | b2
            if val >= 0x800000:
                val -= 0x1000000
            emg.append(val)
        logger.info(f"解析到EMG数据包，通道数: {len(emg)}, 数据范围: {min(emg)}~{max(emg)}")
        return 'EMG', ts, emg
    
    elif pkt_type == 0xBB:
        raw = []
        for i in range(6):
            # 从字节 7 开始，每个通道占用 2 字节
            lo = data[7 + i*2]
            hi = data[8 + i*2]
            val = (hi << 8) | lo
            if val >= 0x8000:
                val -= 0x10000
            raw.append(val)
        
        logger.info(f"解析到IMU原始数据: {raw}")
        
        # 应用单位转换系数
        gyro = [0.0012 * x for x in raw[:3]]  # 陀螺仪: rad/s
        accel = [0.0005978 * x for x in raw[3:]]  # 加速度计: m/s²
        imu = gyro + accel
        
        logger.info(f"转换后IMU数据: 陀螺仪={imu[:3]}, 加速度={imu[3:]}")
        return 'IMU', ts, imu
    
    logger.warning(f"未知数据包类型: {pkt_type}")
    return None, None, None
