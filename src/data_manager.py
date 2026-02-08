#!/usr/bin/env python3
"""
数据管理模块
负责数据的保存和管理
"""

import os
import csv
import time
import src.config as config
from src.config import DATA_DIR, MAX_FILE_SIZE, logger

# 文件计数器
file_counter = 0
current_emg_file = None
current_imu_file = None


def init_data_files():
    """初始化数据文件
    
    Returns:
        bool: 是否成功初始化
    """
    global file_counter, current_emg_file, current_imu_file
    
    try:
        # 确保数据目录存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # 生成新的文件序号
        file_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 创建EMG数据文件
        emg_filename = os.path.join(DATA_DIR, f"emg_{timestamp}_{file_counter}.csv")
        current_emg_file = open(emg_filename, 'w', newline='', encoding='utf-8')
        emg_writer = csv.writer(current_emg_file)
        emg_writer.writerow(['timestamp'] + [f'channel_{i+1}' for i in range(8)])
        
        # 创建IMU数据文件
        imu_filename = os.path.join(DATA_DIR, f"imu_{timestamp}_{file_counter}.csv")
        current_imu_file = open(imu_filename, 'w', newline='', encoding='utf-8')
        imu_writer = csv.writer(current_imu_file)
        imu_writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        logger.info(f"初始化数据文件: {emg_filename}, {imu_filename}")
        return True
        
    except Exception as e:
        logger.error(f"初始化数据文件失败: {e}")
        # 确保文件被关闭
        if current_emg_file:
            try:
                current_emg_file.close()
            except:
                pass
        if current_imu_file:
            try:
                current_imu_file.close()
            except:
                pass
        return False


def save_data_point(data_type, timestamp, data):
    """保存数据点
    
    Args:
        data_type: 数据类型 ('EMG' 或 'IMU')
        timestamp: 时间戳
        data: 数据列表
    """
    if not config.SAVE_DATA:
        return
    
    try:
        if data_type == 'EMG':
            if current_emg_file:
                # 检查文件大小
                current_emg_file.flush()
                if os.path.getsize(current_emg_file.name) >= MAX_FILE_SIZE:
                    # 文件达到上限，创建新文件
                    init_data_files()
                
                writer = csv.writer(current_emg_file)
                writer.writerow([timestamp] + data)
        
        elif data_type == 'IMU':
            if current_imu_file:
                # 检查文件大小
                current_imu_file.flush()
                if os.path.getsize(current_imu_file.name) >= MAX_FILE_SIZE:
                    # 文件达到上限，创建新文件
                    init_data_files()
                
                writer = csv.writer(current_imu_file)
                writer.writerow([timestamp] + data)
        
    except Exception as e:
        logger.error(f"保存数据点失败: {e}")


def close_data_files():
    """关闭数据文件"""
    global current_emg_file, current_imu_file
    
    try:
        if current_emg_file:
            current_emg_file.close()
            current_emg_file = None
        if current_imu_file:
            current_imu_file.close()
            current_imu_file = None
        logger.info("关闭数据文件")
    except Exception as e:
        logger.error(f"关闭数据文件失败: {e}")
