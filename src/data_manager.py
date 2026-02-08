#!/usr/bin/env python3
"""
数据管理模块（批量保存优化版）
负责数据的保存和管理
支持原始数据和预处理数据的独立存储
使用批量保存机制减少I/O操作
"""

import os
import csv
import time
import src.config as config
from src.config import DATA_DIR, MAX_FILE_SIZE, logger

# 文件计数器
file_counter = 0

# 原始数据文件
current_raw_emg_file = None
current_raw_imu_file = None

# 预处理数据文件
current_processed_emg_file = None
current_processed_imu_file = None

# 批量保存缓冲区
raw_emg_batch = []
raw_imu_batch = []
processed_emg_batch = []
processed_imu_batch = []

# 批量保存配置
BATCH_SIZE = 50  # 每50个数据点批量保存一次（降低延迟）
FLUSH_INTERVAL = 0.5  # 最多0.5秒刷新一次（更频繁刷新）
last_flush_time = time.time()


def init_data_files():
    """初始化数据文件
    
    Returns:
        bool: 是否成功初始化
    """
    global file_counter
    global current_raw_emg_file, current_raw_imu_file
    global current_processed_emg_file, current_processed_imu_file
    
    try:
        # 确保数据目录存在
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # 生成新的文件序号
        file_counter += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 创建原始EMG数据文件
        raw_emg_filename = os.path.join(DATA_DIR, f"raw_emg_{timestamp}_{file_counter}.csv")
        current_raw_emg_file = open(raw_emg_filename, 'w', newline='', encoding='utf-8')
        emg_writer = csv.writer(current_raw_emg_file)
        emg_writer.writerow(['timestamp'] + [f'channel_{i+1}' for i in range(8)])
        
        # 创建原始IMU数据文件
        raw_imu_filename = os.path.join(DATA_DIR, f"raw_imu_{timestamp}_{file_counter}.csv")
        current_raw_imu_file = open(raw_imu_filename, 'w', newline='', encoding='utf-8')
        imu_writer = csv.writer(current_raw_imu_file)
        imu_writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        # 创建预处理EMG数据文件
        norm_status = 'norm' if config.PREPROCESSING_NORMALIZE else 'nonorm'
        processed_emg_filename = os.path.join(DATA_DIR, f"processed_emg_{norm_status}_{timestamp}_{file_counter}.csv")
        current_processed_emg_file = open(processed_emg_filename, 'w', newline='', encoding='utf-8')
        emg_writer = csv.writer(current_processed_emg_file)
        emg_writer.writerow(['timestamp'] + [f'channel_{i+1}' for i in range(8)])
        
        # 创建预处理IMU数据文件
        processed_imu_filename = os.path.join(DATA_DIR, f"processed_imu_{norm_status}_{timestamp}_{file_counter}.csv")
        current_processed_imu_file = open(processed_imu_filename, 'w', newline='', encoding='utf-8')
        imu_writer = csv.writer(current_processed_imu_file)
        imu_writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        logger.info(f"初始化数据文件:")
        logger.info(f"  原始EMG: {raw_emg_filename}")
        logger.info(f"  原始IMU: {raw_imu_filename}")
        logger.info(f"  预处理EMG: {processed_emg_filename}")
        logger.info(f"  预处理IMU: {processed_imu_filename}")
        return True
        
    except Exception as e:
        logger.error(f"初始化数据文件失败: {e}")
        # 确保文件被关闭
        close_data_files()
        return False


def save_data_point(data_type, timestamp, data, is_processed=False):
    """保存数据点到批量缓冲区
    
    Args:
        data_type: 数据类型 ('EMG' 或 'IMU')
        timestamp: 时间戳
        data: 数据列表
        is_processed: 是否为预处理数据
    """
    if not config.SAVE_DATA:
        return
    
    global last_flush_time
    
    try:
        # 将数据添加到批量缓冲区
        if data_type == 'EMG':
            if is_processed:
                processed_emg_batch.append((timestamp, data))
            else:
                raw_emg_batch.append((timestamp, data))
        elif data_type == 'IMU':
            if is_processed:
                processed_imu_batch.append((timestamp, data))
            else:
                raw_imu_batch.append((timestamp, data))
        
        # 检查是否需要刷新
        current_time = time.time()
        need_flush = False
        
        # 检查批量大小
        if is_processed:
            if len(processed_emg_batch) >= BATCH_SIZE or len(processed_imu_batch) >= BATCH_SIZE:
                need_flush = True
        else:
            if len(raw_emg_batch) >= BATCH_SIZE or len(raw_imu_batch) >= BATCH_SIZE:
                need_flush = True
        
        # 检查时间间隔
        if current_time - last_flush_time >= FLUSH_INTERVAL:
            need_flush = True
        
        # 刷新数据
        if need_flush:
            flush_batches()
            last_flush_time = current_time
        
    except Exception as e:
        logger.error(f"保存数据点失败: {e}")


def flush_batches():
    """将批量缓冲区的数据写入文件"""
    global raw_emg_batch, raw_imu_batch, processed_emg_batch, processed_imu_batch
    
    try:
        # 保存原始EMG数据
        if len(raw_emg_batch) > 0 and current_raw_emg_file:
            writer = csv.writer(current_raw_emg_file)
            for timestamp, data in raw_emg_batch:
                writer.writerow([timestamp] + data)
            raw_emg_batch.clear()
        
        # 保存原始IMU数据
        if len(raw_imu_batch) > 0 and current_raw_imu_file:
            writer = csv.writer(current_raw_imu_file)
            for timestamp, data in raw_imu_batch:
                writer.writerow([timestamp] + data)
            raw_imu_batch.clear()
        
        # 保存预处理EMG数据
        if len(processed_emg_batch) > 0 and current_processed_emg_file:
            writer = csv.writer(current_processed_emg_file)
            for timestamp, data in processed_emg_batch:
                writer.writerow([timestamp] + data)
            processed_emg_batch.clear()
        
        # 保存预处理IMU数据
        if len(processed_imu_batch) > 0 and current_processed_imu_file:
            writer = csv.writer(current_processed_imu_file)
            for timestamp, data in processed_imu_batch:
                writer.writerow([timestamp] + data)
            processed_imu_batch.clear()
        
        # 检查文件大小并轮换
        check_file_sizes()
        
    except Exception as e:
        logger.error(f"刷新批量数据失败: {e}")


def check_file_sizes():
    """检查文件大小，必要时轮换文件"""
    files_to_check = [
        (current_raw_emg_file, 'raw_emg'),
        (current_raw_imu_file, 'raw_imu'),
        (current_processed_emg_file, 'processed_emg'),
        (current_processed_imu_file, 'processed_imu')
    ]
    
    for file_handle, file_type in files_to_check:
        if file_handle:
            file_handle.flush()
            if os.path.getsize(file_handle.name) >= MAX_FILE_SIZE:
                logger.info(f"文件 {file_type} 达到大小上限，创建新文件")
                init_data_files()


def force_flush():
    """强制刷新所有批量缓冲区"""
    global last_flush_time
    flush_batches()
    last_flush_time = time.time()


def close_data_files():
    """关闭数据文件"""
    global current_raw_emg_file, current_raw_imu_file
    global current_processed_emg_file, current_processed_imu_file
    
    try:
        # 关闭前刷新所有数据
        force_flush()
        
        if current_raw_emg_file:
            current_raw_emg_file.close()
            current_raw_emg_file = None
        if current_raw_imu_file:
            current_raw_imu_file.close()
            current_raw_imu_file = None
        if current_processed_emg_file:
            current_processed_emg_file.close()
            current_processed_emg_file = None
        if current_processed_imu_file:
            current_processed_imu_file.close()
            current_processed_imu_file = None
        logger.info("关闭数据文件")
    except Exception as e:
        logger.error(f"关闭数据文件失败: {e}")


def get_data_file_info():
    """获取当前数据文件信息
    
    Returns:
        dict: 包含当前数据文件路径的字典
    """
    file_info = {
        'raw_emg': current_raw_emg_file.name if current_raw_emg_file else None,
        'raw_imu': current_raw_imu_file.name if current_raw_imu_file else None,
        'processed_emg': current_processed_emg_file.name if current_processed_emg_file else None,
        'processed_imu': current_processed_imu_file.name if current_processed_imu_file else None,
    }
    return file_info