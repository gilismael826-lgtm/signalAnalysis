#!/usr/bin/env python3
"""
信号预处理模块
负责对EMG和IMU信号进行预处理
"""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch
import src.config as config
from src.config import logger
from src import data_manager


class SignalProcessor:
    """信号预处理器"""
    
    def __init__(self, sample_rate=250.0, buffer_size=500):
        """初始化信号预处理器
        
        Args:
            sample_rate: 采样率（Hz）
            buffer_size: 缓冲区大小（样本数）
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.enabled = False
        self.filter_type = 'bandpass'
        self.normalize = False
        
        # 滤波器系数缓存
        self.bandpass_coeffs = None
        self.notch_coeffs = None
        
        # 实时数据缓冲区（用于滤波）
        self.emg_filter_buffers = [[] for _ in range(8)]
        self.imu_filter_buffers = [[] for _ in range(6)]
        
        # 预处理后的数据缓冲区
        self.processed_emg_buffer = []
        self.processed_imu_buffer = []
    
    def set_enabled(self, enabled):
        """启用或禁用预处理
        
        Args:
            enabled: 是否启用预处理
        """
        self.enabled = enabled
        logger.info(f"预处理已{'启用' if enabled else '禁用'}")
    
    def set_filter_type(self, filter_type):
        """设置滤波器类型
        
        Args:
            filter_type: 滤波器类型 ('bandpass', 'notch', 'both', 'none')
        """
        self.filter_type = filter_type
        self.bandpass_coeffs = None
        self.notch_coeffs = None
        logger.info(f"滤波器类型设置为: {filter_type}")
    
    def set_normalize(self, normalize):
        """设置是否归一化
        
        Args:
            normalize: 是否归一化
        """
        self.normalize = normalize
        logger.info(f"归一化已{'启用' if normalize else '禁用'}")
    
    def design_bandpass_filter(self, lowcut=20.0, highcut=100.0, order=4):
        """设计带通滤波器
        
        Args:
            lowcut: 低截止频率（Hz）
            highcut: 高截止频率（Hz）
            order: 滤波器阶数
        """
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.bandpass_coeffs = (b, a)
        logger.info(f"带通滤波器设计完成: {lowcut}-{highcut}Hz, 阶数={order}")
        return b, a
    
    def design_notch_filter(self, notch_freq=50.0, quality_factor=30.0):
        """设计陷波滤波器
        
        Args:
            notch_freq: 陷波频率（Hz）
            quality_factor: 品质因数
        """
        b, a = iirnotch(notch_freq, quality_factor, fs=self.sample_rate)
        self.notch_coeffs = (b, a)
        logger.info(f"陷波滤波器设计完成: {notch_freq}Hz, Q={quality_factor}")
        return b, a
    
    def apply_bandpass_filter(self, data):
        """应用带通滤波器
        
        Args:
            data: 输入数据
            
        Returns:
            滤波后的数据
        """
        if self.bandpass_coeffs is None:
            self.design_bandpass_filter()
        
        b, a = self.bandpass_coeffs
        y = lfilter(b, a, data)
        return y
    
    def apply_notch_filter(self, data):
        """应用陷波滤波器
        
        Args:
            data: 输入数据
            
        Returns:
            滤波后的数据
        """
        if self.notch_coeffs is None:
            self.design_notch_filter()
        
        b, a = self.notch_coeffs
        y = lfilter(b, a, data)
        return y
    
    def apply_normalization(self, data):
        """应用归一化
        
        Args:
            data: 输入数据
            
        Returns:
            归一化后的数据
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            return np.zeros_like(data)
        
        normalized = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized
    
    def detrend(self, data):
        """去趋势
        
        Args:
            data: 输入数据
            
        Returns:
            去趋势后的数据
        """
        detrended = data - np.polyval(np.polyfit(np.arange(len(data)), data, 1), np.arange(len(data)))
        return detrended
    
    def process_emg_data(self, emg_data):
        """处理EMG数据
        
        Args:
            emg_data: EMG数据列表（8个通道）
            
        Returns:
            处理后的EMG数据列表
        """
        if not self.enabled:
            return emg_data
        
        processed = []
        
        for channel_data in emg_data:
            data = np.array(channel_data)
            
            # 去趋势
            data = self.detrend(data)
            
            # 滤波
            if self.filter_type in ['bandpass', 'both']:
                data = self.apply_bandpass_filter(data)
            
            if self.filter_type in ['notch', 'both']:
                data = self.apply_notch_filter(data)
            
            # 归一化
            if self.normalize:
                data = self.apply_normalization(data)
            
            processed.append(data.tolist())
        
        return processed
    
    def process_imu_data(self, imu_data):
        """处理IMU数据
        
        Args:
            imu_data: IMU数据列表（6个通道）
            
        Returns:
            处理后的IMU数据列表
        """
        if not self.enabled:
            return imu_data
        
        processed = []
        
        for channel_data in imu_data:
            data = np.array(channel_data)
            
            # 去趋势
            data = self.detrend(data)
            
            # 归一化
            if self.normalize:
                data = self.apply_normalization(data)
            
            processed.append(data.tolist())
        
        return processed
    
    def update_buffers(self, emg_data, imu_data):
        """更新预处理数据缓冲区
        
        Args:
            emg_data: EMG数据（单个样本，8通道）
            imu_data: IMU数据（单个样本，6通道）
        """
        if not self.enabled:
            return
        
        # 使用相对时间戳（基于样本数）
        if not hasattr(self, 'sample_count'):
            self.sample_count = 0
        
        # 计算相对时间（秒）
        ts = self.sample_count / self.sample_rate
        self.sample_count += 1
        
        # 处理EMG数据
        processed_emg = []
        for i in range(8):
            if i < len(emg_data):
                self.emg_filter_buffers[i].append(emg_data[i])
                if len(self.emg_filter_buffers[i]) > self.buffer_size:
                    self.emg_filter_buffers[i].pop(0)
                
                # 当缓冲区足够大时，进行预处理
                if len(self.emg_filter_buffers[i]) >= 30:  # 至少需要30个样本进行滤波
                    data = np.array(self.emg_filter_buffers[i])
                    
                    # 去趋势
                    data = self.detrend(data)
                    
                    # 滤波
                    if self.filter_type in ['bandpass', 'both']:
                        data = self.apply_bandpass_filter(data)
                    
                    if self.filter_type in ['notch', 'both']:
                        data = self.apply_notch_filter(data)
                    
                    # 归一化
                    if self.normalize:
                        data = self.apply_normalization(data)
                    
                    processed_emg.append(data[-1])  # 取最新的预处理值
                else:
                    processed_emg.append(emg_data[i])  # 缓冲区不足时使用原始值
            else:
                processed_emg.append(0)
        
        # 处理IMU数据
        processed_imu = []
        for i in range(6):
            if i < len(imu_data):
                self.imu_filter_buffers[i].append(imu_data[i])
                if len(self.imu_filter_buffers[i]) > self.buffer_size:
                    self.imu_filter_buffers[i].pop(0)
                
                # 当缓冲区足够大时，进行预处理
                if len(self.imu_filter_buffers[i]) >= 30:  # 至少需要30个样本进行滤波
                    data = np.array(self.imu_filter_buffers[i])
                    
                    # 去趋势
                    data = self.detrend(data)
                    
                    # 归一化
                    if self.normalize:
                        data = self.apply_normalization(data)
                    
                    processed_imu.append(data[-1])  # 取最新的预处理值
                else:
                    processed_imu.append(imu_data[i])  # 缓冲区不足时使用原始值
            else:
                processed_imu.append(0)
        
        # 更新预处理后的数据缓冲区
        if len(processed_emg) > 0:
            self.processed_emg_buffer.append((ts, processed_emg))
            if len(self.processed_emg_buffer) > 10000:
                self.processed_emg_buffer.pop(0)
            
            # 保存预处理EMG数据
            data_manager.save_data_point('EMG', ts, processed_emg, is_processed=True)
        
        if len(processed_imu) > 0:
            self.processed_imu_buffer.append((ts, processed_imu))
            if len(self.processed_imu_buffer) > 10000:
                self.processed_imu_buffer.pop(0)
            
            # 保存预处理IMU数据
            data_manager.save_data_point('IMU', ts, processed_imu, is_processed=True)
    
    def _process_full_buffer(self):
        """处理完整的缓冲区数据（已废弃）"""
        pass


# 全局信号处理器实例
signal_processor = SignalProcessor(sample_rate=250.0)