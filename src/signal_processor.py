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
    
    def __init__(self, emg_sample_rate=250.0, imu_sample_rate=104.0, buffer_size=500):
        """初始化信号预处理器
        
        Args:
            emg_sample_rate: EMG采样率（Hz）
            imu_sample_rate: IMU采样率（Hz）
            buffer_size: 缓冲区大小（样本数）
        """
        self.emg_sample_rate = emg_sample_rate
        self.imu_sample_rate = imu_sample_rate
        self.buffer_size = buffer_size
        self.enabled = False
        self.filter_type = 'bandpass'
        self.normalize = False
        
        # EMG滤波器系数缓存
        self.bandpass_coeffs = None
        self.notch_coeffs = None
        
        # IMU滤波器系数缓存
        self.lowpass_coeffs = None
        
        # 实时数据缓冲区（用于滤波）
        self.emg_filter_buffers = [[] for _ in range(8)]
        self.imu_filter_buffers = [[] for _ in range(6)]
        
        # 预处理后的数据缓冲区
        self.processed_emg_buffer = []
        self.processed_imu_buffer = []
        
        # 样本计数器（分别用于EMG和IMU）
        self.emg_sample_count = 0
        self.imu_sample_count = 0
    
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
        """设计带通滤波器（用于EMG信号）
        
        Args:
            lowcut: 低截止频率（Hz）
            highcut: 高截止频率（Hz）
            order: 滤波器阶数
        """
        nyq = 0.5 * self.emg_sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.bandpass_coeffs = (b, a)
        logger.info(f"带通滤波器设计完成: {lowcut}-{highcut}Hz, 阶数={order}, 采样率={self.emg_sample_rate}Hz")
        return b, a
    
    def design_notch_filter(self, notch_freq=50.0, quality_factor=30.0):
        """设计陷波滤波器（用于EMG信号）
        
        Args:
            notch_freq: 陷波频率（Hz）
            quality_factor: 品质因数
        """
        b, a = iirnotch(notch_freq, quality_factor, fs=self.emg_sample_rate)
        self.notch_coeffs = (b, a)
        logger.info(f"陷波滤波器设计完成: {notch_freq}Hz, Q={quality_factor}, 采样率={self.emg_sample_rate}Hz")
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
    
    def design_lowpass_filter(self, cutoff=10.0, order=4):
        """设计低通滤波器（用于IMU信号）
        
        Args:
            cutoff: 截止频率（Hz）
            order: 滤波器阶数
        """
        nyq = 0.5 * self.imu_sample_rate
        high = cutoff / nyq
        b, a = butter(order, high, btype='low')
        self.lowpass_coeffs = (b, a)
        logger.info(f"低通滤波器设计完成: {cutoff}Hz, 阶数={order}, 采样率={self.imu_sample_rate}Hz")
        return b, a
    
    def apply_lowpass_filter(self, data):
        """应用低通滤波器（用于IMU信号）
        
        Args:
            data: 输入数据
            
        Returns:
            滤波后的数据
        """
        if self.lowpass_coeffs is None:
            self.design_lowpass_filter()
        
        b, a = self.lowpass_coeffs
        y = lfilter(b, a, data)
        return y
    
    def apply_moving_average(self, data, window_size=5):
        """应用移动平均滤波（用于IMU信号）
        
        Args:
            data: 输入数据
            window_size: 窗口大小
            
        Returns:
            滤波后的数据
        """
        if len(data) < window_size:
            return data
        
        padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
        return smoothed
    
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
        """处理IMU数据（用于手势识别）
        
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
            
            # 低通滤波（去除高频噪声，适合手势识别）
            if config.IMU_LOWPASS_ENABLED:
                if self.lowpass_coeffs is None:
                    self.design_lowpass_filter(
                        cutoff=config.IMU_LOWPASS_CUTOFF,
                        order=config.IMU_LOWPASS_ORDER
                    )
                data = self.apply_lowpass_filter(data)
            
            # 移动平均滤波（平滑数据）
            if config.IMU_MOVING_AVG_ENABLED:
                data = self.apply_moving_average(data, window_size=config.IMU_MOVING_AVG_WINDOW)
            
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
        
        # 处理EMG数据
        if len(emg_data) > 0:
            emg_ts = self.emg_sample_count / self.emg_sample_rate
            self.emg_sample_count += 1
            
            processed_emg = []
            for i in range(8):
                if i < len(emg_data):
                    self.emg_filter_buffers[i].append(emg_data[i])
                    if len(self.emg_filter_buffers[i]) > self.buffer_size:
                        self.emg_filter_buffers[i].pop(0)
                    
                    if len(self.emg_filter_buffers[i]) >= 30:
                        data = np.array(self.emg_filter_buffers[i])
                        data = self.detrend(data)
                        
                        if self.filter_type in ['bandpass', 'both']:
                            data = self.apply_bandpass_filter(data)
                        
                        if self.filter_type in ['notch', 'both']:
                            data = self.apply_notch_filter(data)
                        
                        if self.normalize:
                            data = self.apply_normalization(data)
                        
                        processed_emg.append(data[-1])
                    else:
                        processed_emg.append(emg_data[i])
                else:
                    processed_emg.append(0)
            
            if len(processed_emg) > 0:
                self.processed_emg_buffer.append((emg_ts, processed_emg))
                if len(self.processed_emg_buffer) > 10000:
                    self.processed_emg_buffer.pop(0)
                data_manager.save_data_point('EMG', emg_ts, processed_emg, is_processed=True)
        
        # 处理IMU数据
        if len(imu_data) > 0:
            imu_ts = self.imu_sample_count / self.imu_sample_rate
            self.imu_sample_count += 1
            
            processed_imu = []
            for i in range(6):
                if i < len(imu_data):
                    self.imu_filter_buffers[i].append(imu_data[i])
                    if len(self.imu_filter_buffers[i]) > self.buffer_size:
                        self.imu_filter_buffers[i].pop(0)
                    
                    if len(self.imu_filter_buffers[i]) >= 30:
                        data = np.array(self.imu_filter_buffers[i])
                        data = self.detrend(data)
                        
                        if config.IMU_LOWPASS_ENABLED:
                            if self.lowpass_coeffs is None:
                                self.design_lowpass_filter(
                                    cutoff=config.IMU_LOWPASS_CUTOFF,
                                    order=config.IMU_LOWPASS_ORDER
                                )
                            data = self.apply_lowpass_filter(data)
                        
                        if config.IMU_MOVING_AVG_ENABLED:
                            data = self.apply_moving_average(data, window_size=config.IMU_MOVING_AVG_WINDOW)
                        
                        if self.normalize:
                            data = self.apply_normalization(data)
                        
                        processed_imu.append(data[-1])
                    else:
                        processed_imu.append(imu_data[i])
                else:
                    processed_imu.append(0)
            
            if len(processed_imu) > 0:
                self.processed_imu_buffer.append((imu_ts, processed_imu))
                if len(self.processed_imu_buffer) > 10000:
                    self.processed_imu_buffer.pop(0)
                data_manager.save_data_point('IMU', imu_ts, processed_imu, is_processed=True)
    
    def _process_full_buffer(self):
        """处理完整的缓冲区数据（已废弃）"""
        pass


# 全局信号处理器实例
signal_processor = SignalProcessor(
    emg_sample_rate=config.EMG_SAMPLE_RATE,
    imu_sample_rate=config.IMU_SAMPLE_RATE
)