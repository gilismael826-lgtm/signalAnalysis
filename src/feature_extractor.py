#!/usr/bin/env python3
"""
特征提取模块
负责从EMG和IMU信号中提取时域、频域和时频域特征
支持不同采样率的EMG和IMU信号独立处理
"""

import numpy as np
import os
import sys
from scipy import signal
from scipy.stats import entropy

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("pywt库未安装，时频域特征将不可用。请运行: pip install PyWavelets")


class FeatureExtractor:
    """特征提取器
    
    支持EMG和IMU信号的特征提取，处理不同采样率问题
    """
    
    def __init__(self, emg_sample_rate=None, imu_sample_rate=None):
        """初始化特征提取器
        
        Args:
            emg_sample_rate: EMG采样率（Hz），默认使用config中的配置
            imu_sample_rate: IMU采样率（Hz），默认使用config中的配置
        """
        self.emg_sample_rate = emg_sample_rate or config.EMG_SAMPLE_RATE
        self.imu_sample_rate = imu_sample_rate or config.IMU_SAMPLE_RATE
        
        logger.info(f"特征提取器初始化: EMG采样率={self.emg_sample_rate}Hz, IMU采样率={self.imu_sample_rate}Hz")
    
    # ==================== 时域特征 ====================
    
    def extract_mean(self, data):
        """均值
        
        Args:
            data: 输入信号数据
            
        Returns:
            均值
        """
        return float(np.mean(data))
    
    def extract_variance(self, data):
        """方差
        
        Args:
            data: 输入信号数据
            
        Returns:
            方差
        """
        return float(np.var(data))
    
    def extract_std(self, data):
        """标准差
        
        Args:
            data: 输入信号数据
            
        Returns:
            标准差
        """
        return float(np.std(data))
    
    def extract_rms(self, data):
        """均方根（Root Mean Square）
        
        Args:
            data: 输入信号数据
            
        Returns:
            RMS值
        """
        return float(np.sqrt(np.mean(np.square(data))))
    
    def extract_iemg(self, data):
        """积分肌电（Integrated EMG）
        
        Args:
            data: 输入信号数据
            
        Returns:
            IEMG值
        """
        return float(np.sum(np.abs(data)))
    
    def extract_mav(self, data):
        """平均绝对值（Mean Absolute Value）
        
        Args:
            data: 输入信号数据
            
        Returns:
            MAV值
        """
        return float(np.mean(np.abs(data)))
    
    def extract_peak_to_peak(self, data):
        """峰峰值
        
        Args:
            data: 输入信号数据
            
        Returns:
            峰峰值
        """
        return float(np.max(data) - np.min(data))
    
    def extract_zc(self, data, threshold=0):
        """过零率（Zero Crossing Rate）
        
        统计信号穿过阈值的次数
        
        Args:
            data: 输入信号数据
            threshold: 阈值，默认为0
            
        Returns:
            过零率
        """
        zc = 0
        for i in range(len(data) - 1):
            if (data[i] - threshold) * (data[i + 1] - threshold) < 0:
                zc += 1
        return int(zc)
    
    def extract_ssc(self, data, threshold=0):
        """斜率符号变化率（Slope Sign Changes）
        
        统计斜率符号变化的次数
        
        Args:
            data: 输入信号数据
            threshold: 阈值，默认为0
            
        Returns:
            SSC值
        """
        ssc = 0
        for i in range(1, len(data) - 1):
            diff1 = data[i] - data[i - 1]
            diff2 = data[i + 1] - data[i]
            if (diff1 * diff2) < 0 and (abs(diff1) > threshold or abs(diff2) > threshold):
                ssc += 1
        return int(ssc)
    
    def extract_wl(self, data):
        """波形长度（Waveform Length）
        
        信号波形的累计长度
        
        Args:
            data: 输入信号数据
            
        Returns:
            WL值
        """
        return float(np.sum(np.abs(np.diff(data))))
    
    def extract_wamp(self, data, threshold=0):
        """Willison幅度（Willison Amplitude）
        
        统计相邻样本差值超过阈值的次数
        
        Args:
            data: 输入信号数据
            threshold: 阈值
            
        Returns:
            WAMP值
        """
        wamp = 0
        for i in range(len(data) - 1):
            if abs(data[i] - data[i + 1]) > threshold:
                wamp += 1
        return int(wamp)
    
    def extract_kurtosis(self, data):
        """峰度（Kurtosis）
        
        描述信号分布的尖锐程度
        
        Args:
            data: 输入信号数据
            
        Returns:
            峰度值
        """
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurt = np.sum(((data - mean) / std) ** 4) / n
        return float(kurt - 3)  # 减去3得到超额峰度
    
    def extract_skewness(self, data):
        """偏度（Skewness）
        
        描述信号分布的不对称程度
        
        Args:
            data: 输入信号数据
            
        Returns:
            偏度值
        """
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skew = np.sum(((data - mean) / std) ** 3) / n
        return float(skew)
    
    # ==================== 频域特征 ====================
    
    def extract_fft_features(self, data, sample_rate):
        """FFT频域特征
        
        Args:
            data: 输入信号数据
            sample_rate: 采样率
            
        Returns:
            频域特征字典
        """
        n = len(data)
        if n < 2:
            return {
                'median_frequency': 0.0,
                'mean_power_frequency': 0.0,
                'peak_frequency': 0.0,
                'spectral_energy': 0.0,
                'spectral_entropy': 0.0,
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0
            }
        
        # FFT计算
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(n, 1.0 / sample_rate)[:n // 2]
        power_spectrum = np.abs(fft_result)[:n // 2] ** 2 / n
        
        total_power = np.sum(power_spectrum)
        
        if total_power == 0:
            return {
                'median_frequency': 0.0,
                'mean_power_frequency': 0.0,
                'peak_frequency': 0.0,
                'spectral_energy': 0.0,
                'spectral_entropy': 0.0,
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0
            }
        
        # 中值频率（Median Frequency）
        cumulative_power = np.cumsum(power_spectrum)
        mf_idx = np.argmax(cumulative_power >= total_power / 2)
        median_frequency = frequencies[mf_idx] if mf_idx < len(frequencies) else frequencies[-1]
        
        # 平均功率频率（Mean Power Frequency）
        mean_power_frequency = np.sum(frequencies * power_spectrum) / total_power
        
        # 峰值频率（Peak Frequency）
        peak_idx = np.argmax(power_spectrum)
        peak_frequency = frequencies[peak_idx]
        
        # 频谱能量（Spectral Energy）
        spectral_energy = float(total_power)
        
        # 频谱熵（Spectral Entropy）
        psd_norm = power_spectrum / total_power
        psd_norm = psd_norm[psd_norm > 0]  # 移除零值避免log(0)
        spectral_entropy = float(entropy(psd_norm)) if len(psd_norm) > 0 else 0.0
        
        # 频谱质心（Spectral Centroid）
        spectral_centroid = np.sum(frequencies * power_spectrum) / total_power
        
        # 频谱扩展（Spectral Spread）
        spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_spectrum) / total_power)
        
        return {
            'median_frequency': float(median_frequency),
            'mean_power_frequency': float(mean_power_frequency),
            'peak_frequency': float(peak_frequency),
            'spectral_energy': float(spectral_energy),
            'spectral_entropy': float(spectral_entropy),
            'spectral_centroid': float(spectral_centroid),
            'spectral_spread': float(spectral_spread)
        }
    
    def extract_power_band(self, data, sample_rate, low_freq, high_freq):
        """频带功率
        
        计算指定频带内的功率
        
        Args:
            data: 输入信号数据
            sample_rate: 采样率
            low_freq: 低频边界
            high_freq: 高频边界
            
        Returns:
            频带功率
        """
        n = len(data)
        if n < 2:
            return 0.0
        
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(n, 1.0 / sample_rate)[:n // 2]
        power_spectrum = np.abs(fft_result)[:n // 2] ** 2 / n
        
        # 找到频带范围内的索引
        band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        band_power = np.sum(power_spectrum[band_mask])
        
        return float(band_power)
    
    # ==================== 时频域特征 ====================
    
    def extract_wavelet_features(self, data, wavelet='db4', level=3):
        """小波特征
        
        使用离散小波变换提取时频域特征
        
        Args:
            data: 输入信号数据
            wavelet: 小波基函数，默认'db4'
            level: 分解层数，默认3
            
        Returns:
            小波特征字典
        """
        if not PYWT_AVAILABLE:
            return {}
        
        if len(data) < 2 ** (level + 1):
            # 数据长度不足以进行指定层数的分解
            level = int(np.log2(len(data))) - 1
            if level < 1:
                return {}
        
        try:
            coeffs = pywt.wavedec(data, wavelet, level=level)
        except Exception as e:
            logger.warning(f"小波分解失败: {e}")
            return {}
        
        features = {}
        
        # 近似系数特征
        approx = coeffs[0]
        features['wavelet_approx_mean'] = float(np.mean(np.abs(approx)))
        features['wavelet_approx_std'] = float(np.std(approx))
        features['wavelet_approx_energy'] = float(np.sum(approx ** 2))
        
        # 各层细节系数特征
        total_energy = features['wavelet_approx_energy']
        for i, detail in enumerate(coeffs[1:], 1):
            detail_energy = np.sum(detail ** 2)
            total_energy += detail_energy
            
            features[f'wavelet_detail{i}_mean'] = float(np.mean(np.abs(detail)))
            features[f'wavelet_detail{i}_std'] = float(np.std(detail))
            features[f'wavelet_detail{i}_energy'] = float(detail_energy)
        
        # 小波熵
        if total_energy > 0:
            energies = [features['wavelet_approx_energy']]
            for i in range(1, level + 1):
                energies.append(features[f'wavelet_detail{i}_energy'])
            
            energies = np.array(energies)
            energies_norm = energies / total_energy
            energies_norm = energies_norm[energies_norm > 0]
            features['wavelet_entropy'] = float(entropy(energies_norm)) if len(energies_norm) > 0 else 0.0
        else:
            features['wavelet_entropy'] = 0.0
        
        return features
    
    # ==================== 单通道特征提取 ====================
    
    def extract_channel_features(self, data, sample_rate, prefix=''):
        """提取单通道的所有特征
        
        Args:
            data: 单通道信号数据
            sample_rate: 采样率
            prefix: 特征名前缀
            
        Returns:
            特征字典
        """
        features = {}
        
        # 时域特征
        features[f'{prefix}mean'] = self.extract_mean(data)
        features[f'{prefix}variance'] = self.extract_variance(data)
        features[f'{prefix}std'] = self.extract_std(data)
        features[f'{prefix}rms'] = self.extract_rms(data)
        features[f'{prefix}iemg'] = self.extract_iemg(data)
        features[f'{prefix}mav'] = self.extract_mav(data)
        features[f'{prefix}peak_to_peak'] = self.extract_peak_to_peak(data)
        features[f'{prefix}zc'] = self.extract_zc(data)
        features[f'{prefix}ssc'] = self.extract_ssc(data)
        features[f'{prefix}wl'] = self.extract_wl(data)
        features[f'{prefix}wamp'] = self.extract_wamp(data)
        features[f'{prefix}kurtosis'] = self.extract_kurtosis(data)
        features[f'{prefix}skewness'] = self.extract_skewness(data)
        
        # 频域特征
        fft_features = self.extract_fft_features(data, sample_rate)
        for key, value in fft_features.items():
            features[f'{prefix}{key}'] = value
        
        # 频带功率（EMG常用频带）
        if sample_rate >= 100:
            features[f'{prefix}power_20_50Hz'] = self.extract_power_band(data, sample_rate, 20, 50)
            features[f'{prefix}power_50_100Hz'] = self.extract_power_band(data, sample_rate, 50, 100)
            features[f'{prefix}power_20_100Hz'] = self.extract_power_band(data, sample_rate, 20, 100)
        
        # 时频域特征（小波）
        wavelet_features = self.extract_wavelet_features(data)
        for key, value in wavelet_features.items():
            features[f'{prefix}{key}'] = value
        
        return features
    
    # ==================== EMG特征提取 ====================
    
    def extract_emg_features(self, emg_data):
        """提取EMG特征（8通道）
        
        Args:
            emg_data: EMG数据，形状为(8, N)或(N, 8)的数组，或8个通道的数据列表
            
        Returns:
            特征字典
        """
        emg_data = np.array(emg_data)
        
        # 自动判断数据形状并转置
        if emg_data.ndim == 2:
            if emg_data.shape[0] == 8:
                # 形状为 (8, N)，无需转置
                pass
            elif emg_data.shape[1] == 8:
                # 形状为 (N, 8)，需要转置
                emg_data = emg_data.T
            else:
                raise ValueError(f"EMG数据形状错误: {emg_data.shape}，期望 (8, N) 或 (N, 8)")
        else:
            raise ValueError(f"EMG数据维度错误: {emg_data.ndim}，期望2维数组")
        
        features = {}
        
        for i in range(8):
            channel_features = self.extract_channel_features(
                emg_data[i], 
                self.emg_sample_rate, 
                prefix=f'emg_{i}_'
            )
            features.update(channel_features)
        
        # EMG通道间特征（跨通道统计）
        channel_means = [features[f'emg_{i}_mean'] for i in range(8)]
        channel_rms = [features[f'emg_{i}_rms'] for i in range(8)]
        
        features['emg_cross_mean_std'] = float(np.std(channel_means))
        features['emg_cross_rms_std'] = float(np.std(channel_rms))
        features['emg_cross_mean_max'] = float(np.max(channel_means))
        features['emg_cross_rms_max'] = float(np.max(channel_rms))
        
        return features
    
    # ==================== IMU特征提取 ====================
    
    def extract_imu_features(self, imu_data):
        """提取IMU特征（6通道：3轴陀螺仪 + 3轴加速度计）
        
        Args:
            imu_data: IMU数据，形状为(6, N)或(N, 6)的数组，或6个通道的数据列表
            
        Returns:
            特征字典
        """
        imu_data = np.array(imu_data)
        
        # 自动判断数据形状并转置
        if imu_data.ndim == 2:
            if imu_data.shape[0] == 6:
                # 形状为 (6, N)，无需转置
                pass
            elif imu_data.shape[1] == 6:
                # 形状为 (N, 6)，需要转置
                imu_data = imu_data.T
            else:
                raise ValueError(f"IMU数据形状错误: {imu_data.shape}，期望 (6, N) 或 (N, 6)")
        else:
            raise ValueError(f"IMU数据维度错误: {imu_data.ndim}，期望2维数组")
        
        features = {}
        
        # 陀螺仪特征（前3个通道）
        for i in range(3):
            channel_features = self.extract_channel_features(
                imu_data[i], 
                self.imu_sample_rate, 
                prefix=f'gyro_{i}_'
            )
            features.update(channel_features)
        
        # 加速度计特征（后3个通道）
        for i in range(3, 6):
            channel_features = self.extract_channel_features(
                imu_data[i], 
                self.imu_sample_rate, 
                prefix=f'accel_{i-3}_'
            )
            features.update(channel_features)
        
        # IMU合成特征
        # 加速度幅值
        accel_magnitude = np.sqrt(imu_data[3] ** 2 + imu_data[4] ** 2 + imu_data[5] ** 2)
        features['accel_magnitude_mean'] = float(np.mean(accel_magnitude))
        features['accel_magnitude_std'] = float(np.std(accel_magnitude))
        features['accel_magnitude_max'] = float(np.max(accel_magnitude))
        
        # 陀螺仪幅值
        gyro_magnitude = np.sqrt(imu_data[0] ** 2 + imu_data[1] ** 2 + imu_data[2] ** 2)
        features['gyro_magnitude_mean'] = float(np.mean(gyro_magnitude))
        features['gyro_magnitude_std'] = float(np.std(gyro_magnitude))
        features['gyro_magnitude_max'] = float(np.max(gyro_magnitude))
        
        return features
    
    # ==================== 融合特征提取 ====================
    
    def extract_all_features(self, emg_data, imu_data):
        """提取EMG和IMU的所有特征并融合
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            
        Returns:
            融合特征字典
        """
        features = {}
        
        # 提取EMG特征
        if emg_data is not None and len(emg_data) > 0:
            emg_features = self.extract_emg_features(emg_data)
            features.update(emg_features)
        
        # 提取IMU特征
        if imu_data is not None and len(imu_data) > 0:
            imu_features = self.extract_imu_features(imu_data)
            features.update(imu_features)
        
        return features
    
    def get_feature_names(self):
        """获取所有特征名称列表
        
        Returns:
            特征名称列表
        """
        # 使用示例数据生成特征名
        emg_sample = np.random.randn(8, 100)
        imu_sample = np.random.randn(6, 100)
        
        features = self.extract_all_features(emg_sample, imu_sample)
        return list(features.keys())
    
    def get_feature_count(self):
        """获取特征数量
        
        Returns:
            特征数量
        """
        return len(self.get_feature_names())


# 全局特征提取器实例
feature_extractor = FeatureExtractor()
