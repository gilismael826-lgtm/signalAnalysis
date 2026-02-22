#!/usr/bin/env python3
"""
特征提取模块测试
验证特征提取功能的正确性
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor, feature_extractor


def test_basic_features():
    """测试基本特征提取"""
    print("=" * 60)
    print("测试基本特征提取")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建测试数据
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    print(f"测试数据: {data}")
    print(f"均值: {extractor.extract_mean(data):.4f}")
    print(f"方差: {extractor.extract_variance(data):.4f}")
    print(f"标准差: {extractor.extract_std(data):.4f}")
    print(f"RMS: {extractor.extract_rms(data):.4f}")
    print(f"IEMG: {extractor.extract_iemg(data):.4f}")
    print(f"MAV: {extractor.extract_mav(data):.4f}")
    print(f"峰峰值: {extractor.extract_peak_to_peak(data):.4f}")
    print(f"过零率: {extractor.extract_zc(data)}")
    print(f"波形长度: {extractor.extract_wl(data):.4f}")
    print()


def test_frequency_features():
    """测试频域特征提取"""
    print("=" * 60)
    print("测试频域特征提取")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建正弦波测试数据（50Hz）
    sample_rate = 250
    t = np.linspace(0, 1, sample_rate)
    data = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(sample_rate)
    
    print(f"测试数据: 50Hz正弦波 + 噪声, 采样率={sample_rate}Hz")
    
    fft_features = extractor.extract_fft_features(data, sample_rate)
    for key, value in fft_features.items():
        print(f"{key}: {value:.4f}")
    print()


def test_wavelet_features():
    """测试小波特征提取"""
    print("=" * 60)
    print("测试小波特征提取")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建测试数据
    sample_rate = 250
    t = np.linspace(0, 1, sample_rate)
    data = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    print(f"测试数据: 10Hz + 50Hz混合信号")
    
    wavelet_features = extractor.extract_wavelet_features(data)
    for key, value in wavelet_features.items():
        print(f"{key}: {value:.4f}")
    print()


def test_emg_features():
    """测试EMG特征提取"""
    print("=" * 60)
    print("测试EMG特征提取（8通道）")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建模拟EMG数据（8通道，150样本）
    np.random.seed(42)
    emg_data = np.random.randn(8, 150) * 1000
    
    print(f"EMG数据形状: {emg_data.shape}")
    
    features = extractor.extract_emg_features(emg_data)
    
    print(f"提取特征数量: {len(features)}")
    print("\n部分特征示例:")
    for i, (key, value) in enumerate(features.items()):
        if i < 20:
            print(f"  {key}: {value:.4f}")
    print(f"  ... (共{len(features)}个特征)")
    print()


def test_imu_features():
    """测试IMU特征提取"""
    print("=" * 60)
    print("测试IMU特征提取（6通道）")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建模拟IMU数据（6通道，62样本）
    np.random.seed(42)
    imu_data = np.random.randn(6, 62) * 0.5
    
    print(f"IMU数据形状: {imu_data.shape}")
    
    features = extractor.extract_imu_features(imu_data)
    
    print(f"提取特征数量: {len(features)}")
    print("\n部分特征示例:")
    for i, (key, value) in enumerate(features.items()):
        if i < 20:
            print(f"  {key}: {value:.4f}")
    print(f"  ... (共{len(features)}个特征)")
    print()


def test_fused_features():
    """测试融合特征提取"""
    print("=" * 60)
    print("测试EMG+IMU融合特征提取")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 创建模拟数据
    np.random.seed(42)
    emg_data = np.random.randn(8, 150) * 1000
    imu_data = np.random.randn(6, 62) * 0.5
    
    print(f"EMG数据形状: {emg_data.shape}")
    print(f"IMU数据形状: {imu_data.shape}")
    
    features = extractor.extract_all_features(emg_data, imu_data)
    
    print(f"\n融合特征数量: {len(features)}")
    
    # 统计各类特征数量
    emg_count = sum(1 for k in features.keys() if k.startswith('emg_'))
    gyro_count = sum(1 for k in features.keys() if k.startswith('gyro_'))
    accel_count = sum(1 for k in features.keys() if k.startswith('accel_'))
    
    print(f"  - EMG特征: {emg_count}")
    print(f"  - 陀螺仪特征: {gyro_count}")
    print(f"  - 加速度计特征: {accel_count}")
    print()


def test_feature_names():
    """测试特征名称获取"""
    print("=" * 60)
    print("测试特征名称列表")
    print("=" * 60)
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    feature_names = extractor.get_feature_names()
    feature_count = extractor.get_feature_count()
    
    print(f"特征总数: {feature_count}")
    print("\n特征名称列表:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:3d}. {name}")
    print()


def test_global_instance():
    """测试全局实例"""
    print("=" * 60)
    print("测试全局特征提取器实例")
    print("=" * 60)
    
    print(f"EMG采样率: {feature_extractor.emg_sample_rate} Hz")
    print(f"IMU采样率: {feature_extractor.imu_sample_rate} Hz")
    print(f"特征数量: {feature_extractor.get_feature_count()}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("特征提取模块测试")
    print("=" * 60 + "\n")
    
    test_basic_features()
    test_frequency_features()
    test_wavelet_features()
    test_emg_features()
    test_imu_features()
    test_fused_features()
    test_feature_names()
    test_global_instance()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
