#!/usr/bin/env python3
"""
测试信号预处理功能
"""

import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_processor import SignalProcessor

def test_signal_processor():
    """测试信号预处理器"""
    print("=== 测试信号预处理器 ===")
    
    # 创建信号处理器
    processor = SignalProcessor(sample_rate=250.0, buffer_size=100)
    
    # 生成测试信号
    print("\n1. 生成测试信号...")
    emg_data = []
    for i in range(8):
        # 生成带噪声的正弦波信号
        t = np.linspace(0, 1, 250)
        signal = 1000 * np.sin(2 * np.pi * 10 * t)  # 10Hz信号
        noise = np.random.normal(0, 100, 250)  # 添加噪声
        emg_data.append((signal + noise).tolist())
    
    imu_data = []
    for i in range(6):
        # 生成IMU测试信号
        t = np.linspace(0, 1, 250)
        signal = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5Hz信号
        imu_data.append((signal * 10).tolist())
    
    print(f"   EMG数据形状: {np.array(emg_data).shape}")
    print(f"   IMU数据形状: {np.array(imu_data).shape}")
    
    # 测试1: 不启用预处理
    print("\n2. 测试不启用预处理...")
    processor.set_enabled(False)
    processed_emg = processor.process_emg_data(emg_data)
    processed_imu = processor.process_imu_data(imu_data)
    print(f"   处理后EMG数据形状: {np.array(processed_emg).shape}")
    print(f"   处理后IMU数据形状: {np.array(processed_imu).shape}")
    print(f"   数据是否相同: {np.allclose(emg_data, processed_emg)}")
    
    # 测试2: 启用带通滤波
    print("\n3. 测试带通滤波...")
    processor.set_enabled(True)
    processor.set_filter_type('bandpass')
    processor.set_normalize(False)
    
    processed_emg = processor.process_emg_data(emg_data)
    print(f"   处理后EMG数据形状: {np.array(processed_emg).shape}")
    print(f"   滤波后信号范围: [{np.min(processed_emg[0]):.2f}, {np.max(processed_emg[0]):.2f}]")
    
    # 测试3: 启用陷波滤波
    print("\n4. 测试陷波滤波...")
    processor.set_filter_type('notch')
    processed_emg = processor.process_emg_data(emg_data)
    print(f"   处理后EMG数据形状: {np.array(processed_emg).shape}")
    print(f"   滤波后信号范围: [{np.min(processed_emg[0]):.2f}, {np.max(processed_emg[0]):.2f}]")
    
    # 测试4: 启用双重滤波
    print("\n5. 测试双重滤波...")
    processor.set_filter_type('both')
    processed_emg = processor.process_emg_data(emg_data)
    print(f"   处理后EMG数据形状: {np.array(processed_emg).shape}")
    print(f"   滤波后信号范围: [{np.min(processed_emg[0]):.2f}, {np.max(processed_emg[0]):.2f}]")
    
    # 测试5: 启用归一化
    print("\n6. 测试归一化...")
    processor.set_filter_type('bandpass')
    processor.set_normalize(True)
    processed_emg = processor.process_emg_data(emg_data)
    print(f"   处理后EMG数据形状: {np.array(processed_emg).shape}")
    print(f"   归一化后信号范围: [{np.min(processed_emg[0]):.4f}, {np.max(processed_emg[0]):.4f}]")
    print(f"   是否在[-1, 1]范围内: {np.all(np.array(processed_emg) >= -1) and np.all(np.array(processed_emg) <= 1)}")
    
    # 测试6: 测试实时处理
    print("\n7. 测试实时处理...")
    processor.set_enabled(True)
    processor.set_filter_type('bandpass')
    processor.set_normalize(False)
    
    # 模拟实时数据流
    for i in range(10):
        emg_sample = [np.random.normal(0, 100) for _ in range(8)]
        imu_sample = [np.random.normal(0, 0.1) for _ in range(6)]
        processor.update_buffers(emg_sample, imu_sample)
    
    print(f"   已处理10个数据点")
    print(f"   预处理EMG缓冲区大小: {len(processor.processed_emg_buffer)}")
    print(f"   预处理IMU缓冲区大小: {len(processor.processed_imu_buffer)}")
    
    print("\n=== 测试完成 ===")
    print("所有测试通过！")

if __name__ == "__main__":
    try:
        test_signal_processor()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()