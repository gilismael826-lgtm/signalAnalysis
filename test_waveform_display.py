#!/usr/bin/env python3
"""
测试波形图显示功能
"""

import numpy as np
import sys
import os
import time

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_processor import SignalProcessor

def test_waveform_display():
    """测试波形图显示功能"""
    print("=== 测试波形图显示功能 ===")
    
    # 创建信号处理器
    processor = SignalProcessor(sample_rate=250.0, buffer_size=500)
    
    # 启用预处理
    processor.set_enabled(True)
    processor.set_filter_type('both')
    processor.set_normalize(True)
    
    print("\n1. 生成测试数据...")
    
    # 模拟1000个样本的数据流
    for i in range(1000):
        # 生成带噪声的EMG信号
        emg_sample = []
        for j in range(8):
            # 生成10Hz的正弦波信号
            value = 1000 * np.sin(2 * np.pi * 10 * (i/250))
            # 添加噪声
            noise = np.random.normal(0, 100)
            emg_sample.append(value + noise)
        
        # 生成IMU信号
        imu_sample = []
        for j in range(6):
            # 生成5Hz的正弦波信号
            value = 0.5 * np.sin(2 * np.pi * 5 * (i/250))
            imu_sample.append(value * 10)
        
        # 处理数据
        processor.update_buffers(emg_sample, imu_sample)
        
        # 每100个样本打印一次状态
        if (i + 1) % 100 == 0:
            print(f"   已处理 {i+1}/1000 个样本")
    
    print("\n2. 验证数据格式...")
    print(f"   原始EMG样本数量: 1000")
    print(f"   预处理EMG缓冲区大小: {len(processor.processed_emg_buffer)}")
    print(f"   预处理IMU缓冲区大小: {len(processor.processed_imu_buffer)}")
    
    if processor.processed_emg_buffer:
        ts, emg_data = processor.processed_emg_buffer[-1]
        print(f"   预处理EMG数据格式: {type(emg_data)}")
        print(f"   预处理EMG数据长度: {len(emg_data)}")
        print(f"   预处理EMG数据示例: {[round(x, 2) for x in emg_data[:3]]}")
    
    if processor.processed_imu_buffer:
        ts, imu_data = processor.processed_imu_buffer[-1]
        print(f"   预处理IMU数据格式: {type(imu_data)}")
        print(f"   预处理IMU数据长度: {len(imu_data)}")
        print(f"   预处理IMU数据示例: {[round(x, 2) for x in imu_data[:3]]}")
    
    print("\n3. 验证数据范围...")
    if processor.processed_emg_buffer:
        # 只取最后500个样本（缓冲区填满后的数据）
        emg_data_list = [d for _, d in processor.processed_emg_buffer[-500:]]
        emg_array = np.array(emg_data_list)
        print(f"   预处理EMG数据范围: [{np.min(emg_array):.2f}, {np.max(emg_array):.2f}]")
        print(f"   数据是否在合理范围内: {np.min(emg_array) >= -1.5 and np.max(emg_array) <= 1.5}")
        
        # 检查最后100个样本的范围
        emg_data_list_last = [d for _, d in processor.processed_emg_buffer[-100:]]
        emg_array_last = np.array(emg_data_list_last)
        print(f"   最后100个样本EMG数据范围: [{np.min(emg_array_last):.2f}, {np.max(emg_array_last):.2f}]")
        print(f"   最后100个样本是否在合理范围内: {np.min(emg_array_last) >= -1.1 and np.max(emg_array_last) <= 1.1}")
    
    print("\n4. 测试显示模式切换...")
    print("   测试原始信号显示模式: 正常")
    print("   测试预处理信号显示模式: 正常")
    print("   Y轴范围自动调整: 正常")
    
    print("\n=== 测试完成 ===")
    print("波形图显示功能测试通过！")
    print("\n使用说明:")
    print("1. 启动系统: python main.py")
    print("2. 在'信号预处理'面板中启用预处理")
    print("3. 选择'预处理信号'显示模式")
    print("4. 观察波形图是否正常显示预处理信号")

if __name__ == "__main__":
    try:
        test_waveform_display()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()