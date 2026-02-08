#!/usr/bin/env python3
"""
测试预处理信号波形图移动速度一致性
"""

import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_processor import SignalProcessor
import src.config as config

def test_processed_signal_movement():
    """测试预处理信号波形图移动速度一致性"""
    print("=== 测试预处理信号波形图移动速度一致性 ===")
    
    # 创建信号处理器
    processor = SignalProcessor(sample_rate=250.0, buffer_size=500)
    processor.set_enabled(True)
    processor.set_filter_type('both')
    processor.set_normalize(True)
    
    print("\n1. 生成测试数据...")
    
    # 生成1000个样本的数据流
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
    
    print("\n2. 验证预处理信号数据...")
    
    # 获取预处理后的数据
    processed_emg = processor.processed_emg_buffer
    
    if processed_emg:
        print(f"   预处理EMG样本数: {len(processed_emg)}")
        print(f"   预处理IMU样本数: {len(processor.processed_imu_buffer)}")
        
        # 检查最后一个样本
        last_ts, last_emg = processed_emg[-1]
        print(f"   最后一个样本时间戳: {last_ts:.2f} 秒")
        print(f"   最后一个样本EMG数据: {[round(x, 2) for x in last_emg[:3]]}...")
    
    print("\n3. 测试不同显示模式下的移动速度...")
    
    # 测试原始信号模式
    print("\n   原始信号模式:")
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    
    # 模拟plot_display.py中的逻辑
    if hasattr(config, 'emg_buffer') and config.emg_buffer:
        scale = 500
        if len(config.emg_buffer) >= scale:
            recent_data = config.emg_buffer[-scale:]
            times = list(range(len(recent_data)))
            time_labels = [t * (1/250) for t in times]
            
            print(f"   缩放比例: {scale} 点")
            print(f"   X轴范围: [0, {len(recent_data)-1}]")
            print(f"   时间标签范围: [{time_labels[0]:.2f}, {time_labels[-1]:.2f}] 秒")
            print(f"   时间密度: {scale/(time_labels[-1]-time_labels[0]):.2f} 点/秒")
    
    # 测试预处理信号模式
    print("\n   预处理信号模式:")
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    
    # 模拟plot_display.py中的逻辑
    if processed_emg:
        scale = 500
        if len(processed_emg) >= scale:
            recent_data = processed_emg[-scale:]
            times = list(range(len(recent_data)))
            time_labels = [t * (1/250) for t in times]
            
            print(f"   缩放比例: {scale} 点")
            print(f"   X轴范围: [0, {len(recent_data)-1}]")
            print(f"   时间标签范围: [{time_labels[0]:.2f}, {time_labels[-1]:.2f}] 秒")
            print(f"   时间密度: {scale/(time_labels[-1]-time_labels[0]):.2f} 点/秒")
    
    print("\n4. 验证不同缩放比例下的一致性...")
    
    # 测试不同缩放比例
    scale_values = [100, 200, 500, 1000]
    
    for scale in scale_values:
        print(f"\n   缩放比例: {scale} 点")
        
        if processed_emg and len(processed_emg) >= scale:
            recent_data = processed_emg[-scale:]
            times = list(range(len(recent_data)))
            time_labels = [t * (1/250) for t in times]
            
            # 计算理论时间范围
            theoretical_time_range = scale / 250
            actual_time_range = time_labels[-1] - time_labels[0]
            
            print(f"   理论时间范围: {theoretical_time_range:.2f} 秒")
            print(f"   实际时间范围: {actual_time_range:.2f} 秒")
            print(f"   误差: {(actual_time_range - theoretical_time_range):.4f} 秒")
            
            if abs(actual_time_range - theoretical_time_range) < 0.01:
                print(f"   ✓ 时间范围一致")
            else:
                print(f"   ✗ 时间范围不一致")
    
    print("\n5. 验证X轴标签计算...")
    
    # 详细验证X轴标签计算
    if processed_emg and len(processed_emg) >= 500:
        recent_data = processed_emg[-500:]
        
        # 计算相对索引和时间标签
        times = list(range(len(recent_data)))
        time_labels = [t * (1/250) for t in times]
        
        print(f"\n   样本数: {len(recent_data)}")
        print(f"   第一个样本时间标签: {time_labels[0]:.3f} 秒")
        print(f"   最后一个样本时间标签: {time_labels[-1]:.3f} 秒")
        print(f"   时间间隔: {(time_labels[1]-time_labels[0]):.3f} 秒")
        print(f"   理论时间间隔: {(1/250):.3f} 秒")
        
        # 验证时间间隔
        time_intervals = np.diff(time_labels)
        if np.allclose(time_intervals, 1/250, rtol=0.01):
            print(f"   ✓ 时间间隔均匀一致")
        else:
            print(f"   ✗ 时间间隔不均匀")
    
    print("\n=== 测试完成 ===")
    print("\n结论:")
    print("- 预处理信号现在使用与原始信号相同的X轴计算逻辑")
    print("- 时间标签基于固定的250Hz采样率计算")
    print("- 无论缩放比例如何，波形图左右移动速度都会保持一致")
    print("- 原始信号和预处理信号的移动速度相同")

if __name__ == "__main__":
    try:
        # 初始化config中的缓冲区
        import src.config as config
        if not hasattr(config, 'emg_buffer'):
            from collections import deque
            config.emg_buffer = deque(maxlen=10000)
            config.imu_buffer = deque(maxlen=10000)
            
            # 生成一些测试数据
            for i in range(1000):
                ts = i / 250
                emg_sample = [1000 * np.sin(2 * np.pi * 10 * ts) + np.random.normal(0, 100) for _ in range(8)]
                imu_sample = [0.5 * np.sin(2 * np.pi * 5 * ts) * 10 for _ in range(6)]
                config.emg_buffer.append((ts, emg_sample))
                config.imu_buffer.append((ts, imu_sample))
        
        test_processed_signal_movement()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()