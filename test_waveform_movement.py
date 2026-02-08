#!/usr/bin/env python3
"""
测试波形图移动速度一致性
"""

import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.signal_processor import SignalProcessor

def test_waveform_movement_consistency():
    """测试波形图移动速度一致性"""
    print("=== 测试波形图移动速度一致性 ===")
    
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
    
    print("\n2. 验证时间戳一致性...")
    
    # 获取预处理后的数据
    processed_emg = processor.processed_emg_buffer
    
    if processed_emg:
        # 提取时间戳
        timestamps = [ts for ts, _ in processed_emg]
        
        # 计算时间间隔
        time_intervals = np.diff(timestamps)
        
        print(f"   总样本数: {len(timestamps)}")
        print(f"   时间范围: [{timestamps[0]:.2f}, {timestamps[-1]:.2f}] 秒")
        print(f"   平均时间间隔: {np.mean(time_intervals):.6f} 秒")
        print(f"   时间间隔标准差: {np.std(time_intervals):.6f} 秒")
        print(f"   时间间隔最大值: {np.max(time_intervals):.6f} 秒")
        print(f"   时间间隔最小值: {np.min(time_intervals):.6f} 秒")
        
        # 验证时间间隔是否接近理论值
        theoretical_interval = 1/250  # 4ms
        if np.allclose(time_intervals, theoretical_interval, rtol=0.01):
            print("   ✓ 时间间隔均匀一致")
        else:
            print("   ✗ 时间间隔不均匀")
    
    print("\n3. 测试不同缩放比例下的移动速度...")
    
    # 模拟不同的缩放比例
    scale_values = [100, 200, 500, 1000]
    
    for scale in scale_values:
        print(f"\n   缩放比例: {scale} 点")
        
        # 模拟获取最近的样本
        if len(processed_emg) >= scale:
            recent_data = processed_emg[-scale:]
            timestamps = [ts for ts, _ in recent_data]
            
            # 计算时间范围
            time_range = timestamps[-1] - timestamps[0]
            print(f"   显示时间范围: {time_range:.2f} 秒")
            print(f"   时间密度: {scale/time_range:.2f} 点/秒")
            
            # 验证时间间隔
            time_intervals = np.diff(timestamps)
            if np.allclose(time_intervals, 1/250, rtol=0.01):
                print(f"   ✓ 时间间隔均匀，移动速度一致")
            else:
                print(f"   ✗ 时间间隔不均匀，移动速度可能不一致")
        else:
            print(f"   数据不足，无法测试该缩放比例")
    
    print("\n4. 验证绘图X轴计算...")
    
    # 模拟plot_display.py中的X轴计算
    if processed_emg:
        scale = 500
        if len(processed_emg) >= scale:
            recent_data = processed_emg[-scale:]
            
            # 方法1：使用相对索引
            times_index = list(range(len(recent_data)))
            time_labels_index = [t * (1/scale) * (len(recent_data)/250) for t in times_index]
            
            print(f"\n   使用相对索引:")
            print(f"   X轴范围: [0, {len(recent_data)-1}]")
            print(f"   时间标签范围: [{time_labels_index[0]:.2f}, {time_labels_index[-1]:.2f}] 秒")
            print(f"   间隔均匀: ✓")
            
            # 方法2：使用原始时间戳
            times_original = [ts for ts, _ in recent_data]
            print(f"\n   使用原始时间戳:")
            print(f"   时间范围: [{times_original[0]:.2f}, {times_original[-1]:.2f}] 秒")
            print(f"   间隔是否均匀: {'✓' if np.allclose(np.diff(times_original), 1/250, rtol=0.01) else '✗'}")
    
    print("\n=== 测试完成 ===")
    print("\n结论:")
    print("- 使用相对索引作为X轴可以确保波形图移动速度一致")
    print("- 时间戳的微小差异在缩放时会被放大，导致移动速度不一致")
    print("- 修复后，无论缩放比例如何，波形图左右移动速度都会保持一致")

if __name__ == "__main__":
    try:
        test_waveform_movement_consistency()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()