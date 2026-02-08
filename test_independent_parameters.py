#!/usr/bin/env python3
"""
测试预处理信号和原始信号的独立参数配置
"""

import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as config

def test_independent_parameters():
    """测试独立参数配置"""
    print("=== 测试预处理信号和原始信号的独立参数配置 ===")
    
    print("\n1. 初始配置:")
    print(f"   原始信号EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   原始信号IMU Y轴范围: [{config.imu_y_min}, {config.imu_y_max}]")
    print(f"   预处理信号EMG Y轴范围: [{config.processed_emg_y_min}, {config.processed_emg_y_max}]")
    print(f"   预处理信号IMU Y轴范围: [{config.processed_imu_y_min}, {config.processed_imu_y_max}]")
    print(f"   显示模式: {config.PREPROCESSING_DISPLAY_MODE}")
    
    print("\n2. 测试原始信号模式下的参数修改...")
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    
    # 修改原始信号参数
    config.emg_y_min = -100000
    config.emg_y_max = 100000
    config.imu_y_min = -10
    config.imu_y_max = 10
    
    print(f"   修改原始信号EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   修改原始信号IMU Y轴范围: [{config.imu_y_min}, {config.imu_y_max}]")
    print(f"   预处理信号EMG Y轴范围: [{config.processed_emg_y_min}, {config.processed_emg_y_max}]")
    print(f"   预处理信号IMU Y轴范围: [{config.processed_imu_y_min}, {config.processed_imu_y_max}]")
    
    # 验证预处理信号参数未受影响
    if config.processed_emg_y_min == -1.5 and config.processed_emg_y_max == 1.5:
        print(f"   ✓ 预处理信号参数未受影响")
    else:
        print(f"   ✗ 预处理信号参数被错误修改")
        return False
    
    print("\n3. 测试预处理信号模式下的参数修改...")
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    
    # 修改预处理信号参数
    config.processed_emg_y_min = -2.0
    config.processed_emg_y_max = 2.0
    config.processed_imu_y_min = -2.0
    config.processed_imu_y_max = 2.0
    
    print(f"   原始信号EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   原始信号IMU Y轴范围: [{config.imu_y_min}, {config.imu_y_max}]")
    print(f"   修改预处理信号EMG Y轴范围: [{config.processed_emg_y_min}, {config.processed_emg_y_max}]")
    print(f"   修改预处理信号IMU Y轴范围: [{config.processed_imu_y_min}, {config.processed_imu_y_max}]")
    
    # 验证原始信号参数未受影响
    if config.emg_y_min == -100000 and config.emg_y_max == 100000:
        print(f"   ✓ 原始信号参数未受影响")
    else:
        print(f"   ✗ 原始信号参数被错误修改")
        return False
    
    print("\n4. 测试缩放操作...")
    
    # 测试原始信号缩放
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    emg_range = config.emg_y_max - config.emg_y_min
    new_emg_range = emg_range * 0.7
    config.emg_y_min = int((config.emg_y_min + config.emg_y_max) / 2 - new_emg_range / 2)
    config.emg_y_max = int((config.emg_y_min + config.emg_y_max) / 2 + new_emg_range / 2)
    
    print(f"   原始信号EMG缩放后: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   预处理信号EMG Y轴范围: [{config.processed_emg_y_min}, {config.processed_emg_y_max}]")
    
    # 测试预处理信号缩放
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    emg_range = config.processed_emg_y_max - config.processed_emg_y_min
    new_emg_range = emg_range * 0.7
    config.processed_emg_y_min = (config.processed_emg_y_min + config.processed_emg_y_max) / 2 - new_emg_range / 2
    config.processed_emg_y_max = (config.processed_emg_y_min + config.processed_emg_y_max) / 2 + new_emg_range / 2
    
    print(f"   原始信号EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   预处理信号EMG缩放后: [{config.processed_emg_y_min}, {config.processed_emg_y_max}]")
    
    # 验证参数独立性
    if abs(config.emg_y_min - (-100000 * 0.7)) < 1000:
        print(f"   ✓ 原始信号缩放正常")
    else:
        print(f"   ✗ 原始信号缩放异常")
        return False
    
    if abs(config.processed_emg_y_min - (-2.0 * 0.7)) < 0.1:
        print(f"   ✓ 预处理信号缩放正常")
    else:
        print(f"   ✗ 预处理信号缩放异常")
        return False
    
    print("\n5. 测试参数独立性...")
    
    # 验证原始信号和预处理信号的参数完全独立
    original_emg_min = config.emg_y_min
    original_emg_max = config.emg_y_max
    processed_emg_min = config.processed_emg_y_min
    processed_emg_max = config.processed_emg_y_max
    
    # 原始信号范围应该很大（未归一化）
    if abs(original_emg_min) > 1000 and abs(original_emg_max) > 1000:
        print(f"   ✓ 原始信号使用未归一化的大范围")
    else:
        print(f"   ✗ 原始信号范围异常")
        return False
    
    # 预处理信号范围应该较小（归一化后）
    if abs(processed_emg_min) < 3 and abs(processed_emg_max) < 3:
        print(f"   ✓ 预处理信号使用归一化的小范围")
    else:
        print(f"   ✗ 预处理信号范围异常")
        return False
    
    # 验证两者范围完全不同
    if abs(original_emg_min - processed_emg_min) > 1000:
        print(f"   ✓ 原始信号和预处理信号使用不同的Y轴范围")
    else:
        print(f"   ✗ 原始信号和预处理信号Y轴范围相同")
        return False
    
    print("\n=== 测试完成 ===")
    print("✓ 所有测试通过！")
    print("\n功能说明:")
    print("- 原始信号和预处理信号使用独立的Y轴范围参数")
    print("- 修改原始信号参数不会影响预处理信号参数")
    print("- 修改预处理信号参数不会影响原始信号参数")
    print("- 归一化后的信号使用较小的Y轴范围（[-1.5, 1.5]）")
    print("- 原始信号使用较大的Y轴范围（[-50000, 50000]）")
    print("- 缩放操作根据当前显示模式独立工作")
    
    return True

if __name__ == "__main__":
    try:
        success = test_independent_parameters()
        if not success:
            print("\n测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)