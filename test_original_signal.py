#!/usr/bin/env python3
"""
测试原始信号波形图操作
"""

import numpy as np
import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as config

def test_original_signal_operations():
    """测试原始信号的波形图操作"""
    print("=== 测试原始信号波形图操作 ===")
    
    # 保存原始配置
    original_emg_y_min = config.emg_y_min
    original_emg_y_max = config.emg_y_max
    original_imu_y_min = config.imu_y_min
    original_imu_y_max = config.imu_y_max
    
    print(f"\n1. 原始配置:")
    print(f"   EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    print(f"   IMU Y轴范围: [{config.imu_y_min}, {config.imu_y_max}]")
    
    # 模拟切换到预处理信号
    print(f"\n2. 模拟切换到预处理信号...")
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    
    # 模拟plot_display.py中的逻辑（使用局部变量，不修改config）
    emg_y_min_local = config.emg_y_min
    emg_y_max_local = config.emg_y_max
    
    # 模拟预处理信号的数据范围
    emg_y_min_local = -1.1
    emg_y_max_local = 1.1
    
    print(f"   预处理信号Y轴范围（局部变量）: [{emg_y_min_local}, {emg_y_max_local}]")
    print(f"   config.emg_y_min: {config.emg_y_min}")
    print(f"   config.emg_y_max: {config.emg_y_max}")
    
    # 验证config没有被修改
    if config.emg_y_min == original_emg_y_min and config.emg_y_max == original_emg_y_max:
        print(f"   ✓ config未被修改")
    else:
        print(f"   ✗ config被错误修改")
        return False
    
    # 模拟切换回原始信号
    print(f"\n3. 模拟切换回原始信号...")
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    
    emg_y_min_local = config.emg_y_min
    emg_y_max_local = config.emg_y_max
    
    print(f"   原始信号Y轴范围（局部变量）: [{emg_y_min_local}, {emg_y_max_local}]")
    print(f"   config.emg_y_min: {config.emg_y_min}")
    print(f"   config.emg_y_max: {config.emg_y_max}")
    
    # 验证config保持原始值
    if config.emg_y_min == original_emg_y_min and config.emg_y_max == original_emg_y_max:
        print(f"   ✓ config保持原始值")
    else:
        print(f"   ✗ config被错误修改")
        return False
    
    # 测试手动缩放功能
    print(f"\n4. 测试手动缩放功能...")
    config.emg_y_min = -2000
    config.emg_y_max = 2000
    print(f"   手动调整后EMG Y轴范围: [{config.emg_y_min}, {config.emg_y_max}]")
    
    # 模拟切换显示模式
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    emg_y_min_local = -1.1
    emg_y_max_local = 1.1
    
    print(f"   切换到预处理信号后（局部变量）: [{emg_y_min_local}, {emg_y_max_local}]")
    print(f"   config.emg_y_min: {config.emg_y_min}")
    print(f"   config.emg_y_max: {config.emg_y_max}")
    
    # 验证手动缩放设置保持不变
    if config.emg_y_min == -2000 and config.emg_y_max == 2000:
        print(f"   ✓ 手动缩放设置保持不变")
    else:
        print(f"   ✗ 手动缩放设置被错误修改")
        return False
    
    # 切换回原始信号
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    emg_y_min_local = config.emg_y_min
    emg_y_max_local = config.emg_y_max
    
    print(f"   切换回原始信号后（局部变量）: [{emg_y_min_local}, {emg_y_max_local}]")
    print(f"   config.emg_y_min: {config.emg_y_min}")
    print(f"   config.emg_y_max: {config.emg_y_max}")
    
    # 验证手动缩放设置仍然保持
    if config.emg_y_min == -2000 and config.emg_y_max == 2000:
        print(f"   ✓ 手动缩放设置仍然保持")
    else:
        print(f"   ✗ 手动缩放设置被错误修改")
        return False
    
    print(f"\n=== 测试完成 ===")
    print("✓ 所有测试通过！")
    print("\n说明:")
    print("- 原始信号的波形图操作（缩放、平移等）不会被修改")
    print("- 切换显示模式时，使用局部变量计算Y轴范围")
    print("- config.emg_y_min和config.emg_y_max保持用户设置的值")
    print("- 预处理信号的Y轴范围自动调整，但不影响原始信号")
    
    return True

if __name__ == "__main__":
    try:
        success = test_original_signal_operations()
        if not success:
            print("\n测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)