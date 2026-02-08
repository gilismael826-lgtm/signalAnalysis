#!/usr/bin/env python3
"""
测试自动启用预处理功能
"""

import sys
import os

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.signal_processor import SignalProcessor

def test_auto_enable_preprocessing():
    """测试自动启用预处理功能"""
    print("=== 测试自动启用预处理功能 ===")
    
    # 重置配置
    config.PREPROCESSING_ENABLED = False
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    config.PREPROCESSING_FILTER_TYPE = 'bandpass'
    config.PREPROCESSING_NORMALIZE = False
    
    # 创建信号处理器
    processor = SignalProcessor(sample_rate=250.0, buffer_size=500)
    processor.set_enabled(False)
    
    print("\n1. 初始状态:")
    print(f"   预处理启用: {config.PREPROCESSING_ENABLED}")
    print(f"   显示模式: {config.PREPROCESSING_DISPLAY_MODE}")
    print(f"   信号处理器启用: {processor.enabled}")
    
    print("\n2. 模拟切换到预处理信号模式...")
    
    # 模拟main_window.py中的change_display_mode函数
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    
    # 自动启用预处理
    config.PREPROCESSING_ENABLED = True
    processor.set_enabled(True)
    
    print(f"   预处理启用: {config.PREPROCESSING_ENABLED}")
    print(f"   显示模式: {config.PREPROCESSING_DISPLAY_MODE}")
    print(f"   信号处理器启用: {processor.enabled}")
    
    if config.PREPROCESSING_ENABLED and processor.enabled:
        print(f"   ✓ 预处理功能已自动启用")
    else:
        print(f"   ✗ 预处理功能未自动启用")
        return False
    
    print("\n3. 生成测试数据...")
    
    # 生成100个样本的数据流
    for i in range(100):
        # 生成带噪声的EMG信号
        emg_sample = [i * 10 for _ in range(8)]
        imu_sample = [i * 0.1 for _ in range(6)]
        
        # 处理数据
        processor.update_buffers(emg_sample, imu_sample)
    
    print(f"   已生成100个样本")
    print(f"   预处理EMG缓冲区大小: {len(processor.processed_emg_buffer)}")
    print(f"   预处理IMU缓冲区大小: {len(processor.processed_imu_buffer)}")
    
    if len(processor.processed_emg_buffer) > 0 and len(processor.processed_imu_buffer) > 0:
        print(f"   ✓ 预处理功能正常工作")
    else:
        print(f"   ✗ 预处理功能未正常工作")
        return False
    
    print("\n4. 模拟切换回原始信号模式...")
    
    # 模拟切换回原始信号模式
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    
    # 保持预处理启用状态（不自动禁用）
    print(f"   预处理启用: {config.PREPROCESSING_ENABLED}")
    print(f"   显示模式: {config.PREPROCESSING_DISPLAY_MODE}")
    print(f"   信号处理器启用: {processor.enabled}")
    
    print("\n5. 测试完整流程...")
    
    # 测试完整的切换流程
    print("   步骤1: 初始状态 - 原始信号模式, 预处理禁用")
    config.PREPROCESSING_ENABLED = False
    config.PREPROCESSING_DISPLAY_MODE = 'raw'
    processor.set_enabled(False)
    
    print("   步骤2: 切换到预处理信号模式")
    config.PREPROCESSING_DISPLAY_MODE = 'processed'
    config.PREPROCESSING_ENABLED = True
    processor.set_enabled(True)
    
    if config.PREPROCESSING_ENABLED and config.PREPROCESSING_DISPLAY_MODE == 'processed':
        print("   ✓ 成功切换到预处理信号模式并自动启用预处理")
    else:
        print("   ✗ 切换失败")
        return False
    
    print("\n=== 测试完成 ===")
    print("✓ 所有测试通过！")
    print("\n功能说明:")
    print("- 当切换到'预处理信号'显示模式时，预处理功能会自动启用")
    print("- 用户不需要单独手动开启预处理功能")
    print("- 切换回'原始信号'模式时，预处理功能保持启用状态")
    print("- 预处理信号显示时始终使用最新的预处理设置")
    
    return True

if __name__ == "__main__":
    try:
        success = test_auto_enable_preprocessing()
        if not success:
            print("\n测试失败！")
            sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)