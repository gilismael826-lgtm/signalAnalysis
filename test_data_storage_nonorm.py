#!/usr/bin/env python3
"""
测试数据存储功能（禁用归一化）
验证原始数据和预处理数据的独立存储
"""

import sys
import os
import time

# 添加项目路径到sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src import data_manager
from src.signal_processor import signal_processor


def test_data_storage_nonorm():
    """测试数据存储功能（禁用归一化）"""
    print("=== 测试数据存储功能（禁用归一化）===")
    
    # 配置测试参数
    config.SAVE_DATA = True
    config.DATA_DIR = "test_data"
    config.PREPROCESSING_ENABLED = True
    config.PREPROCESSING_NORMALIZE = False  # 禁用归一化
    
    # 初始化数据文件
    print("\n1. 初始化数据文件...")
    success = data_manager.init_data_files()
    if not success:
        print("✗ 初始化数据文件失败")
        return False
    print("✓ 数据文件初始化成功")
    
    # 获取数据文件信息
    file_info = data_manager.get_data_file_info()
    print("\n2. 当前数据文件:")
    print(f"   原始EMG: {file_info['raw_emg']}")
    print(f"   原始IMU: {file_info['raw_imu']}")
    print(f"   预处理EMG: {file_info['processed_emg']}")
    print(f"   预处理IMU: {file_info['processed_imu']}")
    
    # 检查文件名是否包含nonorm
    has_nonorm = False
    for key, path in file_info.items():
        if 'processed_' in key and 'nonorm' in path:
            has_nonorm = True
            break
    
    if has_nonorm:
        print("✓ 预处理数据文件名包含'nonorm'标识")
    else:
        print("✗ 预处理数据文件名未包含'nonorm'标识")
        return False
    
    # 模拟数据保存
    print("\n3. 模拟数据保存...")
    timestamp = time.time()
    
    # 保存原始EMG数据
    emg_data = [100, 200, 300, 400, 500, 600, 700, 800]
    data_manager.save_data_point('EMG', timestamp, emg_data, is_processed=False)
    print("   ✓ 保存原始EMG数据")
    
    # 保存原始IMU数据
    imu_data = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    data_manager.save_data_point('IMU', timestamp, imu_data, is_processed=False)
    print("   ✓ 保存原始IMU数据")
    
    # 保存预处理EMG数据
    processed_emg_data = [95, 195, 295, 395, 495, 595, 695, 795]  # 未归一化的数据
    data_manager.save_data_point('EMG', timestamp, processed_emg_data, is_processed=True)
    print("   ✓ 保存预处理EMG数据")
    
    # 保存预处理IMU数据
    processed_imu_data = [0.09, 0.19, 0.29, 0.9, 1.9, 2.9]  # 未归一化的数据
    data_manager.save_data_point('IMU', timestamp, processed_imu_data, is_processed=True)
    print("   ✓ 保存预处理IMU数据")
    
    # 关闭数据文件
    print("\n4. 关闭数据文件...")
    data_manager.close_data_files()
    print("   ✓ 数据文件已关闭")
    
    # 检查文件是否存在
    print("\n5. 检查数据文件...")
    test_files = [
        file_info['raw_emg'],
        file_info['raw_imu'],
        file_info['processed_emg'],
        file_info['processed_imu']
    ]
    
    all_exist = True
    for file_path in test_files:
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✓ {os.path.basename(file_path)} ({file_size} bytes)")
        else:
            print(f"   ✗ {os.path.basename(file_path)} 不存在")
            all_exist = False
    
    # 读取并显示文件内容
    print("\n6. 查看文件内容...")
    for file_path in test_files:
        if file_path and os.path.exists(file_path):
            print(f"\n   {os.path.basename(file_path)}:")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[:5]:  # 只显示前5行
                        print(f"      {line.strip()}")
                    if len(lines) > 5:
                        print(f"      ... (共{len(lines)}行)")
            except Exception as e:
                print(f"      ✗ 读取失败: {e}")
    
    print("\n=== 测试完成 ===")
    
    if all_exist and has_nonorm:
        print("✓ 所有测试通过！")
        print("\n功能说明:")
        print("- 原始数据保存到以'raw_'开头的文件")
        print("- 预处理数据保存到以'processed_'开头的文件")
        print("- 启用归一化时，文件名包含'norm'标识")
        print("- 禁用归一化时，文件名包含'nonorm'标识")
        print("- EMG和IMU数据分别保存到不同的文件")
        print("- 文件名包含时间戳和序号，便于区分")
        return True
    else:
        print("✗ 测试失败！")
        return False


if __name__ == "__main__":
    try:
        success = test_data_storage_nonorm()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)