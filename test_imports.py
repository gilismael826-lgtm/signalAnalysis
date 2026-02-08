#!/usr/bin/env python3
"""
模块引用测试脚本
检查所有模块之间的导入关系是否正确
"""

import sys
import os

# 添加 src 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试所有模块的导入"""
    print("=" * 60)
    print("开始测试模块导入...")
    print("=" * 60)
    
    modules = [
        "config",
        "data_manager",
        "data_parser",
        "serial_comm",
        "plot_display",
        "zoom_control",
        "system_control_module",
        "main_window"
    ]
    
    success_count = 0
    failed_count = 0
    
    for module_name in modules:
        try:
            print(f"\n测试导入模块: {module_name}")
            module = __import__(module_name)
            print(f"  ✓ {module_name} 导入成功")
            success_count += 1
        except Exception as e:
            print(f"  ✗ {module_name} 导入失败: {e}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"导入测试完成: 成功 {success_count}/{len(modules)}, 失败 {failed_count}/{len(modules)}")
    print("=" * 60)
    
    return failed_count == 0

def test_config():
    """测试config模块"""
    print("\n" + "=" * 60)
    print("测试 config 模块...")
    print("=" * 60)
    
    try:
        from config import (
            SERIAL_PORT, BAUDRATE, SAVE_DATA, DATA_DIR,
            emg_buffer, imu_buffer, logger,
            x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max,
            zoom_presets
        )
        print("  ✓ 所有配置变量导入成功")
        print(f"  - SERIAL_PORT: {SERIAL_PORT}")
        print(f"  - BAUDRATE: {BAUDRATE}")
        print(f"  - SAVE_DATA: {SAVE_DATA}")
        print(f"  - DATA_DIR: {DATA_DIR}")
        print(f"  - x_scale: {x_scale}")
        print(f"  - emg_y_range: {emg_y_min} ~ {emg_y_max}")
        print(f"  - imu_y_range: {imu_y_min} ~ {imu_y_max}")
        print(f"  - zoom_presets: {list(zoom_presets.keys())}")
        return True
    except Exception as e:
        print(f"  ✗ config 模块测试失败: {e}")
        return False

def test_data_manager():
    """测试data_manager模块"""
    print("\n" + "=" * 60)
    print("测试 data_manager 模块...")
    print("=" * 60)
    
    try:
        from data_manager import (
            init_data_files, save_data_point,
            rotate_data_files, close_data_files
        )
        print("  ✓ data_manager 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ data_manager 模块测试失败: {e}")
        return False

def test_data_parser():
    """测试data_parser模块"""
    print("\n" + "=" * 60)
    print("测试 data_parser 模块...")
    print("=" * 60)
    
    try:
        from data_parser import parse_packet
        print("  ✓ data_parser 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ data_parser 模块测试失败: {e}")
        return False

def test_serial_comm():
    """测试serial_comm模块"""
    print("\n" + "=" * 60)
    print("测试 serial_comm 模块...")
    print("=" * 60)
    
    try:
        from serial_comm import serial_reader
        print("  ✓ serial_comm 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ serial_comm 模块测试失败: {e}")
        return False

def test_plot_display():
    """测试plot_display模块"""
    print("\n" + "=" * 60)
    print("测试 plot_display 模块...")
    print("=" * 60)
    
    try:
        from plot_display import update_plot
        print("  ✓ plot_display 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ plot_display 模块测试失败: {e}")
        return False

def test_zoom_control():
    """测试zoom_control模块"""
    print("\n" + "=" * 60)
    print("测试 zoom_control 模块...")
    print("=" * 60)
    
    try:
        from zoom_control import (
            quick_zoom_in, quick_zoom_out, quick_reset_zoom,
            quick_emg_zoom_in, quick_emg_zoom_out,
            quick_imu_zoom_in, quick_imu_zoom_out,
            zoom_control, apply_zoom_preset
        )
        print("  ✓ zoom_control 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ zoom_control 模块测试失败: {e}")
        return False

def test_system_control_module():
    """测试system_control_module模块"""
    print("\n" + "=" * 60)
    print("测试 system_control_module 模块...")
    print("=" * 60)
    
    try:
        from system_control_module import (
            start_acquisition, stop_acquisition, clear_buffers,
            export_config, analyze_data, view_documentation, system_settings
        )
        print("  ✓ system_control_module 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ system_control_module 模块测试失败: {e}")
        return False

def test_main_window():
    """测试main_window模块"""
    print("\n" + "=" * 60)
    print("测试 main_window 模块...")
    print("=" * 60)
    
    try:
        from main_window import create_main_window
        print("  ✓ main_window 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ✗ main_window 模块测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("模块引用测试脚本")
    print("=" * 60)
    
    # 测试所有模块的导入
    all_imports_ok = test_imports()
    
    if not all_imports_ok:
        print("\n" + "!" * 60)
        print("警告: 部分模块导入失败，请检查导入关系")
        print("!" * 60)
        return False
    
    # 测试各个模块
    tests = [
        ("config", test_config),
        ("data_manager", test_data_manager),
        ("data_parser", test_data_parser),
        ("serial_comm", test_serial_comm),
        ("plot_display", test_plot_display),
        ("zoom_control", test_zoom_control),
        ("system_control_module", test_system_control_module),
        ("main_window", test_main_window)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！模块引用关系正确。")
    else:
        print("✗ 部分测试失败，请检查模块引用关系。")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
