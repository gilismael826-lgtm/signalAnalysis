#!/usr/bin/env python3
"""测试主窗口模块导入"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.main_window import create_main_window, update_recognition_with_data
    print("主窗口模块导入成功!")
    print("手势识别函数可用:", update_recognition_with_data is not None)
except Exception as e:
    print(f"导入失败: {e}")
