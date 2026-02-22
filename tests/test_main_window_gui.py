#!/usr/bin/env python3
"""测试主窗口模块导入"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.main_window import create_main_window
    print("主窗口模块导入成功!")
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()
