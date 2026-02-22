#!/usr/bin/env python3
"""
主程序入口
"""

import os
import sys
import src.config as config
from src.config import logger
from src.main_window import create_main_window


def main():
    """主函数"""
    try:
        # 确保必要的目录存在
        directories = ['logs', 'data', 'data/raw', 'data/datasets', 'data/users', 'models']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        logger.info("=== 启动 EMG/IMU 系统控制中心 ===")
        logger.info(f"系统配置: 波特率={config.BAUDRATE}, 保存数据={config.SAVE_DATA}")
        
        # 创建并运行主窗口
        root = create_main_window()
        root.mainloop()
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        print(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
