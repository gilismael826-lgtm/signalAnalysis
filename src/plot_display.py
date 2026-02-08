#!/usr/bin/env python3
"""
绘图显示模块
负责实时数据的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import src.config as config
from src.config import running, x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max, emg_buffer, imu_buffer, logger


def update_plot(root, axs, canvas):
    """更新图表
    
    Args:
        root: Tkinter根窗口对象
        axs: 图表轴对象列表
        canvas: 画布对象
    """
    if not config.running:
        return
    
    try:
        if not axs or not canvas:
            logger.error("图表对象未初始化")
            root.after(50, lambda: update_plot(root, axs, canvas))
            return
        
        for ax in axs:
            ax.clear()
        
        if len(config.emg_buffer) > 0:
            recent_emg = list(config.emg_buffer)[-int(config.x_scale):]
            times = [t for t, _ in recent_emg]
            emg_data = [d for _, d in recent_emg]
            
            emg_array = np.array(emg_data)
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
            for i in range(8):
                axs[i].plot(times, emg_array[:, i], color=colors[i], linewidth=1)
                axs[i].set_ylabel(f'EMG {i+1} (μV)')
                axs[i].set_ylim(config.emg_y_min, config.emg_y_max)
                axs[i].grid(True, alpha=0.3)
        
        if len(config.imu_buffer) > 0:
            _, latest_imu = config.imu_buffer[-1]
            axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], latest_imu)
            axs[8].set_ylabel('IMU Values')
            axs[8].set_ylim(config.imu_y_min, config.imu_y_max)
            axs[8].grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas.draw()
        
        root.after(50, lambda: update_plot(root, axs, canvas))
        
    except Exception as e:
        logger.error(f"更新图表失败: {e}")
        root.after(50, lambda: update_plot(root, axs, canvas))
