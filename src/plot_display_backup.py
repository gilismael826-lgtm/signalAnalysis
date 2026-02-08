#!/usr/bin/env python3
"""
绘图显示模块
负责实时数据的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import src.config as config
from src.config import running, x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max, emg_buffer, imu_buffer, logger
from src.config import PREPROCESSING_DISPLAY_MODE, processed_emg_y_min, processed_emg_y_max, processed_imu_y_min, processed_imu_y_max
from src.signal_processor import signal_processor


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
        
        # 根据显示模式选择数据源和Y轴范围
        if config.PREPROCESSING_DISPLAY_MODE == 'processed':
            emg_source = signal_processor.processed_emg_buffer
            imu_source = signal_processor.processed_imu_buffer
            signal_label = "预处理"
            
            # 根据是否开启归一化，选择合适的Y轴范围
            if signal_processor.normalize:
                # 开启归一化时，使用小范围
                emg_y_min_local = config.processed_emg_y_min
                emg_y_max_local = config.processed_emg_y_max
                imu_y_min_local = config.processed_imu_y_min
                imu_y_max_local = config.processed_imu_y_max
            else:
                # 关闭归一化时，使用与原始信号相似的范围
                emg_y_min_local = config.emg_y_min
                emg_y_max_local = config.emg_y_max
                imu_y_min_local = config.imu_y_min
                imu_y_max_local = config.imu_y_max
        else:
            emg_source = config.emg_buffer
            imu_source = config.imu_buffer
            signal_label = "原始"
            emg_y_min_local = config.emg_y_min
            emg_y_max_local = config.emg_y_max
            imu_y_min_local = config.imu_y_min
            imu_y_max_local = config.imu_y_max
        
        if len(emg_source) > 0:
            recent_emg = list(emg_source)[-int(config.x_scale):]
            # 使用相对索引作为X轴，确保均匀间隔
            times = list(range(len(recent_emg)))
            # 计算时间标签（秒）- 基于采样率
            time_labels = [t * (1/250) for t in times]
            emg_data = [d for _, d in recent_emg]
            
            emg_array = np.array(emg_data)
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
            for i in range(8):
                axs[i].plot(times, emg_array[:, i], color=colors[i], linewidth=1)
                axs[i].set_ylabel(f'EMG {i+1} ({signal_label})')
                axs[i].set_ylim(emg_y_min_local, emg_y_max_local)
                axs[i].grid(True, alpha=0.3)
                # 设置X轴标签为时间（秒）
                axs[i].set_xticks(times[::len(times)//10])
                axs[i].set_xticklabels([f'{tl:.1f}' for tl in time_labels[::len(time_labels)//10]])
                axs[i].set_xlabel('时间 (秒)')
        
        if len(imu_source) > 0:
            _, latest_imu = imu_source[-1]
            axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], latest_imu)
            axs[8].set_ylabel(f'IMU ({signal_label})')
            axs[8].set_ylim(imu_y_min_local, imu_y_max_local)
            axs[8].grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas.draw()
        
        root.after(50, lambda: update_plot(root, axs, canvas))
        
    except Exception as e:
        logger.error(f"更新图表失败: {e}")
        root.after(50, lambda: update_plot(root, axs, canvas))
