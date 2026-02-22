#!/usr/bin/env python3
"""
绘图显示模块（set_data优化版）
使用set_data方法更新图表，提高性能
"""

import numpy as np
import matplotlib.pyplot as plt
import src.config as config
from src.config import running, x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max, emg_buffer, imu_buffer, logger
from src.config import PREPROCESSING_DISPLAY_MODE, processed_emg_y_min, processed_emg_y_max, processed_imu_y_min, processed_imu_y_max
from src.signal_processor import signal_processor

# 全局变量，存储图表对象
lines = []  # EMG线条对象
bar_container = None  # IMU柱状图对象
initialized = False  # 是否已初始化
last_emg_data = None
last_imu_data = None
last_signal_label = None
last_emg_y_range = None
last_imu_y_range = None


def init_plot(axs):
    """初始化图表对象
    
    Args:
        axs: 图表轴对象列表
    """
    global lines, bar_container, initialized
    
    if initialized:
        return
    
    # 初始化EMG线条
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
    lines = []
    for i in range(8):
        line, = axs[i].plot([], [], color=colors[i], linewidth=1)
        lines.append(line)
        axs[i].set_ylabel(f'EMG {i+1}')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_xlabel('时间 (秒)')
    
    # 初始化IMU柱状图
    bar_container = axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], [0]*6)
    axs[8].set_ylabel('IMU')
    axs[8].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    initialized = True
    logger.info("图表对象初始化完成（set_data模式）")


def update_plot(root, axs, canvas):
    """更新图表（set_data优化版）
    
    Args:
        root: Tkinter根窗口对象
        axs: 图表轴对象列表
        canvas: 画布对象
    """
    global last_emg_data, last_imu_data, last_signal_label
    global last_emg_y_range, last_imu_y_range
    
    if not config.running:
        return
    
    try:
        if not axs or not canvas:
            logger.error("图表对象未初始化")
            root.after(50, lambda: update_plot(root, axs, canvas))
            return
        
        # 首次运行时初始化
        if not initialized:
            init_plot(axs)
        
        # 根据显示模式选择数据源和Y轴范围
        if config.PREPROCESSING_DISPLAY_MODE == 'processed':
            emg_source = signal_processor.processed_emg_buffer
            imu_source = signal_processor.processed_imu_buffer
            signal_label = "预处理"
            
            if signal_processor.normalize:
                emg_y_min_local = config.processed_emg_y_min
                emg_y_max_local = config.processed_emg_y_max
                imu_y_min_local = config.processed_imu_y_min
                imu_y_max_local = config.processed_imu_y_max
            else:
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
        
        # 检查是否需要完全重绘（标签或Y轴范围变化）
        need_full_redraw = (
            last_signal_label != signal_label or
            last_emg_y_range != (emg_y_min_local, emg_y_max_local) or
            last_imu_y_range != (imu_y_min_local, imu_y_max_local)
        )
        
        # 更新EMG数据
        if len(emg_source) > 0:
            recent_emg = list(emg_source)[-int(config.x_scale):]
            
            # 固定X轴范围，确保波形移动速度一致
            times = list(range(len(recent_emg)))
            
            # 固定X轴范围为[0, x_scale]，确保波形移动速度一致
            x_max = config.x_scale
            
            time_labels = [t * (1/config.EMG_SAMPLE_RATE) for t in range(x_max + 1)]
            emg_data = [d for _, d in recent_emg]
            emg_array = np.array(emg_data)
            
            for i in range(8):
                if i < len(lines):
                    # 使用set_data更新线条数据
                    lines[i].set_data(times, emg_array[:, i])
                    
                    # 更新Y轴范围和标签
                    axs[i].set_ylabel(f'EMG {i+1} ({signal_label})')
                    axs[i].set_ylim(emg_y_min_local, emg_y_max_local)
                    
                    # 固定X轴范围为[0, x_scale]
                    axs[i].set_xlim(0, x_max)
                    
                    # 更新X轴标签（固定间隔）
                    tick_positions = list(range(0, x_max + 1, max(1, x_max // 10)))
                    tick_labels = [f'{t * (1/config.EMG_SAMPLE_RATE):.1f}' for t in tick_positions]
                    axs[i].set_xticks(tick_positions)
                    axs[i].set_xticklabels(tick_labels)
        
        # 更新IMU数据
        if len(imu_source) > 0:
            _, latest_imu = imu_source[-1]
            
            # 更新柱状图
            for rect, h in zip(bar_container, latest_imu):
                rect.set_height(h)
            
            # 更新Y轴范围和标签
            axs[8].set_ylabel(f'IMU ({signal_label})')
            axs[8].set_ylim(imu_y_min_local, imu_y_max_local)
        
        # 重绘画布
        canvas.draw_idle()
        
        # 更新手势识别器
        try:
            from src.main_window import update_recognition_with_data
            if len(emg_source) > 0 and len(imu_source) > 0:
                # 获取最新的EMG和IMU数据
                _, latest_emg = emg_source[-1]
                _, latest_imu = imu_source[-1]
                update_recognition_with_data(latest_emg, latest_imu)
        except Exception as e:
            pass  # 识别器未初始化时忽略错误
        
        # 更新最后的状态记录
        last_signal_label = signal_label
        last_emg_y_range = (emg_y_min_local, emg_y_max_local)
        last_imu_y_range = (imu_y_min_local, imu_y_max_local)
        
        # 100ms后再次调用（降低更新频率以减少卡顿）
        root.after(100, lambda: update_plot(root, axs, canvas))
        
    except Exception as e:
        logger.error(f"更新图表失败: {e}")
        root.after(100, lambda: update_plot(root, axs, canvas))


def reset_plot():
    """重置图表状态"""
    global initialized, last_emg_data, last_imu_data
    global last_signal_label, last_emg_y_range, last_imu_y_range
    
    initialized = False
    last_emg_data = None
    last_imu_data = None
    last_signal_label = None
    last_emg_y_range = None
    last_imu_y_range = None
    
    logger.info("图表状态已重置")