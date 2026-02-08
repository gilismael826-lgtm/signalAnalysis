#!/usr/bin/env python3
"""
缩放控制模块
负责波形图的缩放控制
"""

import tkinter as tk
from tkinter import ttk, messagebox
import src.config as config
from src.config import (
    x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max,
    zoom_presets, logger
)


def apply_zoom_preset(preset_name, zoom_info_var=None, emg_zoom_info_var=None, imu_zoom_info_var=None):
    """应用缩放预设
    
    Args:
        preset_name: 预设名称
        zoom_info_var: 横轴缩放信息变量
        emg_zoom_info_var: EMG纵轴缩放信息变量
        imu_zoom_info_var: IMU纵轴缩放信息变量
    """
    if preset_name in zoom_presets:
        config.x_scale, config.emg_y_min, config.emg_y_max, config.imu_y_min, config.imu_y_max = zoom_presets[preset_name]
        if zoom_info_var:
            zoom_info_var.set(f"当前: {config.x_scale} 点")
        if emg_zoom_info_var:
            emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
        if imu_zoom_info_var:
            imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
        logger.info(f"应用缩放预设: {preset_name}")


def quick_zoom_in(zoom_info_var=None):
    """快速放大
    
    Args:
        zoom_info_var: 横轴缩放信息变量
    """
    new_scale = max(1, int(config.x_scale * 0.7))
    config.x_scale = new_scale
    if zoom_info_var:
        zoom_info_var.set(f"当前: {config.x_scale} 点")
    logger.info(f"快速放大: x_scale={config.x_scale}")


def quick_zoom_out(zoom_info_var=None):
    """快速缩小
    
    Args:
        zoom_info_var: 横轴缩放信息变量
    """
    new_scale = min(10000, int(config.x_scale * 1.5))
    config.x_scale = new_scale
    if zoom_info_var:
        zoom_info_var.set(f"当前: {config.x_scale} 点")
    logger.info(f"快速缩小: x_scale={config.x_scale}")


def quick_reset_zoom(zoom_info_var=None, emg_zoom_info_var=None, imu_zoom_info_var=None):
    """快速重置缩放
    
    Args:
        zoom_info_var: 横轴缩放信息变量
        emg_zoom_info_var: EMG纵轴缩放信息变量
        imu_zoom_info_var: IMU纵轴缩放信息变量
    """
    config.x_scale = 5000
    config.emg_y_min = -50000
    config.emg_y_max = 50000
    config.imu_y_min = -5
    config.imu_y_max = 5
    if zoom_info_var:
        zoom_info_var.set(f"当前: {config.x_scale} 点")
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
    logger.info("快速重置缩放为默认值")


def quick_emg_zoom_in(emg_zoom_info_var=None):
    """EMG纵轴快速放大
    
    Args:
        emg_zoom_info_var: EMG纵轴缩放信息变量
    """
    current_range = config.emg_y_max - config.emg_y_min
    new_range = max(100, int(current_range * 0.7))
    center = (config.emg_y_min + config.emg_y_max) / 2
    config.emg_y_min = int(center - new_range / 2)
    config.emg_y_max = int(center + new_range / 2)
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
    logger.info(f"EMG纵轴放大: {config.emg_y_min}~{config.emg_y_max}")


def quick_emg_zoom_out(emg_zoom_info_var=None):
    """EMG纵轴快速缩小
    
    Args:
        emg_zoom_info_var: EMG纵轴缩放信息变量
    """
    current_range = config.emg_y_max - config.emg_y_min
    new_range = min(1000000, int(current_range * 1.5))
    center = (config.emg_y_min + config.emg_y_max) / 2
    config.emg_y_min = int(center - new_range / 2)
    config.emg_y_max = int(center + new_range / 2)
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
    logger.info(f"EMG纵轴缩小: {config.emg_y_min}~{config.emg_y_max}")


def quick_imu_zoom_in(imu_zoom_info_var=None):
    """IMU纵轴快速放大
    
    Args:
        imu_zoom_info_var: IMU纵轴缩放信息变量
    """
    current_range = config.imu_y_max - config.imu_y_min
    new_range = max(0.5, current_range * 0.7)
    center = (config.imu_y_min + config.imu_y_max) / 2
    config.imu_y_min = round(center - new_range / 2, 2)
    config.imu_y_max = round(center + new_range / 2, 2)
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
    logger.info(f"IMU纵轴放大: {config.imu_y_min}~{config.imu_y_max}")


def quick_imu_zoom_out(imu_zoom_info_var=None):
    """IMU纵轴快速缩小
    
    Args:
        imu_zoom_info_var: IMU纵轴缩放信息变量
    """
    current_range = config.imu_y_max - config.imu_y_min
    new_range = min(200, current_range * 1.5)
    center = (config.imu_y_min + config.imu_y_max) / 2
    config.imu_y_min = round(center - new_range / 2, 2)
    config.imu_y_max = round(center + new_range / 2, 2)
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
    logger.info(f"IMU纵轴缩小: {config.imu_y_min}~{config.imu_y_max}")


def zoom_control(root, zoom_info_var=None, emg_zoom_info_var=None, imu_zoom_info_var=None):
    """缩放控制窗口
    
    Args:
        root: Tkinter根窗口对象
        zoom_info_var: 横轴缩放信息变量
        emg_zoom_info_var: EMG纵轴缩放信息变量
        imu_zoom_info_var: IMU纵轴缩放信息变量
    """
    zoom_window = tk.Toplevel(root)
    zoom_window.title("波形缩放控制")
    zoom_window.geometry("450x500")
    
    quick_frame = ttk.LabelFrame(zoom_window, text="快速缩放", padding="10")
    quick_frame.pack(fill=tk.X, padx=10, pady=5)
    
    quick_btn_frame = ttk.Frame(quick_frame)
    quick_btn_frame.pack(fill=tk.X)
    ttk.Button(quick_btn_frame, text="放大", command=lambda: quick_zoom_in(zoom_info_var), width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(quick_btn_frame, text="缩小", command=lambda: quick_zoom_out(zoom_info_var), width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(quick_btn_frame, text="重置", command=lambda: quick_reset_zoom(zoom_info_var, emg_zoom_info_var, imu_zoom_info_var), width=10).pack(side=tk.LEFT, padx=5)
    
    preset_frame = ttk.LabelFrame(zoom_window, text="预设配置", padding="10")
    preset_frame.pack(fill=tk.X, padx=10, pady=5)
    
    for preset_name in zoom_presets:
        ttk.Button(preset_frame, text=preset_name, command=lambda name=preset_name: apply_zoom_preset(name, zoom_info_var, emg_zoom_info_var, imu_zoom_info_var), width=20).pack(pady=2)
    
    x_scale_frame = ttk.LabelFrame(zoom_window, text="横轴缩放", padding="10")
    x_scale_frame.pack(fill=tk.X, padx=10, pady=5)
    
    x_scale_var = tk.StringVar(value=str(config.x_scale))
    ttk.Entry(x_scale_frame, textvariable=x_scale_var).pack(fill=tk.X)
    ttk.Label(x_scale_frame, text="显示数据点数量 (1-10000)").pack(anchor=tk.W, pady=2)
    
    emg_y_frame = ttk.LabelFrame(zoom_window, text="EMG纵轴范围", padding="10")
    emg_y_frame.pack(fill=tk.X, padx=10, pady=5)
    
    emg_y_min_var = tk.StringVar(value=str(config.emg_y_min))
    emg_y_max_var = tk.StringVar(value=str(config.emg_y_max))
    
    emg_y_frame_inner = ttk.Frame(emg_y_frame)
    emg_y_frame_inner.pack(fill=tk.X)
    
    ttk.Label(emg_y_frame_inner, text="最小值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(emg_y_frame_inner, textvariable=emg_y_min_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(emg_y_frame_inner, text="最大值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(emg_y_frame_inner, textvariable=emg_y_max_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(emg_y_frame, text="单位: μV").pack(anchor=tk.W, pady=2)
    
    imu_y_frame = ttk.LabelFrame(zoom_window, text="IMU纵轴范围", padding="10")
    imu_y_frame.pack(fill=tk.X, padx=10, pady=5)
    
    imu_y_min_var = tk.StringVar(value=str(config.imu_y_min))
    imu_y_max_var = tk.StringVar(value=str(config.imu_y_max))
    
    imu_y_frame_inner = ttk.Frame(imu_y_frame)
    imu_y_frame_inner.pack(fill=tk.X)
    
    ttk.Label(imu_y_frame_inner, text="最小值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(imu_y_frame_inner, textvariable=imu_y_min_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(imu_y_frame_inner, text="最大值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(imu_y_frame_inner, textvariable=imu_y_max_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(imu_y_frame, text="单位: rad/s, m/s²").pack(anchor=tk.W, pady=2)
    
    def save_zoom_settings():
        try:
            new_x_scale = int(x_scale_var.get())
            if new_x_scale < 1 or new_x_scale > 10000:
                raise ValueError("横轴缩放范围应在1-10000之间")
            
            new_emg_y_min = int(emg_y_min_var.get())
            new_emg_y_max = int(emg_y_max_var.get())
            if new_emg_y_min >= new_emg_y_max:
                raise ValueError("EMG最小值应小于最大值")
            
            new_imu_y_min = float(imu_y_min_var.get())
            new_imu_y_max = float(imu_y_max_var.get())
            if new_imu_y_min >= new_imu_y_max:
                raise ValueError("IMU最小值应小于最大值")
            
            config.x_scale = new_x_scale
            config.emg_y_min = new_emg_y_min
            config.emg_y_max = new_emg_y_max
            config.imu_y_min = new_imu_y_min
            config.imu_y_max = new_imu_y_max
            
            if zoom_info_var:
                zoom_info_var.set(f"当前: {config.x_scale} 点")
            if emg_zoom_info_var:
                emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
            if imu_zoom_info_var:
                imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
            
            zoom_window.destroy()
            logger.info(f"保存缩放设置: x_scale={config.x_scale}, emg_y=({config.emg_y_min},{config.emg_y_max}), imu_y=({config.imu_y_min},{config.imu_y_max})")
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")
    
    def reset_zoom_settings():
        config.x_scale = 3000
        config.emg_y_min = -2000
        config.emg_y_max = 2000
        config.imu_y_min = -5
        config.imu_y_max = 5
        
        x_scale_var.set(str(config.x_scale))
        emg_y_min_var.set(str(config.emg_y_min))
        emg_y_max_var.set(str(config.emg_y_max))
        imu_y_min_var.set(str(config.imu_y_min))
        imu_y_max_var.set(str(config.imu_y_max))
        
        if zoom_info_var:
            zoom_info_var.set(f"当前: {config.x_scale} 点")
        if emg_zoom_info_var:
            emg_zoom_info_var.set(f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
        if imu_zoom_info_var:
            imu_zoom_info_var.set(f"IMU: {config.imu_y_min}~{config.imu_y_max}")
        
        logger.info("重置缩放设置为默认值")
    
    button_frame = ttk.Frame(zoom_window)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(button_frame, text="重置", command=reset_zoom_settings, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="保存", command=save_zoom_settings, width=10).pack(side=tk.RIGHT, padx=5)
    
    logger.info("打开缩放控制窗口")
