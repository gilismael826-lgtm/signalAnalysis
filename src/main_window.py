#!/usr/bin/env python3
"""
主界面模块
整合所有模块，提供完整的GUI界面
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import src.config as config
from src.config import (
    x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max,
    SAVE_DATA, BAUDRATE, logger
)
from src.system_control_module import (
    start_acquisition, stop_acquisition, clear_buffers,
    export_config, analyze_data, view_documentation, system_settings
)
from src.zoom_control import (
    quick_zoom_in, quick_zoom_out, quick_reset_zoom,
    quick_emg_zoom_in, quick_emg_zoom_out,
    quick_imu_zoom_in, quick_imu_zoom_out, zoom_control
)
from src.plot_display import update_plot

# 全局变量
tree = None
axs = []
canvas = None


def create_main_window():
    """创建主窗口"""
    global root, port_var, save_var, status_var, status_label, start_btn, stop_btn
    global zoom_info_var, emg_zoom_info_var, imu_zoom_info_var, port_combobox, axs, canvas
    
    root = tk.Tk()
    root.title("EMG/IMU 系统控制中心")
    root.geometry("1400x900")
    root.minsize(1200, 700)
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    control_panel = ttk.Frame(main_frame, width=300)
    control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
    
    ttk.Label(control_panel, text="系统控制", font=("Arial", 14, "bold")).pack(pady=10)
    
    acquisition_frame = ttk.LabelFrame(control_panel, text="数据采集", padding="10")
    acquisition_frame.pack(fill=tk.X, pady=5)
    
    port_frame = ttk.Frame(acquisition_frame)
    port_frame.pack(fill=tk.X, pady=5)
    ttk.Label(port_frame, text="串口号:", width=10).pack(side=tk.LEFT)
    port_var = tk.StringVar()
    port_combobox = ttk.Combobox(port_frame, textvariable=port_var, width=20)
    port_combobox.pack(side=tk.LEFT, padx=5)
    
    refresh_btn = ttk.Button(port_frame, text="刷新", command=refresh_ports)
    refresh_btn.pack(side=tk.LEFT, padx=5)
    
    save_var = tk.BooleanVar(value=config.SAVE_DATA)
    save_check = ttk.Checkbutton(acquisition_frame, text="保存数据", variable=save_var)
    save_check.pack(anchor=tk.W, pady=5)
    
    button_frame = ttk.Frame(acquisition_frame)
    button_frame.pack(fill=tk.X, pady=5)
    
    start_btn = ttk.Button(button_frame, text="开始采集", 
                         command=lambda: start_acquisition(root, port_var, save_var, status_var, status_label, start_btn, stop_btn, lambda r: update_plot(r, axs, canvas)), 
                         width=12)
    start_btn.pack(side=tk.LEFT, padx=5)
    
    stop_btn = ttk.Button(button_frame, text="停止采集", 
                        command=lambda: stop_acquisition(status_var, status_label, start_btn, stop_btn), 
                        width=12, state="disabled")
    stop_btn.pack(side=tk.LEFT, padx=5)
    
    status_frame = ttk.Frame(acquisition_frame)
    status_frame.pack(fill=tk.X, pady=5)
    ttk.Label(status_frame, text="状态:", width=10).pack(side=tk.LEFT)
    status_var = tk.StringVar(value="已停止")
    status_label = ttk.Label(status_frame, textvariable=status_var, foreground="red", font=("Arial", 10, "bold"))
    status_label.pack(side=tk.LEFT, padx=5)
    
    quick_zoom_frame = ttk.LabelFrame(control_panel, text="快速缩放", padding="10")
    quick_zoom_frame.pack(fill=tk.X, pady=5)
    
    x_zoom_frame = ttk.Frame(quick_zoom_frame)
    x_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(x_zoom_frame, text="横轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(x_zoom_frame, text="放大", command=lambda: quick_zoom_in(zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(x_zoom_frame, text="缩小", command=lambda: quick_zoom_out(zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(x_zoom_frame, text="重置", command=lambda: quick_reset_zoom(zoom_info_var, emg_zoom_info_var, imu_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    
    zoom_info_var = tk.StringVar(value=f"当前: {config.x_scale} 点")
    ttk.Label(quick_zoom_frame, textvariable=zoom_info_var).pack(pady=2)
    
    emg_zoom_frame = ttk.Frame(quick_zoom_frame)
    emg_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(emg_zoom_frame, text="EMG纵轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(emg_zoom_frame, text="放大", command=lambda: quick_emg_zoom_in(emg_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(emg_zoom_frame, text="缩小", command=lambda: quick_emg_zoom_out(emg_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    
    emg_zoom_info_var = tk.StringVar(value=f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
    ttk.Label(quick_zoom_frame, textvariable=emg_zoom_info_var).pack(pady=2)
    
    imu_zoom_frame = ttk.Frame(quick_zoom_frame)
    imu_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(imu_zoom_frame, text="IMU纵轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(imu_zoom_frame, text="放大", command=lambda: quick_imu_zoom_in(imu_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(imu_zoom_frame, text="缩小", command=lambda: quick_imu_zoom_out(imu_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    
    imu_zoom_info_var = tk.StringVar(value=f"IMU: {config.imu_y_min}~{config.imu_y_max}")
    ttk.Label(quick_zoom_frame, textvariable=imu_zoom_info_var).pack(pady=2)
    
    advanced_frame = ttk.LabelFrame(control_panel, text="高级功能", padding="10")
    advanced_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(advanced_frame, text="清空数据缓冲区", command=clear_buffers, width=25).pack(pady=5)
    ttk.Button(advanced_frame, text="导出配置", command=export_config, width=25).pack(pady=5)
    ttk.Button(advanced_frame, text="波形缩放控制", 
               command=lambda: zoom_control(root, zoom_info_var, emg_zoom_info_var, imu_zoom_info_var), 
               width=25).pack(pady=5)
    
    analysis_frame = ttk.LabelFrame(control_panel, text="数据分析", padding="10")
    analysis_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(analysis_frame, text="分析数据文件", command=lambda: analyze_data(root), width=25).pack(pady=5)
    
    tools_frame = ttk.LabelFrame(control_panel, text="系统工具", padding="10")
    tools_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(tools_frame, text="查看文档", command=lambda: view_documentation(root), width=25).pack(pady=5)
    ttk.Button(tools_frame, text="系统设置", command=lambda: system_settings(root), width=25).pack(pady=5)
    
    plot_frame = ttk.LabelFrame(main_frame, text="实时数据", padding="10")
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
    
    create_plots(plot_frame)
    
    info_frame = ttk.Frame(root, padding="10")
    info_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(info_frame, text="EMG 通道: 8个 (单位: μV)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text="IMU 通道: 6个 (单位: rad/s, m/s²)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"波特率: {config.BAUDRATE}").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"横轴缩放: {config.x_scale} 点").pack(side=tk.LEFT, padx=10)
    
    refresh_ports()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    logger.info("系统控制中心已启动")
    return root


def create_plots(plot_frame):
    """创建图表"""
    global fig, canvas, axs
    
    fig = plt.Figure(figsize=(10, 8), dpi=100)
    axs = []
    
    for i in range(8):
        ax = fig.add_subplot(9, 1, i+1)
        ax.set_ylabel(f'EMG {i+1}')
        ax.set_ylim(config.emg_y_min, config.emg_y_max)
        ax.grid(True, alpha=0.3)
        axs.append(ax)
    
    ax = fig.add_subplot(9, 1, 9)
    ax.set_ylabel('IMU')
    ax.set_ylim(config.imu_y_min, config.imu_y_max)
    ax.grid(True, alpha=0.3)
    axs.append(ax)
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def refresh_ports():
    """刷新串口列表"""
    ports = list_serial_ports()
    port_combobox['values'] = ports
    if ports:
        config.SERIAL_PORT = ports[0]
    logger.info(f"刷新串口列表: {ports}")


def list_serial_ports():
    """列出所有可用的串口"""
    ports = []
    try:
        import serial.tools.list_ports
        ports = [port.device for port in serial.tools.list_ports.comports()]
    except ImportError:
        logger.error("pyserial未安装，无法列出串口")
    return ports


def on_closing():
    """窗口关闭处理"""
    if config.running:
        stop_acquisition(status_var, status_label, start_btn, stop_btn)
    root.destroy()
