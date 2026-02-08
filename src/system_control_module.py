#!/usr/bin/env python3
"""
系统控制模块
负责系统核心功能
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import json
import threading
import src.config as config
from src.config import running, logger
from src.serial_comm import serial_reader
from src.data_manager import save_data_point, init_data_files


def start_acquisition(root, port_var, save_var, status_var, status_label, start_btn, stop_btn, update_plot_func):
    """开始数据采集
    
    Args:
        root: Tkinter根窗口对象
        port_var: 串口号变量
        save_var: 保存数据变量
        status_var: 状态变量
        status_label: 状态标签
        start_btn: 开始按钮
        stop_btn: 停止按钮
        update_plot_func: 更新图表函数
    """
    global serial_thread
    
    if config.running:
        messagebox.showinfo("信息", "数据采集已经在运行中")
        return
    
    port = port_var.get()
    if not port:
        messagebox.showerror("错误", "请选择串口号")
        return
    
    config.SERIAL_PORT = port
    config.SAVE_DATA = save_var.get()
    
    if config.SAVE_DATA:
        if not init_data_files():
            messagebox.showerror("错误", "创建数据文件失败")
            return
    
    config.running = True
    serial_thread = threading.Thread(target=serial_reader, daemon=True)
    serial_thread.start()
    
    update_plot_func(root)
    
    status_var.set("运行中")
    status_label.config(foreground="green")
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")
    logger.info("开始数据采集")


def stop_acquisition(status_var, status_label, start_btn, stop_btn):
    """停止数据采集
    
    Args:
        status_var: 状态变量
        status_label: 状态标签
        start_btn: 开始按钮
        stop_btn: 停止按钮
    """
    config.running = False
    status_var.set("已停止")
    status_label.config(foreground="red")
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")
    logger.info("停止数据采集")


def clear_buffers():
    """清空数据缓冲区"""
    config.emg_buffer.clear()
    config.imu_buffer.clear()
    logger.info("清空数据缓冲区")
    messagebox.showinfo("信息", "数据缓冲区已清空")


def export_config():
    """导出配置"""
    config_data = {
        "serial_port": config.SERIAL_PORT,
        "baudrate": config.BAUDRATE,
        "save_data": config.SAVE_DATA,
        "x_scale": config.x_scale,
        "emg_y_min": config.emg_y_min,
        "emg_y_max": config.emg_y_max,
        "imu_y_min": config.imu_y_min,
        "imu_y_max": config.imu_y_max
    }
    
    with open("config_export.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    logger.info("导出配置到 config_export.json")
    messagebox.showinfo("信息", "配置已导出到 config_export.json")


def analyze_data(root):
    """分析数据文件
    
    Args:
        root: Tkinter根窗口对象
    """
    analysis_window = tk.Toplevel(root)
    analysis_window.title("数据分析")
    analysis_window.geometry("400x300")
    
    ttk.Label(analysis_window, text="数据分析功能", font=("Arial", 14, "bold")).pack(pady=20)
    ttk.Label(analysis_window, text="1. 数据文件存储在 data/ 目录") .pack(anchor=tk.W, padx=20, pady=5)
    ttk.Label(analysis_window, text="2. 支持 CSV 格式的数据文件") .pack(anchor=tk.W, padx=20, pady=5)
    ttk.Label(analysis_window, text="3. 可使用 Excel 或 Python 进行分析") .pack(anchor=tk.W, padx=20, pady=5)
    
    ttk.Button(analysis_window, text="打开数据目录", 
               command=lambda: subprocess.run(["explorer", os.path.join(os.getcwd(), "data")])).pack(pady=20)
    
    logger.info("打开数据分析窗口")


def view_documentation(root):
    """查看文档
    
    Args:
        root: Tkinter根窗口对象
    """
    doc_window = tk.Toplevel(root)
    doc_window.title("系统文档")
    doc_window.geometry("500x400")
    
    ttk.Label(doc_window, text="系统文档", font=("Arial", 14, "bold")).pack(pady=20)
    
    doc_text = """EMG/IMU 系统控制中心使用说明：

1. 基本操作：
   - 选择串口号
   - 勾选保存数据选项
   - 点击开始采集按钮

2. 数据显示：
   - 上方8个图表为EMG通道数据
   - 下方图表为IMU通道数据

3. 缩放控制：
   - 快速缩放：放大、缩小、重置
   - 波形缩放控制：详细的缩放设置

4. 高级功能：
   - 清空数据缓冲区
   - 导出配置
   - 分析数据文件

5. 系统设置：
   - 调整波特率
   - 配置数据存储

6. 故障排除：
   - 串口连接失败：检查设备是否正确连接
   - 数据不显示：检查波特率设置是否正确
   - 保存失败：检查磁盘空间是否充足
"""
    
    text_widget = tk.Text(doc_window, wrap=tk.WORD, padding=10)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=20)
    text_widget.insert(tk.END, doc_text)
    text_widget.config(state=tk.DISABLED)
    
    logger.info("打开系统文档窗口")


def system_settings(root):
    """系统设置
    
    Args:
        root: Tkinter根窗口对象
    """
    settings_window = tk.Toplevel(root)
    settings_window.title("系统设置")
    settings_window.geometry("400x300")
    
    ttk.Label(settings_window, text="系统设置", font=("Arial", 14, "bold")).pack(pady=20)
    
    baudrate_frame = ttk.LabelFrame(settings_window, text="串口设置", padding="10")
    baudrate_frame.pack(fill=tk.X, padx=20, pady=10)
    
    baudrate_var = tk.StringVar(value=str(config.BAUDRATE))
    baudrate_options = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
    
    ttk.Label(baudrate_frame, text="波特率:").pack(anchor=tk.W, pady=5)
    ttk.Combobox(baudrate_frame, textvariable=baudrate_var, values=baudrate_options).pack(fill=tk.X, pady=5)
    
    data_frame = ttk.LabelFrame(settings_window, text="数据存储", padding="10")
    data_frame.pack(fill=tk.X, padx=20, pady=10)
    
    ttk.Label(data_frame, text="数据目录: " + config.DATA_DIR).pack(anchor=tk.W, pady=5)
    ttk.Label(data_frame, text="最大文件大小: " + str(config.MAX_FILE_SIZE // 1024 // 1024) + " MB").pack(anchor=tk.W, pady=5)
    
    def save_settings():
        """保存设置"""
        try:
            config.BAUDRATE = int(baudrate_var.get())
            logger.info(f"保存系统设置: 波特率={config.BAUDRATE}")
            messagebox.showinfo("信息", "设置已保存")
            settings_window.destroy()
        except ValueError as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")
    
    ttk.Button(settings_window, text="保存", command=save_settings, width=15).pack(pady=20)
    
    logger.info("打开系统设置窗口")
