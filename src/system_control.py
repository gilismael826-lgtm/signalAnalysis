#!/usr/bin/env python3
"""
系统控制中心 - 交互式窗口
提供完整的系统控制功能，包括数据采集、数据分析等
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import sys
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from collections import deque
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
SERIAL_PORT = "COM5"
BAUDRATE = 921600
SAVE_DATA = True
DATA_DIR = "data"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# 数据缓冲区
emg_buffer = deque(maxlen=10000)
imu_buffer = deque(maxlen=10000)

# 文件句柄
emg_data_file = None
imu_data_file = None
data_file_size = 0

# 串口对象
ser = None
serial_thread = None
running = False

# 图表对象
fig = None
canvas = None
axs = []
plot_emg = []
plot_imu = []
zoom_info_var = None
emg_zoom_info_var = None
imu_zoom_info_var = None

# 缩放参数
x_scale = 3000  # 横轴显示数据点数量
emg_y_min = -2000
emg_y_max = 2000
imu_y_min = -5
imu_y_max = 5

# 缩放预设
zoom_presets = {
    "默认": (3000, -2000, 2000, -5, 5),
    "详细分析": (50, -1000, 1000, -2, 2),
    "长时观察": (300, -3000, 3000, -10, 10),
    "微小信号": (100, -500, 500, -1, 1),
    "强信号": (100, -5000, 5000, -15, 15)
}

# ========== 辅助函数 ==========
def list_serial_ports():
    """列出所有可用的串口"""
    ports = []
    try:
        import serial.tools.list_ports
        ports = [port.device for port in serial.tools.list_ports.comports()]
    except ImportError:
        logger.error("pyserial未安装，无法列出串口")
    return ports

def init_data_files():
    """初始化数据文件"""
    global emg_data_file, imu_data_file, data_file_size
    
    if not SAVE_DATA:
        return
        
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emg_filename = os.path.join(DATA_DIR, f"emg_{timestamp}.csv")
    imu_filename = os.path.join(DATA_DIR, f"imu_{timestamp}.csv")
    
    try:
        emg_data_file = open(emg_filename, 'w')
        imu_data_file = open(imu_filename, 'w')
        
        # 写入CSV文件头
        emg_header = "timestamp,ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8\n"
        imu_header = "timestamp,gx,gy,gz,ax,ay,az\n"
        emg_data_file.write(emg_header)
        imu_data_file.write(imu_header)
        
        data_file_size = 0
        logger.info(f"数据文件已创建: {emg_filename}, {imu_filename}")
        return True
    except Exception as e:
        logger.error(f"创建数据文件失败: {e}")
        return False

def save_data_point(data_type, timestamp, data):
    """保存单个数据点"""
    global data_file_size
    
    if not SAVE_DATA:
        return
    
    # 检查文件大小，达到上限时轮换文件
    if data_file_size > MAX_FILE_SIZE:
        rotate_data_files()
    
    try:
        if data_type == 'EMG' and emg_data_file:
            csv_line = f"{timestamp},{','.join(map(str, data))}\n"
            emg_data_file.write(csv_line)
            data_file_size += len(csv_line)
            
        elif data_type == 'IMU' and imu_data_file:
            csv_line = f"{timestamp},{','.join(map(str, data))}\n"
            imu_data_file.write(csv_line)
            data_file_size += len(csv_line)
            
    except Exception as e:
        logger.error(f"保存数据失败: {e}")

def rotate_data_files():
    """轮换数据文件（循环存储）"""
    global emg_data_file, imu_data_file, data_file_size
    
    try:
        # 关闭当前文件
        close_data_files()
        
        # 删除旧的数据文件
        delete_old_data_files()
        
        # 初始化新文件
        init_data_files()
        
        logger.info("数据文件已轮换，开始新的文件记录")
        
    except Exception as e:
        logger.error(f"轮换数据文件失败: {e}")

def delete_old_data_files():
    """删除旧的数据文件"""
    try:
        if not os.path.exists(DATA_DIR):
            return
        
        # 获取所有数据文件
        emg_files = [f for f in os.listdir(DATA_DIR) if f.startswith('emg_') and f.endswith('.csv')]
        imu_files = [f for f in os.listdir(DATA_DIR) if f.startswith('imu_') and f.endswith('.csv')]
        
        # 按修改时间排序，删除最旧的文件
        emg_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)))
        imu_files.sort(key=lambda x: os.path.getmtime(os.path.join(DATA_DIR, x)))
        
        # 删除最旧的文件
        if emg_files:
            oldest_emg = os.path.join(DATA_DIR, emg_files[0])
            os.remove(oldest_emg)
            logger.info(f"删除旧的EMG文件: {oldest_emg}")
        
        if imu_files:
            oldest_imu = os.path.join(DATA_DIR, imu_files[0])
            os.remove(oldest_imu)
            logger.info(f"删除旧的IMU文件: {oldest_imu}")
            
    except Exception as e:
        logger.error(f"删除旧数据文件失败: {e}")

def close_data_files():
    """关闭数据文件"""
    try:
        global emg_data_file, imu_data_file
        if emg_data_file:
            emg_data_file.close()
            emg_data_file = None
        if imu_data_file:
            imu_data_file.close()
            imu_data_file = None
        logger.info("数据文件已关闭")
    except Exception as e:
        logger.error(f"关闭数据文件失败: {e}")

def parse_packet(data):
    """解析数据包"""
    if len(data) < 29:
        return None, None, None
    
    if data[0:3] != b'\xd2\xd2\xd2':
        return None, None, None
    
    pkt_type = data[3]
    ts = time.time()
    
    if pkt_type == 0xAA:
        emg = []
        for i in range(8):
            b0, b1, b2 = data[5 + i*3 : 8 + i*3]
            val = (b0 << 16) | (b1 << 8) | b2
            if val >= 0x800000:
                val -= 0x1000000
            emg.append(val)
        return 'EMG', ts, emg
    
    elif pkt_type == 0xBB:
        raw = []
        for i in range(6):
            lo = data[7 + i*2]
            hi = data[8 + i*2]
            val = (hi << 8) | lo
            if val >= 0x8000:
                val -= 0x10000
            raw.append(val)
        gyro = [0.0012 * x for x in raw[:3]]
        accel = [0.0005978 * x for x in raw[3:]]
        imu = gyro + accel
        return 'IMU', ts, imu
    
    return None, None, None

# ========== 串口接收线程 ==========
def serial_reader():
    """串口数据接收线程"""
    global running, ser
    
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        logger.info(f"✅ 已连接 {SERIAL_PORT}，波特率 {BAUDRATE}")
        
        buffer = b''
        packet_count = 0
        error_count = 0
        
        while running:
            try:
                data = ser.read(1000)
                if data:
                    buffer += data
                    
                    while len(buffer) >= 29:
                        start = buffer.find(b'\xd2\xd2\xd2')
                        if start == -1:
                            buffer = b''
                            error_count += 1
                            break
                        
                        if start + 29 <= len(buffer):
                            packet = buffer[start:start+29]
                            buffer = buffer[start+29:]
                            
                            pkt_type, ts, data = parse_packet(packet)
                            if pkt_type:
                                packet_count += 1
                                
                                if pkt_type == 'EMG':
                                    emg_buffer.append((ts, data))
                                    save_data_point('EMG', ts, data)
                                elif pkt_type == 'IMU':
                                    imu_buffer.append((ts, data))
                                    save_data_point('IMU', ts, data)
                            else:
                                error_count += 1
            except Exception as e:
                logger.error(f"读取串口数据失败: {e}")
                time.sleep(0.1)
        
        ser.close()
        logger.info("串口已关闭")
        
    except Exception as e:
        logger.error(f"串口连接失败: {e}")
        running = False

# ========== 绘图更新函数 ==========
def update_plot():
    """更新图表"""
    if not running:
        return
    
    try:
        for ax in axs:
            ax.clear()
        
        if len(emg_buffer) > 0:
            recent_emg = list(emg_buffer)[-int(x_scale):]
            times = [t for t, _ in recent_emg]
            emg_data = [d for _, d in recent_emg]
            
            emg_array = np.array(emg_data)
            colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange']
            for i in range(8):
                axs[i].plot(times, emg_array[:, i], color=colors[i], linewidth=1)
                axs[i].set_ylabel(f'EMG {i+1} (μV)')
                axs[i].set_ylim(emg_y_min, emg_y_max)
                axs[i].grid(True, alpha=0.3)
        
        if len(imu_buffer) > 0:
            _, latest_imu = imu_buffer[-1]
            axs[8].bar(['gx', 'gy', 'gz', 'ax', 'ay', 'az'], latest_imu)
            axs[8].set_ylabel('IMU Values')
            axs[8].set_ylim(imu_y_min, imu_y_max)
            axs[8].grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas.draw()
        
        root.after(50, update_plot)
        
    except Exception as e:
        logger.error(f"更新图表失败: {e}")
        root.after(50, update_plot)

# ========== 系统控制函数 ==========
def start_acquisition():
    """开始数据采集"""
    global running, serial_thread, SERIAL_PORT
    
    if running:
        messagebox.showinfo("信息", "数据采集已经在运行中")
        return
    
    port = port_var.get()
    if not port:
        messagebox.showerror("错误", "请选择串口号")
        return
    
    SERIAL_PORT = port
    
    if save_var.get():
        if not init_data_files():
            messagebox.showerror("错误", "创建数据文件失败")
            return
    
    running = True
    serial_thread = threading.Thread(target=serial_reader, daemon=True)
    serial_thread.start()
    
    update_plot()
    
    status_var.set("运行中")
    status_label.config(foreground="green")
    start_btn.config(state="disabled")
    stop_btn.config(state="normal")
    logger.info("开始数据采集")

def stop_acquisition():
    """停止数据采集"""
    global running
    
    if not running:
        messagebox.showinfo("信息", "数据采集已经停止")
        return
    
    running = False
    
    if serial_thread:
        serial_thread.join(timeout=2)
    
    close_data_files()
    
    status_var.set("已停止")
    status_label.config(foreground="red")
    start_btn.config(state="normal")
    stop_btn.config(state="disabled")
    logger.info("停止数据采集")

def clear_buffers():
    """清空数据缓冲区"""
    global emg_buffer, imu_buffer
    emg_buffer.clear()
    imu_buffer.clear()
    messagebox.showinfo("信息", "数据缓冲区已清空")
    logger.info("清空数据缓冲区")

def export_config():
    """导出配置"""
    try:
        import json
        config_data = {
            "serial_port": SERIAL_PORT,
            "baudrate": BAUDRATE,
            "data_dir": DATA_DIR,
            "save_data": SAVE_DATA,
            "x_scale": x_scale,
            "emg_y_min": emg_y_min,
            "emg_y_max": emg_y_max,
            "imu_y_min": imu_y_min,
            "imu_y_max": imu_y_max
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="config.json"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("信息", "配置已导出")
            logger.info(f"导出配置到: {file_path}")
    except Exception as e:
        messagebox.showerror("错误", f"导出配置失败: {e}")
        logger.error(f"导出配置失败: {e}")

def analyze_data():
    """数据分析功能"""
    try:
        # 选择数据文件
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 创建分析窗口
        analysis_window = tk.Toplevel(root)
        analysis_window.title("数据分析")
        analysis_window.geometry("800x600")
        
        # 显示基本信息
        info_frame = ttk.LabelFrame(analysis_window, text="数据信息", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text=f"文件: {os.path.basename(file_path)}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"行数: {len(df)}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"列数: {len(df.columns)}").pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"列名: {', '.join(df.columns)}").pack(anchor=tk.W)
        
        # 显示统计信息
        stats_frame = ttk.LabelFrame(analysis_window, text="统计信息", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 计算统计信息
        stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=20)
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        stats_info = df.describe()
        stats_text.insert(tk.END, str(stats_info))
        
        # 添加关闭按钮
        ttk.Button(analysis_window, text="关闭", command=analysis_window.destroy).pack(pady=10)
        
        logger.info(f"分析数据文件: {file_path}")
        
    except Exception as e:
        messagebox.showerror("错误", f"数据分析失败: {e}")
        logger.error(f"数据分析失败: {e}")

def view_documentation():
    """查看文档"""
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        messagebox.showerror("错误", "文档目录不存在")
        return
    
    # 创建文档窗口
    doc_window = tk.Toplevel(root)
    doc_window.title("系统文档")
    doc_window.geometry("600x400")
    
    # 文档列表
    listbox_frame = ttk.Frame(doc_window)
    listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    listbox = tk.Listbox(listbox_frame)
    listbox.pack(fill=tk.BOTH, expand=True)
    
    # 添加文档文件
    for filename in os.listdir(docs_dir):
        if filename.endswith('.md'):
            listbox.insert(tk.END, filename)
    
    # 打开文档函数
    def open_doc():
        selection = listbox.curselection()
        if selection:
            doc_file = os.path.join(docs_dir, listbox.get(selection[0]))
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 创建文档内容窗口
                content_window = tk.Toplevel(root)
                content_window.title(f"文档: {listbox.get(selection[0])}")
                content_window.geometry("800x600")
                
                text_widget = tk.Text(content_window, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                text_widget.insert(tk.END, content)
                text_widget.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("错误", f"打开文档失败: {e}")
    
    ttk.Button(listbox_frame, text="打开", command=open_doc).pack(pady=5)
    logger.info("打开文档窗口")

def system_settings():
    """系统设置"""
    settings_window = tk.Toplevel(root)
    settings_window.title("系统设置")
    settings_window.geometry("400x300")
    
    # 波特率设置
    baudrate_frame = ttk.LabelFrame(settings_window, text="波特率", padding="10")
    baudrate_frame.pack(fill=tk.X, padx=10, pady=5)
    
    baudrate_var = tk.StringVar(value=str(BAUDRATE))
    ttk.Entry(baudrate_frame, textvariable=baudrate_var).pack(fill=tk.X)
    
    # 数据目录设置
    datadir_frame = ttk.LabelFrame(settings_window, text="数据目录", padding="10")
    datadir_frame.pack(fill=tk.X, padx=10, pady=5)
    
    datadir_var = tk.StringVar(value=DATA_DIR)
    ttk.Entry(datadir_frame, textvariable=datadir_var).pack(fill=tk.X)
    
    # 保存设置
    def save_settings():
        global BAUDRATE, DATA_DIR
        try:
            BAUDRATE = int(baudrate_var.get())
            DATA_DIR = datadir_var.get()
            messagebox.showinfo("信息", "设置已保存")
            settings_window.destroy()
            logger.info(f"保存设置: 波特率={BAUDRATE}, 数据目录={DATA_DIR}")
        except ValueError:
            messagebox.showerror("错误", "波特率必须是数字")
    
    ttk.Button(settings_window, text="保存", command=save_settings).pack(pady=10)
    logger.info("打开系统设置")

def apply_zoom_preset(preset_name):
    """应用缩放预设"""
    global x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max
    if preset_name in zoom_presets:
        x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max = zoom_presets[preset_name]
        if zoom_info_var:
            zoom_info_var.set(f"当前: {x_scale} 点")
        if emg_zoom_info_var:
            emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
        if imu_zoom_info_var:
            imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
        logger.info(f"应用缩放预设: {preset_name}, x_scale={x_scale}, emg_y=({emg_y_min},{emg_y_max}), imu_y=({imu_y_min},{imu_y_max})")

def quick_zoom_in():
    """快速放大"""
    global x_scale
    new_scale = max(1, int(x_scale * 0.7))
    x_scale = new_scale
    if zoom_info_var:
        zoom_info_var.set(f"当前: {x_scale} 点")
    logger.info(f"快速放大: x_scale={x_scale}")

def quick_zoom_out():
    """快速缩小"""
    global x_scale
    new_scale = min(10000, int(x_scale * 1.5))
    x_scale = new_scale
    if zoom_info_var:
        zoom_info_var.set(f"当前: {x_scale} 点")
    logger.info(f"快速缩小: x_scale={x_scale}")

def quick_reset_zoom():
    """快速重置缩放"""
    global x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max
    x_scale = 3000
    emg_y_min = -2000
    emg_y_max = 2000
    imu_y_min = -5
    imu_y_max = 5
    if zoom_info_var:
        zoom_info_var.set(f"当前: {x_scale} 点")
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
    logger.info("快速重置缩放为默认值")

def quick_emg_zoom_in():
    """EMG纵轴快速放大"""
    global emg_y_min, emg_y_max
    current_range = emg_y_max - emg_y_min
    new_range = max(100, int(current_range * 0.7))
    center = (emg_y_min + emg_y_max) / 2
    emg_y_min = int(center - new_range / 2)
    emg_y_max = int(center + new_range / 2)
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
    logger.info(f"EMG纵轴放大: {emg_y_min}~{emg_y_max}")

def quick_emg_zoom_out():
    """EMG纵轴快速缩小"""
    global emg_y_min, emg_y_max
    current_range = emg_y_max - emg_y_min
    new_range = min(400000, int(current_range * 1.5))
    center = (emg_y_min + emg_y_max) / 2
    emg_y_min = int(center - new_range / 2)
    emg_y_max = int(center + new_range / 2)
    if emg_zoom_info_var:
        emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
    logger.info(f"EMG纵轴缩小: {emg_y_min}~{emg_y_max}")

def quick_imu_zoom_in():
    """IMU纵轴快速放大"""
    global imu_y_min, imu_y_max
    current_range = imu_y_max - imu_y_min
    new_range = max(0.5, current_range * 0.7)
    center = (imu_y_min + imu_y_max) / 2
    imu_y_min = round(center - new_range / 2, 2)
    imu_y_max = round(center + new_range / 2, 2)
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
    logger.info(f"IMU纵轴放大: {imu_y_min}~{imu_y_max}")

def quick_imu_zoom_out():
    """IMU纵轴快速缩小"""
    global imu_y_min, imu_y_max
    current_range = imu_y_max - imu_y_min
    new_range = min(200, current_range * 1.5)
    center = (imu_y_min + imu_y_max) / 2
    imu_y_min = round(center - new_range / 2, 2)
    imu_y_max = round(center + new_range / 2, 2)
    if imu_zoom_info_var:
        imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
    logger.info(f"IMU纵轴缩小: {imu_y_min}~{imu_y_max}")

def zoom_control():
    """缩放控制"""
    zoom_window = tk.Toplevel(root)
    zoom_window.title("波形缩放控制")
    zoom_window.geometry("450x500")
    
    # 快速缩放按钮
    quick_frame = ttk.LabelFrame(zoom_window, text="快速缩放", padding="10")
    quick_frame.pack(fill=tk.X, padx=10, pady=5)
    
    quick_btn_frame = ttk.Frame(quick_frame)
    quick_btn_frame.pack(fill=tk.X)
    ttk.Button(quick_btn_frame, text="放大", command=quick_zoom_in, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(quick_btn_frame, text="缩小", command=quick_zoom_out, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(quick_btn_frame, text="重置", command=quick_reset_zoom, width=10).pack(side=tk.LEFT, padx=5)
    
    # 预设配置
    preset_frame = ttk.LabelFrame(zoom_window, text="预设配置", padding="10")
    preset_frame.pack(fill=tk.X, padx=10, pady=5)
    
    for preset_name in zoom_presets:
        ttk.Button(preset_frame, text=preset_name, command=lambda name=preset_name: apply_zoom_preset(name), width=20).pack(pady=2)
    
    # 横轴缩放设置
    x_scale_frame = ttk.LabelFrame(zoom_window, text="横轴缩放", padding="10")
    x_scale_frame.pack(fill=tk.X, padx=10, pady=5)
    
    x_scale_var = tk.StringVar(value=str(x_scale))
    ttk.Entry(x_scale_frame, textvariable=x_scale_var).pack(fill=tk.X)
    ttk.Label(x_scale_frame, text="显示数据点数量 (1-10000)").pack(anchor=tk.W, pady=2)
    
    # EMG纵轴设置
    emg_y_frame = ttk.LabelFrame(zoom_window, text="EMG纵轴范围", padding="10")
    emg_y_frame.pack(fill=tk.X, padx=10, pady=5)
    
    emg_y_min_var = tk.StringVar(value=str(emg_y_min))
    emg_y_max_var = tk.StringVar(value=str(emg_y_max))
    
    emg_y_frame_inner = ttk.Frame(emg_y_frame)
    emg_y_frame_inner.pack(fill=tk.X)
    
    ttk.Label(emg_y_frame_inner, text="最小值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(emg_y_frame_inner, textvariable=emg_y_min_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(emg_y_frame_inner, text="最大值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(emg_y_frame_inner, textvariable=emg_y_max_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(emg_y_frame, text="单位: μV").pack(anchor=tk.W, pady=2)
    
    # IMU纵轴设置
    imu_y_frame = ttk.LabelFrame(zoom_window, text="IMU纵轴范围", padding="10")
    imu_y_frame.pack(fill=tk.X, padx=10, pady=5)
    
    imu_y_min_var = tk.StringVar(value=str(imu_y_min))
    imu_y_max_var = tk.StringVar(value=str(imu_y_max))
    
    imu_y_frame_inner = ttk.Frame(imu_y_frame)
    imu_y_frame_inner.pack(fill=tk.X)
    
    ttk.Label(imu_y_frame_inner, text="最小值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(imu_y_frame_inner, textvariable=imu_y_min_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(imu_y_frame_inner, text="最大值: ", width=10).pack(side=tk.LEFT)
    ttk.Entry(imu_y_frame_inner, textvariable=imu_y_max_var, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Label(imu_y_frame, text="单位: rad/s, m/s²").pack(anchor=tk.W, pady=2)
    
    # 保存缩放设置
    def save_zoom_settings():
        global x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max
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
            
            x_scale = new_x_scale
            emg_y_min = new_emg_y_min
            emg_y_max = new_emg_y_max
            imu_y_min = new_imu_y_min
            imu_y_max = new_imu_y_max
            
            if zoom_info_var:
                zoom_info_var.set(f"当前: {x_scale} 点")
            if emg_zoom_info_var:
                emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
            if imu_zoom_info_var:
                imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
            
            zoom_window.destroy()
            logger.info(f"保存缩放设置: x_scale={x_scale}, emg_y=({emg_y_min},{emg_y_max}), imu_y=({imu_y_min},{imu_y_max})")
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {e}")
    
    # 重置缩放设置
    def reset_zoom_settings():
        global x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max
        x_scale = 3000
        emg_y_min = -2000
        emg_y_max = 2000
        imu_y_min = -5
        imu_y_max = 5
        
        x_scale_var.set(str(x_scale))
        emg_y_min_var.set(str(emg_y_min))
        emg_y_max_var.set(str(emg_y_max))
        imu_y_min_var.set(str(imu_y_min))
        imu_y_max_var.set(str(imu_y_max))
        
        if zoom_info_var:
            zoom_info_var.set(f"当前: {x_scale} 点")
        if emg_zoom_info_var:
            emg_zoom_info_var.set(f"EMG: {emg_y_min}~{emg_y_max} μV")
        if imu_zoom_info_var:
            imu_zoom_info_var.set(f"IMU: {imu_y_min}~{imu_y_max}")
        
        logger.info("重置缩放设置为默认值")
    
    # 按钮
    button_frame = ttk.Frame(zoom_window)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    ttk.Button(button_frame, text="重置", command=reset_zoom_settings, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="保存", command=save_zoom_settings, width=10).pack(side=tk.RIGHT, padx=5)
    
    logger.info("打开缩放控制窗口")

def refresh_ports():
    """刷新串口列表"""
    ports = list_serial_ports()
    port_combobox['values'] = ports
    if ports:
        port_var.set(ports[0])
    logger.info(f"刷新串口列表: {ports}")

def on_closing():
    """窗口关闭处理"""
    if running:
        stop_acquisition()
    root.destroy()

# ========== 主窗口创建 ==========
if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    root.title("EMG/IMU 系统控制中心")
    root.geometry("1400x900")
    root.minsize(1200, 700)
    
    # 创建主框架
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 创建左侧控制面板
    control_panel = ttk.Frame(main_frame, width=300)
    control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
    
    # 系统控制标题
    ttk.Label(control_panel, text="系统控制", font=("Arial", 14, "bold")).pack(pady=10)
    
    # 数据采集控制
    acquisition_frame = ttk.LabelFrame(control_panel, text="数据采集", padding="10")
    acquisition_frame.pack(fill=tk.X, pady=5)
    
    # 串口选择
    port_frame = ttk.Frame(acquisition_frame)
    port_frame.pack(fill=tk.X, pady=5)
    ttk.Label(port_frame, text="串口号:", width=10).pack(side=tk.LEFT)
    port_var = tk.StringVar()
    port_combobox = ttk.Combobox(port_frame, textvariable=port_var, width=20)
    port_combobox.pack(side=tk.LEFT, padx=5)
    
    refresh_btn = ttk.Button(port_frame, text="刷新", command=refresh_ports)
    refresh_btn.pack(side=tk.LEFT, padx=5)
    
    # 数据保存选项
    save_var = tk.BooleanVar(value=SAVE_DATA)
    save_check = ttk.Checkbutton(acquisition_frame, text="保存数据", variable=save_var)
    save_check.pack(anchor=tk.W, pady=5)
    
    # 控制按钮
    button_frame = ttk.Frame(acquisition_frame)
    button_frame.pack(fill=tk.X, pady=5)
    
    start_btn = ttk.Button(button_frame, text="开始采集", command=start_acquisition, width=12)
    start_btn.pack(side=tk.LEFT, padx=5)
    
    stop_btn = ttk.Button(button_frame, text="停止采集", command=stop_acquisition, width=12, state="disabled")
    stop_btn.pack(side=tk.LEFT, padx=5)
    
    # 状态显示
    status_frame = ttk.Frame(acquisition_frame)
    status_frame.pack(fill=tk.X, pady=5)
    ttk.Label(status_frame, text="状态:", width=10).pack(side=tk.LEFT)
    status_var = tk.StringVar(value="已停止")
    status_label = ttk.Label(status_frame, textvariable=status_var, foreground="red", font=("Arial", 10, "bold"))
    status_label.pack(side=tk.LEFT, padx=5)
    
    # 快速缩放控制
    quick_zoom_frame = ttk.LabelFrame(control_panel, text="快速缩放", padding="10")
    quick_zoom_frame.pack(fill=tk.X, pady=5)
    
    # 横轴缩放
    x_zoom_frame = ttk.Frame(quick_zoom_frame)
    x_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(x_zoom_frame, text="横轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(x_zoom_frame, text="放大", command=quick_zoom_in, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(x_zoom_frame, text="缩小", command=quick_zoom_out, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(x_zoom_frame, text="重置", command=quick_reset_zoom, width=8).pack(side=tk.LEFT, padx=2)
    
    zoom_info_var = tk.StringVar(value=f"当前: {x_scale} 点")
    ttk.Label(quick_zoom_frame, textvariable=zoom_info_var).pack(pady=2)
    
    # EMG纵轴缩放
    emg_zoom_frame = ttk.Frame(quick_zoom_frame)
    emg_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(emg_zoom_frame, text="EMG纵轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(emg_zoom_frame, text="放大", command=quick_emg_zoom_in, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(emg_zoom_frame, text="缩小", command=quick_emg_zoom_out, width=8).pack(side=tk.LEFT, padx=2)
    
    emg_zoom_info_var = tk.StringVar(value=f"EMG: {emg_y_min}~{emg_y_max} μV")
    ttk.Label(quick_zoom_frame, textvariable=emg_zoom_info_var).pack(pady=2)
    
    # IMU纵轴缩放
    imu_zoom_frame = ttk.Frame(quick_zoom_frame)
    imu_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(imu_zoom_frame, text="IMU纵轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(imu_zoom_frame, text="放大", command=quick_imu_zoom_in, width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(imu_zoom_frame, text="缩小", command=quick_imu_zoom_out, width=8).pack(side=tk.LEFT, padx=2)
    
    imu_zoom_info_var = tk.StringVar(value=f"IMU: {imu_y_min}~{imu_y_max}")
    ttk.Label(quick_zoom_frame, textvariable=imu_zoom_info_var).pack(pady=2)
    
    # 高级功能
    advanced_frame = ttk.LabelFrame(control_panel, text="高级功能", padding="10")
    advanced_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(advanced_frame, text="清空数据缓冲区", command=clear_buffers, width=25).pack(pady=5)
    ttk.Button(advanced_frame, text="导出配置", command=export_config, width=25).pack(pady=5)
    ttk.Button(advanced_frame, text="波形缩放控制", command=zoom_control, width=25).pack(pady=5)
    
    # 数据分析
    analysis_frame = ttk.LabelFrame(control_panel, text="数据分析", padding="10")
    analysis_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(analysis_frame, text="分析数据文件", command=analyze_data, width=25).pack(pady=5)
    
    # 系统工具
    tools_frame = ttk.LabelFrame(control_panel, text="系统工具", padding="10")
    tools_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(tools_frame, text="查看文档", command=view_documentation, width=25).pack(pady=5)
    ttk.Button(tools_frame, text="系统设置", command=system_settings, width=25).pack(pady=5)
    
    # 创建右侧图表区域
    plot_frame = ttk.LabelFrame(main_frame, text="实时数据", padding="10")
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
    
    # 创建 matplotlib 图表
    fig = plt.Figure(figsize=(10, 8), dpi=100)
    axs = []
    
    for i in range(8):
        ax = fig.add_subplot(9, 1, i+1)
        ax.set_ylabel(f'EMG {i+1}')
        ax.set_ylim(emg_y_min, emg_y_max)
        ax.grid(True, alpha=0.3)
        axs.append(ax)
    
    ax = fig.add_subplot(9, 1, 9)
    ax.set_ylabel('IMU')
    ax.set_ylim(imu_y_min, imu_y_max)
    ax.grid(True, alpha=0.3)
    axs.append(ax)
    
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # 创建底部信息栏
    info_frame = ttk.Frame(root, padding="10")
    info_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(info_frame, text="EMG 通道: 8个 (单位: μV)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text="IMU 通道: 6个 (单位: rad/s, m/s²)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text="波特率: 921600").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"横轴缩放: {x_scale} 点").pack(side=tk.LEFT, padx=10)
    
    # 初始化串口列表
    refresh_ports()
    
    # 窗口关闭处理
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 启动主循环
    logger.info("系统控制中心已启动")
    root.mainloop()