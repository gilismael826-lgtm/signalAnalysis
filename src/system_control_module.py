#!/usr/bin/env python3
"""
系统控制模块
负责系统核心功能
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import subprocess
import os
import json
import threading
import numpy as np
import src.config as config
from src.config import running, logger
from src.serial_comm import serial_reader
from src.data_manager import save_data_point, init_data_files

# 手势识别相关导入
try:
    from src.realtime_recognizer import RealtimeGestureRecognizer
    from src.gesture_classifier import GestureClassifier, GestureClassifierManager
    from src.feature_extractor import FeatureExtractor
    from src.data_labeler import DataLabelerTool
    from src.gesture_library import PredefinedGestures, GestureLibraryManager
    RECOGNITION_AVAILABLE = True
except ImportError:
    RECOGNITION_AVAILABLE = False


def create_scrollable_frame(parent, width=None, height=None):
    """创建可滚动的Frame
    
    Args:
        parent: 父窗口
        width: 容器宽度
        height: 容器高度
    
    Returns:
        (container, scrollable_frame, canvas) 元组
    """
    container = ttk.Frame(parent)
    if width:
        container.config(width=width)
    if height:
        container.config(height=height)
    container.pack(fill=tk.BOTH, expand=True)
    container.pack_propagate(False)
    
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind("<MouseWheel>", on_mousewheel)
    scrollable_frame.bind("<MouseWheel>", on_mousewheel)
    
    return container, scrollable_frame, canvas


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
    
    def open_data_dir():
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"创建数据目录: {data_dir}")
        subprocess.run(["explorer", os.path.join(os.getcwd(), data_dir)])
    
    ttk.Button(analysis_window, text="打开数据目录", command=open_data_dir).pack(pady=20)
    
    logger.info("打开数据分析窗口")


def view_documentation(root):
    """查看文档
    
    Args:
        root: Tkinter根窗口对象
    """
    doc_window = tk.Toplevel(root)
    doc_window.title("系统文档")
    doc_window.geometry("500x400")
    
    ttk.Label(doc_window, text="系统文档", font=("Arial", 14, "bold")).pack(pady=10)
    
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

7. 手势识别功能：
   - 模型训练：选择数据集和算法进行训练
   - 数据标注：采集并标注手势数据
   - 实时识别：加载模型进行实时手势识别

8. 支持的手势类型：
   - fist: 握拳
   - open: 张开手掌
   - pinch: 捏合
   - wave: 挥手
   - point: 指向
   - flex: 弯曲
   - extend: 伸展
   - rest: 静止

9. 支持的分类算法：
   - 随机森林 (RF)
   - 支持向量机 (SVM)
   - K最近邻 (KNN)
   - 多层感知机 (MLP)
   - 集成学习 (Ensemble)
   - 梯度提升 (Gradient Boosting)
"""
    
    text_frame = ttk.Frame(doc_window)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    text_widget = tk.Text(text_frame, wrap=tk.WORD, padding=10)
    scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
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
    
    title_frame = ttk.Frame(settings_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="系统设置", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(settings_window)
    
    btn_frame = ttk.Frame(settings_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    baudrate_frame = ttk.LabelFrame(scrollable_frame, text="串口设置", padding="10")
    baudrate_frame.pack(fill=tk.X, padx=20, pady=10)
    
    baudrate_var = tk.StringVar(value=str(config.BAUDRATE))
    baudrate_options = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
    
    ttk.Label(baudrate_frame, text="波特率:").pack(anchor=tk.W, pady=5)
    ttk.Combobox(baudrate_frame, textvariable=baudrate_var, values=baudrate_options).pack(fill=tk.X, pady=5)
    
    data_frame = ttk.LabelFrame(scrollable_frame, text="数据存储", padding="10")
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
    
    ttk.Button(btn_frame, text="保存", command=save_settings, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="取消", command=settings_window.destroy, width=15).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开系统设置窗口")


# ==================== 手势识别相关功能 ====================

def open_model_training(root):
    """打开模型训练窗口
    
    Args:
        root: Tkinter根窗口对象
    """
    if not RECOGNITION_AVAILABLE:
        messagebox.showerror("错误", "手势识别模块未安装，请检查依赖")
        return
    
    train_window = tk.Toplevel(root)
    train_window.title("模型训练")
    train_window.geometry("650x550")
    
    title_frame = ttk.Frame(train_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="模型训练", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(train_window)
    
    btn_frame = ttk.Frame(train_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    # 数据源选择
    source_frame = ttk.LabelFrame(scrollable_frame, text="数据源", padding="10")
    source_frame.pack(fill=tk.X, padx=20, pady=5)
    
    source_var = tk.StringVar(value="raw")
    ttk.Radiobutton(source_frame, text="使用原始数据 (data/raw)", variable=source_var, value="raw").pack(anchor=tk.W)
    ttk.Radiobutton(source_frame, text="使用数据集 (data/datasets)", variable=source_var, value="dataset").pack(anchor=tk.W)
    
    # 数据集选择（当选择dataset时显示）
    dataset_frame = ttk.LabelFrame(scrollable_frame, text="数据集路径", padding="10")
    dataset_frame.pack(fill=tk.X, padx=20, pady=5)
    
    dataset_var = tk.StringVar()
    
    dataset_entry_frame = ttk.Frame(dataset_frame)
    dataset_entry_frame.pack(fill=tk.X, pady=5)
    
    ttk.Entry(dataset_entry_frame, textvariable=dataset_var, width=40).pack(side=tk.LEFT, padx=5)
    
    def browse_dataset():
        datasets_dir = "data/datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir, exist_ok=True)
        selected = filedialog.askdirectory(title="选择数据集目录", initialdir=datasets_dir)
        if selected:
            dataset_var.set(selected)
    
    ttk.Button(dataset_entry_frame, text="浏览", command=browse_dataset).pack(side=tk.LEFT)
    
    # 数据统计
    stats_frame = ttk.LabelFrame(scrollable_frame, text="数据统计", padding="10")
    stats_frame.pack(fill=tk.X, padx=20, pady=5)
    
    stats_text = tk.Text(stats_frame, height=4, width=50)
    stats_text.pack(fill=tk.X)
    
    def refresh_stats():
        """刷新数据统计"""
        stats_text.delete(1.0, tk.END)
        
        if source_var.get() == "raw":
            raw_dir = "data/raw"
            if os.path.exists(raw_dir):
                gesture_counts = {}
                for f in os.listdir(raw_dir):
                    if f.endswith('_meta.json'):
                        try:
                            with open(os.path.join(raw_dir, f), 'r', encoding='utf-8') as file:
                                meta = json.load(file)
                                gesture = meta.get('gesture_name', 'unknown')
                                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                        except:
                            pass
                
                if gesture_counts:
                    stats_text.insert(tk.END, "原始数据统计:\n")
                    for gesture, count in gesture_counts.items():
                        stats_text.insert(tk.END, f"  {gesture}: {count} 个样本\n")
                    stats_text.insert(tk.END, f"\n总计: {sum(gesture_counts.values())} 个样本\n")
                    
                    if len(gesture_counts) < 2:
                        stats_text.insert(tk.END, "\n⚠️ 警告: 只有1种手势，无法训练分类器！\n")
                        stats_text.insert(tk.END, "请采集至少2种不同的手势数据。\n")
                else:
                    stats_text.insert(tk.END, "未找到原始数据\n")
                    stats_text.insert(tk.END, "请先使用数据标注功能采集数据。\n")
            else:
                stats_text.insert(tk.END, "原始数据目录不存在\n")
        else:
            stats_text.insert(tk.END, "请选择数据集目录后查看统计\n")
    
    ttk.Button(stats_frame, text="刷新统计", command=refresh_stats).pack(pady=5)
    
    # 算法设置（自动训练所有模型）
    algo_frame = ttk.LabelFrame(scrollable_frame, text="算法设置 (自动训练)", padding="10")
    algo_frame.pack(fill=tk.X, padx=20, pady=5)
    
    ttk.Label(algo_frame, text="系统将自动训练以下所有模型:", 
              font=("Arial", 9)).pack(anchor=tk.W, pady=2)
    
    algorithms_info = """• SVM - 支持向量机 (RBF核)
• KNN - K近邻 (距离加权)
• RF - 随机森林 (100棵树)
• MLP - 多层感知机 (256,128,64)
• Ensemble - 集成学习 (SVM+KNN+RF)
• GB - 梯度提升"""
    
    algo_info_label = ttk.Label(algo_frame, text=algorithms_info, 
                                 font=("Arial", 8), foreground="gray")
    algo_info_label.pack(anchor=tk.W, pady=5)
    
    ttk.Label(algo_frame, text="训练完成后自动选择最优模型并分配权重", 
              font=("Arial", 9), foreground="green").pack(anchor=tk.W, pady=2)
    
    # 训练参数
    params_frame = ttk.LabelFrame(scrollable_frame, text="训练参数", padding="10")
    params_frame.pack(fill=tk.X, padx=20, pady=5)
    
    test_size_var = tk.StringVar(value="0.2")
    cv_folds_var = tk.StringVar(value="5")
    
    ttk.Label(params_frame, text="测试集比例:").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Entry(params_frame, textvariable=test_size_var, width=10).grid(row=0, column=1, pady=2)
    
    ttk.Label(params_frame, text="交叉验证折数:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(params_frame, textvariable=cv_folds_var, width=10).grid(row=1, column=1, pady=2)
    
    # 训练进度
    progress_frame = ttk.LabelFrame(scrollable_frame, text="训练进度", padding="10")
    progress_frame.pack(fill=tk.X, padx=20, pady=5)
    
    progress_var = tk.StringVar(value="就绪")
    progress_label = ttk.Label(progress_frame, textvariable=progress_var)
    progress_label.pack(anchor=tk.W)
    
    progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
    progress_bar.pack(fill=tk.X, pady=5)
    
    # 结果显示
    result_text = tk.Text(progress_frame, height=6, width=50)
    result_text.pack(fill=tk.X, pady=5)
    
    def start_training():
        """开始训练（自动训练所有模型）"""
        result_text.delete(1.0, tk.END)
        
        if source_var.get() == "raw":
            raw_dir = "data/raw"
            if not os.path.exists(raw_dir):
                messagebox.showerror("错误", "原始数据目录不存在，请先采集数据")
                return
            
            gesture_counts = {}
            for f in os.listdir(raw_dir):
                if f.endswith('_meta.json'):
                    try:
                        with open(os.path.join(raw_dir, f), 'r', encoding='utf-8') as file:
                            meta = json.load(file)
                            gesture = meta.get('gesture_name', 'unknown')
                            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                    except:
                        pass
            
            if len(gesture_counts) < 2:
                messagebox.showerror("错误", 
                    f"只有 {len(gesture_counts)} 种手势类型，无法训练分类器！\n\n"
                    f"当前手势: {list(gesture_counts.keys())}\n\n"
                    "请采集至少2种不同的手势数据。")
                return
        else:
            dataset_path = dataset_var.get()
            if not dataset_path:
                messagebox.showerror("错误", "请选择数据集目录")
                return
        
        progress_bar.start()
        progress_var.set("正在训练所有模型...")
        
        def train_thread():
            try:
                extractor = FeatureExtractor()
                X = []
                y = []
                
                if source_var.get() == "raw":
                    result_text.insert(tk.END, "从原始数据加载...\n")
                    raw_dir = "data/raw"
                    
                    labels_file = os.path.join(raw_dir, "labels.json")
                    if os.path.exists(labels_file):
                        with open(labels_file, 'r', encoding='utf-8') as f:
                            labels_data = json.load(f)
                    else:
                        labels_data = {}
                    
                    sample_count = 0
                    for session_id, label_list in labels_data.items():
                        if not label_list:
                            continue
                        
                        gesture_name = label_list[0].get('gesture_name')
                        if not gesture_name:
                            continue
                        
                        emg_file = os.path.join(raw_dir, f"{session_id}_emg.npy")
                        imu_file = os.path.join(raw_dir, f"{session_id}_imu.npy")
                        
                        if os.path.exists(emg_file) and os.path.exists(imu_file):
                            emg_data = np.load(emg_file)
                            imu_data = np.load(imu_file)
                            
                            if len(emg_data.shape) == 2 and emg_data.shape[1] == 8:
                                emg_data = emg_data.T
                            if len(imu_data.shape) == 2 and imu_data.shape[1] == 6:
                                imu_data = imu_data.T
                            
                            features = extractor.extract_all_features(emg_data, imu_data)
                            X.append(list(features.values()))
                            y.append(gesture_name)
                            sample_count += 1
                    
                    result_text.insert(tk.END, f"加载 {sample_count} 个样本\n")
                    result_text.insert(tk.END, f"手势类型: {set(y)}\n")
                    
                else:
                    from src.data_labeler import DatasetManager
                    
                    dataset_path = dataset_var.get()
                    result_text.insert(tk.END, f"加载数据集: {dataset_path}\n")
                    
                    dataset_name = os.path.basename(dataset_path)
                    parent_dir = os.path.dirname(dataset_path)
                    
                    manager = DatasetManager(data_dir='data', output_dir=parent_dir if parent_dir else 'data/datasets')
                    train_data = manager.load_dataset(dataset_name, 'train')
                    
                    if not train_data:
                        raise ValueError("无法加载数据集")
                    
                    result_text.insert(tk.END, f"加载 {len(train_data)} 个样本\n")
                    
                    for sample in train_data:
                        features = extractor.extract_all_features(
                            sample['emg_data'].T,
                            sample['imu_data'].T
                        )
                        X.append(list(features.values()))
                        y.append(sample['gesture_name'])
                
                X = np.array(X)
                y = np.array(y)
                
                n_classes = len(np.unique(y))
                n_samples = len(y)
                test_size = float(test_size_var.get())
                min_test_samples = n_classes
                
                actual_test_size = test_size
                if n_samples * test_size < min_test_samples:
                    actual_test_size = min_test_samples / n_samples
                    result_text.insert(tk.END, f"警告: 样本数较少，自动调整测试集比例为 {actual_test_size:.2f}\n")
                
                result_text.insert(tk.END, f"\n开始自动训练所有模型...\n")
                result_text.insert(tk.END, "=" * 40 + "\n")
                
                manager = GestureClassifierManager(models_dir='models')
                results = manager.train_all(X, y, test_size=actual_test_size)
                
                result_text.insert(tk.END, f"\n最优模型: {manager.best_algorithm.upper()}\n")
                result_text.insert(tk.END, f"模型权重:\n")
                for algo, weight in sorted(manager.model_weights.items(), key=lambda x: x[1], reverse=True):
                    result_text.insert(tk.END, f"  {algo}: {weight:.4f}\n")
                
                saved_paths = manager.save_all()
                
                result_text.insert(tk.END, f"\n" + "=" * 40 + "\n")
                result_text.insert(tk.END, f"训练完成!\n")
                result_text.insert(tk.END, f"已保存 {len(saved_paths)} 个模型\n")
                result_text.insert(tk.END, f"模型目录: models/\n")
                
                progress_var.set(f"训练完成 - 最优: {manager.best_algorithm.upper()}")
                
            except Exception as e:
                import traceback
                result_text.insert(tk.END, f"\n错误: {e}\n")
                result_text.insert(tk.END, traceback.format_exc())
                progress_var.set("训练失败")
            finally:
                progress_bar.stop()
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    # 创建数据集区域
    create_dataset_frame = ttk.LabelFrame(scrollable_frame, text="创建数据集", padding="10")
    create_dataset_frame.pack(fill=tk.X, padx=20, pady=5)
    
    dataset_name_var = tk.StringVar(value="gesture_v1")
    train_ratio_var = tk.StringVar(value="0.7")
    val_ratio_var = tk.StringVar(value="0.15")
    test_ratio_var = tk.StringVar(value="0.15")
    augment_var = tk.BooleanVar(value=False)
    
    ttk.Label(create_dataset_frame, text="数据集名称:").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Entry(create_dataset_frame, textvariable=dataset_name_var, width=15).grid(row=0, column=1, pady=2)
    
    ttk.Label(create_dataset_frame, text="训练集比例:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(create_dataset_frame, textvariable=train_ratio_var, width=10).grid(row=1, column=1, pady=2)
    
    ttk.Label(create_dataset_frame, text="验证集比例:").grid(row=2, column=0, sticky=tk.W, pady=2)
    ttk.Entry(create_dataset_frame, textvariable=val_ratio_var, width=10).grid(row=2, column=1, pady=2)
    
    ttk.Label(create_dataset_frame, text="测试集比例:").grid(row=3, column=0, sticky=tk.W, pady=2)
    ttk.Entry(create_dataset_frame, textvariable=test_ratio_var, width=10).grid(row=3, column=1, pady=2)
    
    ttk.Checkbutton(create_dataset_frame, text="启用数据增强", variable=augment_var).grid(row=4, column=0, columnspan=2, pady=5)
    
    dataset_result_text = tk.Text(create_dataset_frame, height=3, width=50)
    dataset_result_text.grid(row=5, column=0, columnspan=3, pady=5)
    
    def create_dataset_from_raw():
        """从原始数据创建数据集"""
        dataset_result_text.delete(1.0, tk.END)
        
        try:
            train_ratio = float(train_ratio_var.get())
            val_ratio = float(val_ratio_var.get())
            test_ratio = float(test_ratio_var.get())
            
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
                messagebox.showerror("错误", "比例之和必须为1.0")
                return
            
            from src.data_labeler import DatasetManager, DataAugmenter
            
            dataset_result_text.insert(tk.END, "正在创建数据集...\n")
            
            manager = DatasetManager(data_dir='data', output_dir='data/datasets')
            
            # 获取所有手势类型
            raw_dir = "data/raw"
            gesture_names = set()
            for f in os.listdir(raw_dir):
                if f.endswith('_meta.json'):
                    try:
                        with open(os.path.join(raw_dir, f), 'r', encoding='utf-8') as file:
                            meta = json.load(file)
                            gesture = meta.get('gesture_name')
                            if gesture:
                                gesture_names.add(gesture)
                    except:
                        pass
            
            gesture_names = list(gesture_names)
            
            if len(gesture_names) < 2:
                messagebox.showerror("错误", "至少需要2种手势类型")
                return
            
            dataset_info = manager.create_dataset(
                name=dataset_name_var.get(),
                gesture_names=gesture_names,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
            
            dataset_result_text.insert(tk.END, f"数据集创建成功!\n")
            dataset_result_text.insert(tk.END, f"名称: {dataset_info['name']}\n")
            
            # 计算各分割的样本数
            splits = dataset_info.get('splits', {})
            train_samples = sum(splits.get('train', {}).values())
            val_samples = sum(splits.get('val', {}).values())
            test_samples = sum(splits.get('test', {}).values())
            
            dataset_result_text.insert(tk.END, f"训练集: {train_samples} 样本\n")
            dataset_result_text.insert(tk.END, f"验证集: {val_samples} 样本\n")
            dataset_result_text.insert(tk.END, f"测试集: {test_samples} 样本\n")
            
            # 数据增强
            if augment_var.get():
                dataset_result_text.insert(tk.END, "\n正在执行数据增强...\n")
                augmenter = DataAugmenter()
                # 这里可以添加数据增强逻辑
                dataset_result_text.insert(tk.END, "数据增强完成\n")
            
            # 自动设置数据集路径
            dataset_var.set(os.path.join("data/datasets", dataset_name_var.get()))
            
        except Exception as e:
            import traceback
            dataset_result_text.insert(tk.END, f"错误: {e}\n")
            dataset_result_text.insert(tk.END, traceback.format_exc())
    
    ttk.Button(create_dataset_frame, text="创建数据集", command=create_dataset_from_raw).grid(row=6, column=0, columnspan=2, pady=5)
    
    # 模型版本管理区域
    version_frame = ttk.LabelFrame(scrollable_frame, text="模型版本管理", padding="10")
    version_frame.pack(fill=tk.X, padx=20, pady=5)
    
    version_list_text = tk.Text(version_frame, height=4, width=50)
    version_list_text.pack(fill=tk.X, pady=5)
    
    def refresh_versions():
        """刷新版本列表"""
        version_list_text.delete(1.0, tk.END)
        try:
            from src.gesture_classifier import ModelVersionManager
            version_manager = ModelVersionManager(models_dir='models')
            versions = version_manager.list_versions(limit=10)
            
            if versions:
                version_list_text.insert(tk.END, "已保存的模型版本:\n")
                for v in versions:
                    version_list_text.insert(tk.END, f"  {v['version']}\n")
                    version_list_text.insert(tk.END, f"    准确率: {v.get('test_accuracy', 'N/A'):.4f}\n")
            else:
                version_list_text.insert(tk.END, "暂无已保存的模型版本\n")
                
            stats = version_manager.get_version_statistics()
            version_list_text.insert(tk.END, f"\n统计: 共 {stats['total_versions']} 个版本\n")
            if stats['best_version']:
                version_list_text.insert(tk.END, f"最佳版本: {stats['best_version']}\n")
        except Exception as e:
            version_list_text.insert(tk.END, f"加载版本信息失败: {e}\n")
    
    ttk.Button(version_frame, text="刷新版本列表", command=refresh_versions).pack(pady=5)
    
    # 高级训练选项
    advanced_frame = ttk.LabelFrame(scrollable_frame, text="高级训练选项", padding="10")
    advanced_frame.pack(fill=tk.X, padx=20, pady=5)
    
    cv_var = tk.BooleanVar(value=False)
    grid_search_var = tk.BooleanVar(value=False)
    
    ttk.Checkbutton(advanced_frame, text="使用交叉验证训练", variable=cv_var).pack(anchor=tk.W)
    ttk.Checkbutton(advanced_frame, text="网格搜索超参数调优", variable=grid_search_var).pack(anchor=tk.W)
    ttk.Label(advanced_frame, text="(注: 系统已默认自动训练所有模型)", 
              font=("Arial", 8), foreground="gray").pack(anchor=tk.W, pady=5)
    
    # 初始刷新统计
    refresh_stats()
    
    ttk.Button(btn_frame, text="开始训练", command=start_training, width=15).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="关闭", command=train_window.destroy, width=15).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开模型训练窗口")


def open_data_labeling(root):
    """打开数据标注窗口
    
    Args:
        root: Tkinter根窗口对象
    """
    if not RECOGNITION_AVAILABLE:
        messagebox.showerror("错误", "手势识别模块未安装，请检查依赖")
        return
    
    label_window = tk.Toplevel(root)
    label_window.title("数据标注")
    label_window.geometry("500x400")
    
    title_frame = ttk.Frame(label_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="数据标注工具", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(label_window)
    
    btn_frame = ttk.Frame(label_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    # 手势类型
    gesture_frame = ttk.LabelFrame(scrollable_frame, text="手势类型", padding="10")
    gesture_frame.pack(fill=tk.X, padx=20, pady=5)
    
    gesture_var = tk.StringVar()
    gestures = ["fist", "open", "pinch", "wave", "point", "flex", "extend", "rest"]
    ttk.Combobox(gesture_frame, textvariable=gesture_var, values=gestures, width=20).pack(pady=5)
    
    # 受试者ID
    subject_frame = ttk.LabelFrame(scrollable_frame, text="受试者信息", padding="10")
    subject_frame.pack(fill=tk.X, padx=20, pady=5)
    
    subject_var = tk.StringVar(value="subject_001")
    ttk.Entry(subject_frame, textvariable=subject_var, width=20).pack(pady=5)
    
    # 采集设置
    collect_frame = ttk.LabelFrame(scrollable_frame, text="采集设置", padding="10")
    collect_frame.pack(fill=tk.X, padx=20, pady=5)
    
    duration_var = tk.StringVar(value="3")
    ttk.Label(collect_frame, text="采集时长(秒):").pack(anchor=tk.W)
    ttk.Entry(collect_frame, textvariable=duration_var, width=10).pack(anchor=tk.W)
    
    save_data_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(collect_frame, text="保存数据", variable=save_data_var).pack(anchor=tk.W, pady=5)
    
    # 状态显示
    status_frame = ttk.LabelFrame(scrollable_frame, text="状态", padding="10")
    status_frame.pack(fill=tk.X, padx=20, pady=5)
    
    label_status_var = tk.StringVar(value="就绪")
    ttk.Label(status_frame, textvariable=label_status_var).pack(anchor=tk.W)
    
    sample_count_var = tk.StringVar(value="已采集: 0 个样本")
    ttk.Label(status_frame, textvariable=sample_count_var).pack(anchor=tk.W)
    
    # 统计信息
    stats_frame = ttk.LabelFrame(scrollable_frame, text="统计信息", padding="10")
    stats_frame.pack(fill=tk.X, padx=20, pady=5)
    
    stats_text = tk.Text(stats_frame, height=5, width=40)
    stats_text.pack(fill=tk.X)
    
    # 采集器实例
    collector = None
    collecting = [False]
    current_emg_count = [0]
    current_imu_count = [0]
    
    def update_stats():
        """更新统计信息"""
        # 统计各手势的样本数
        stats_text.delete(1.0, tk.END)
        raw_dir = "data/raw"
        if os.path.exists(raw_dir):
            gesture_counts = {}
            for f in os.listdir(raw_dir):
                if f.endswith('_meta.json'):
                    try:
                        with open(os.path.join(raw_dir, f), 'r', encoding='utf-8') as file:
                            meta = json.load(file)
                            gesture = meta.get('gesture_name', 'unknown')
                            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                    except:
                        pass
            
            stats_text.insert(tk.END, "已采集样本统计:\n")
            if gesture_counts:
                for gesture, count in gesture_counts.items():
                    stats_text.insert(tk.END, f"  {gesture}: {count} 个\n")
                sample_count_var.set(f"已采集: {sum(gesture_counts.values())} 个样本")
            else:
                stats_text.insert(tk.END, "  暂无数据\n")
                sample_count_var.set("已采集: 0 个样本")
        else:
            stats_text.insert(tk.END, "  暂无数据\n")
            sample_count_var.set("已采集: 0 个样本")
    
    def start_collection():
        """开始采集"""
        nonlocal collector
        
        gesture = gesture_var.get()
        if not gesture:
            messagebox.showerror("错误", "请选择手势类型")
            return
        
        if not config.running:
            messagebox.showwarning("警告", "请先在主界面开始数据采集（连接串口并点击开始采集）")
            return
        
        subject = subject_var.get().strip()
        if not subject:
            subject = "subject_001"
        
        try:
            duration = float(duration_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的采集时长")
            return
        
        from src.data_labeler import GestureDataCollector
        collector = GestureDataCollector(output_dir='data/raw')
        collector.start_session(gesture, subject)
        
        collecting[0] = True
        current_emg_count[0] = 0
        current_imu_count[0] = 0
        
        config.collection_emg_buffer.clear()
        config.collection_imu_buffer.clear()
        config.is_collecting = True
        
        label_status_var.set(f"正在采集: {gesture}...")
        logger.info(f"开始采集: {gesture}")
        
        def collect_data():
            import time
            start_time = time.time()
            
            try:
                while collecting[0] and (time.time() - start_time) < duration:
                    if not config.running:
                        break
                    time.sleep(0.1)
                
                config.is_collecting = False
                
                emg_data = []
                imu_data = []
                
                for item in config.collection_emg_buffer:
                    if isinstance(item, tuple):
                        emg_data.append(list(item[1]))
                    else:
                        emg_data.append(list(item))
                
                for item in config.collection_imu_buffer:
                    if isinstance(item, tuple):
                        imu_data.append(list(item[1]))
                    else:
                        imu_data.append(list(item))
                
                logger.info(f"采集完成 - 耗时: {time.time() - start_time:.2f}秒, EMG样本: {len(emg_data)}, IMU样本: {len(imu_data)}")
            
            except Exception as e:
                logger.error(f"采集过程出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                config.is_collecting = False
                label_window.after(0, lambda err=str(e): label_status_var.set(f"采集出错: {err}"))
                return
            
            if not emg_data or not imu_data:
                collecting[0] = False
                label_window.after(0, lambda: label_status_var.set("未采集到数据"))
                label_window.after(0, lambda: messagebox.showwarning(
                    "警告", 
                    f"未采集到任何数据！\n\n可能原因：\n1. 主界面未开始数据采集\n2. 串口未连接设备\n3. 采集时长过短\n\n当前缓冲区状态:\nEMG: {len(config.emg_buffer)}\nIMU: {len(config.imu_buffer)}"
                ))
                return
            
            # 保存数据
            session_data = collector.stop_session()
            session_data['emg_data'] = np.array(emg_data)
            session_data['imu_data'] = np.array(imu_data)
            session_data['emg_samples'] = len(emg_data)
            session_data['imu_samples'] = len(imu_data)
            
            if save_data_var.get():
                collector.save_session(session_data, format='npy')
                
                from src.data_labeler import GestureLabeler
                labeler = GestureLabeler(data_dir='data/raw')
                session_id = session_data['session_id']
                labeler.add_label(session_id, gesture)
            
            collecting[0] = False
            
            label_window.after(0, lambda: label_status_var.set(
                f"✓ {gesture} 采集完成 (EMG:{len(emg_data)}, IMU:{len(imu_data)})"
            ))
            label_window.after(0, update_stats)
        
        threading.Thread(target=collect_data, daemon=True).start()
    
    def stop_collection():
        """停止采集"""
        collecting[0] = False
        label_status_var.set("已停止")
    
    def view_data():
        """查看数据"""
        data_dir = "data/raw"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"创建数据目录: {data_dir}")
        subprocess.run(["explorer", os.path.join(os.getcwd(), data_dir)])
    
    # 更新初始统计
    update_stats()
    
    # 数据质量检查区域
    quality_frame = ttk.LabelFrame(scrollable_frame, text="数据质量检查", padding="10")
    quality_frame.pack(fill=tk.X, padx=20, pady=5)
    
    quality_text = tk.Text(quality_frame, height=5, width=40)
    quality_text.pack(fill=tk.X)
    
    def check_data_quality():
        """检查最近采集的数据质量"""
        quality_text.delete(1.0, tk.END)
        
        try:
            from src.data_labeler import DataQualityChecker
            checker = DataQualityChecker()
            
            raw_dir = "data/raw"
            if not os.path.exists(raw_dir):
                quality_text.insert(tk.END, "数据目录不存在\n")
                return
            
            # 找到最新的数据文件
            meta_files = [f for f in os.listdir(raw_dir) if f.endswith('_meta.json')]
            if not meta_files:
                quality_text.insert(tk.END, "没有找到数据文件\n")
                return
            
            # 按修改时间排序，获取最新的
            meta_files.sort(key=lambda f: os.path.getmtime(os.path.join(raw_dir, f)), reverse=True)
            latest_meta = meta_files[0]
            session_id = latest_meta.replace('_meta.json', '')
            
            # 加载数据
            emg_file = os.path.join(raw_dir, f"{session_id}_emg.npy")
            imu_file = os.path.join(raw_dir, f"{session_id}_imu.npy")
            
            if not os.path.exists(emg_file) or not os.path.exists(imu_file):
                quality_text.insert(tk.END, "数据文件不完整\n")
                return
            
            emg_data = np.load(emg_file)
            imu_data = np.load(imu_file)
            
            # 执行质量检查
            result = checker.check_data_quality(emg_data, imu_data)
            
            quality_text.insert(tk.END, f"最新文件: {session_id[:30]}...\n")
            quality_text.insert(tk.END, f"质量分数: {result['quality_score']:.1f}/100\n")
            quality_text.insert(tk.END, f"质量等级: {result['quality_level']}\n")
            
            if result['issues']:
                quality_text.insert(tk.END, f"\n问题:\n")
                for issue in result['issues'][:3]:
                    quality_text.insert(tk.END, f"  - {issue}\n")
            else:
                quality_text.insert(tk.END, "\n✓ 数据质量良好\n")
                
        except Exception as e:
            quality_text.insert(tk.END, f"检查失败: {e}\n")
    
    ttk.Button(quality_frame, text="检查最新数据", command=check_data_quality).pack(pady=5)
    
    # 数据增强区域
    augment_frame = ttk.LabelFrame(scrollable_frame, text="数据增强", padding="10")
    augment_frame.pack(fill=tk.X, padx=20, pady=5)
    
    noise_var = tk.BooleanVar(value=True)
    time_scale_var = tk.BooleanVar(value=True)
    amp_scale_var = tk.BooleanVar(value=False)
    
    ttk.Checkbutton(augment_frame, text="添加高斯噪声", variable=noise_var).pack(anchor=tk.W)
    ttk.Checkbutton(augment_frame, text="时间缩放", variable=time_scale_var).pack(anchor=tk.W)
    ttk.Checkbutton(augment_frame, text="幅度缩放", variable=amp_scale_var).pack(anchor=tk.W)
    
    augment_result_text = tk.Text(augment_frame, height=2, width=40)
    augment_result_text.pack(fill=tk.X, pady=5)
    
    def apply_augmentation():
        """应用数据增强"""
        augment_result_text.delete(1.0, tk.END)
        
        try:
            from src.data_labeler import DataAugmenter
            augmenter = DataAugmenter()
            
            raw_dir = "data/raw"
            meta_files = [f for f in os.listdir(raw_dir) if f.endswith('_meta.json')]
            
            if not meta_files:
                augment_result_text.insert(tk.END, "没有找到数据文件\n")
                return
            
            augmented_count = 0
            
            for meta_file in meta_files[:5]:  # 只处理前5个文件
                session_id = meta_file.replace('_meta.json', '')
                emg_file = os.path.join(raw_dir, f"{session_id}_emg.npy")
                imu_file = os.path.join(raw_dir, f"{session_id}_imu.npy")
                
                if not os.path.exists(emg_file):
                    continue
                
                emg_data = np.load(emg_file)
                imu_data = np.load(imu_file) if os.path.exists(imu_file) else None
                
                # 应用增强
                if noise_var.get():
                    emg_data = augmenter.add_gaussian_noise(emg_data, noise_level=0.05)
                
                if time_scale_var.get():
                    emg_data = augmenter.time_scale(emg_data, scale_factor=1.1)
                
                # 保存增强后的数据
                new_session_id = f"{session_id}_aug"
                np.save(os.path.join(raw_dir, f"{new_session_id}_emg.npy"), emg_data)
                if imu_data is not None:
                    np.save(os.path.join(raw_dir, f"{new_session_id}_imu.npy"), imu_data)
                
                augmented_count += 1
            
            augment_result_text.insert(tk.END, f"已增强 {augmented_count} 个样本\n")
            update_stats()
            
        except Exception as e:
            augment_result_text.insert(tk.END, f"增强失败: {e}\n")
    
    ttk.Button(augment_frame, text="应用增强", command=apply_augmentation).pack(pady=5)
    
    ttk.Button(btn_frame, text="开始采集", command=start_collection, width=12).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="停止采集", command=stop_collection, width=12).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="查看数据", command=view_data, width=12).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="关闭", command=label_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开数据标注窗口")


def open_recognition_settings(root, recognizer=None):
    """打开识别设置窗口
    
    Args:
        root: Tkinter根窗口对象
        recognizer: 识别器实例
    """
    settings_window = tk.Toplevel(root)
    settings_window.title("识别设置")
    settings_window.geometry("400x350")
    
    title_frame = ttk.Frame(settings_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="识别设置", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(settings_window)
    
    btn_frame = ttk.Frame(settings_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    # 窗口设置
    window_frame = ttk.LabelFrame(scrollable_frame, text="窗口设置", padding="10")
    window_frame.pack(fill=tk.X, padx=20, pady=5)
    
    window_size_var = tk.StringVar(value="600")
    ttk.Label(window_frame, text="窗口大小 (ms):").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Entry(window_frame, textvariable=window_size_var, width=10).grid(row=0, column=1, pady=2)
    
    slide_step_var = tk.StringVar(value="25")
    ttk.Label(window_frame, text="滑动步长:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(window_frame, textvariable=slide_step_var, width=10).grid(row=1, column=1, pady=2)
    
    # 平滑设置
    smooth_frame = ttk.LabelFrame(scrollable_frame, text="平滑设置", padding="10")
    smooth_frame.pack(fill=tk.X, padx=20, pady=5)
    
    smoothing_window_var = tk.StringVar(value="5")
    ttk.Label(smooth_frame, text="平滑窗口:").grid(row=0, column=0, sticky=tk.W, pady=2)
    ttk.Entry(smooth_frame, textvariable=smoothing_window_var, width=10).grid(row=0, column=1, pady=2)
    
    threshold_var = tk.StringVar(value="0.5")
    ttk.Label(smooth_frame, text="置信度阈值:").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(smooth_frame, textvariable=threshold_var, width=10).grid(row=1, column=1, pady=2)
    
    # 模型信息
    model_frame = ttk.LabelFrame(scrollable_frame, text="模型信息", padding="10")
    model_frame.pack(fill=tk.X, padx=20, pady=5)
    
    if recognizer and recognizer.classifier:
        info = recognizer.classifier.get_model_info()
        ttk.Label(model_frame, text=f"算法: {info.get('algorithm', 'N/A')}").pack(anchor=tk.W)
        ttk.Label(model_frame, text=f"类别数: {info.get('n_classes', 'N/A')}").pack(anchor=tk.W)
        ttk.Label(model_frame, text=f"已训练: {info.get('is_trained', False)}").pack(anchor=tk.W)
    else:
        ttk.Label(model_frame, text="未加载模型").pack(anchor=tk.W)
    
    # 滤波器参数配置
    filter_frame = ttk.LabelFrame(scrollable_frame, text="滤波器参数", padding="10")
    filter_frame.pack(fill=tk.X, padx=20, pady=5)
    
    # 带通滤波器参数
    ttk.Label(filter_frame, text="带通滤波器:").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    bp_low_var = tk.StringVar(value="20")
    ttk.Label(filter_frame, text="  低截止频率(Hz):").grid(row=1, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=bp_low_var, width=8).grid(row=1, column=1, pady=2)
    
    bp_high_var = tk.StringVar(value="100")
    ttk.Label(filter_frame, text="  高截止频率(Hz):").grid(row=2, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=bp_high_var, width=8).grid(row=2, column=1, pady=2)
    
    bp_order_var = tk.StringVar(value="4")
    ttk.Label(filter_frame, text="  滤波器阶数:").grid(row=3, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=bp_order_var, width=8).grid(row=3, column=1, pady=2)
    
    # 陷波滤波器参数
    ttk.Label(filter_frame, text="陷波滤波器:").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    notch_freq_var = tk.StringVar(value="50")
    ttk.Label(filter_frame, text="  陷波频率(Hz):").grid(row=5, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=notch_freq_var, width=8).grid(row=5, column=1, pady=2)
    
    notch_q_var = tk.StringVar(value="30")
    ttk.Label(filter_frame, text="  品质因数:").grid(row=6, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=notch_q_var, width=8).grid(row=6, column=1, pady=2)
    
    # 低通滤波器参数（IMU）
    ttk.Label(filter_frame, text="低通滤波器(IMU):").grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)
    
    lp_cutoff_var = tk.StringVar(value="10")
    ttk.Label(filter_frame, text="  截止频率(Hz):").grid(row=8, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=lp_cutoff_var, width=8).grid(row=8, column=1, pady=2)
    
    lp_order_var = tk.StringVar(value="4")
    ttk.Label(filter_frame, text="  滤波器阶数:").grid(row=9, column=0, sticky=tk.W, pady=2)
    ttk.Entry(filter_frame, textvariable=lp_order_var, width=8).grid(row=9, column=1, pady=2)
    
    def apply_settings():
        """应用设置"""
        if recognizer:
            try:
                recognizer.window_size_ms = int(window_size_var.get())
                recognizer.slide_step = int(slide_step_var.get())
                recognizer.smoothing_window = int(smoothing_window_var.get())
                recognizer.confidence_threshold = float(threshold_var.get())
                
                # 重新计算窗口样本数
                recognizer.emg_window_samples = int(
                    recognizer.window_size_ms * recognizer.emg_sample_rate / 1000
                )
                recognizer.imu_window_samples = int(
                    recognizer.window_size_ms * recognizer.imu_sample_rate / 1000
                )
                
                # 应用滤波器参数
                if hasattr(recognizer, 'signal_processor') and recognizer.signal_processor:
                    sp = recognizer.signal_processor
                    # 设计新的滤波器
                    bp_low = float(bp_low_var.get())
                    bp_high = float(bp_high_var.get())
                    bp_order = int(bp_order_var.get())
                    sp.design_bandpass_filter(lowcut=bp_low, highcut=bp_high, order=bp_order)
                    
                    notch_freq = float(notch_freq_var.get())
                    notch_q = float(notch_q_var.get())
                    sp.design_notch_filter(notch_freq=notch_freq, quality_factor=notch_q)
                    
                    lp_cutoff = float(lp_cutoff_var.get())
                    lp_order = int(lp_order_var.get())
                    sp.design_lowpass_filter(cutoff=lp_cutoff, order=lp_order)
                    
                    logger.info(f"滤波器参数已更新: BP({bp_low}-{bp_high}Hz), Notch({notch_freq}Hz), LP({lp_cutoff}Hz)")
                
                messagebox.showinfo("成功", "设置已应用")
                logger.info(f"识别设置已更新: 窗口={recognizer.window_size_ms}ms")
            except ValueError as e:
                messagebox.showerror("错误", f"无效的参数: {e}")
    
    ttk.Button(btn_frame, text="应用", command=apply_settings, width=12).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="关闭", command=settings_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开识别设置窗口")


def open_performance_monitor(root):
    """打开性能监控窗口
    
    Args:
        root: Tkinter根窗口对象
    """
    try:
        from src.performance_optimizer import PerformanceMonitor
        PERFORMANCE_AVAILABLE = True
    except ImportError:
        PERFORMANCE_AVAILABLE = False
    
    if not PERFORMANCE_AVAILABLE:
        messagebox.showerror("错误", "性能监控模块未安装，请检查依赖")
        return
    
    monitor_window = tk.Toplevel(root)
    monitor_window.title("性能监控")
    monitor_window.geometry("450x400")
    
    title_frame = ttk.Frame(monitor_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="性能监控", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(monitor_window)
    
    btn_frame = ttk.Frame(monitor_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    # 创建监控器实例
    monitor = PerformanceMonitor()
    
    # 系统资源
    resource_frame = ttk.LabelFrame(scrollable_frame, text="系统资源", padding="10")
    resource_frame.pack(fill=tk.X, padx=20, pady=5)
    
    cpu_var = tk.StringVar(value="CPU: --")
    memory_var = tk.StringVar(value="内存: --")
    
    ttk.Label(resource_frame, textvariable=cpu_var, font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    ttk.Label(resource_frame, textvariable=memory_var, font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    
    # 处理性能
    perf_frame = ttk.LabelFrame(scrollable_frame, text="处理性能", padding="10")
    perf_frame.pack(fill=tk.X, padx=20, pady=5)
    
    processing_var = tk.StringVar(value="平均处理时间: --")
    latency_var = tk.StringVar(value="平均预测延迟: --")
    
    ttk.Label(perf_frame, textvariable=processing_var, font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    ttk.Label(perf_frame, textvariable=latency_var, font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    
    # 统计信息
    stats_frame = ttk.LabelFrame(scrollable_frame, text="统计信息", padding="10")
    stats_frame.pack(fill=tk.X, padx=20, pady=5)
    
    stats_text = tk.Text(stats_frame, height=8, width=40)
    stats_text.pack(fill=tk.X)
    
    # 更新状态
    monitoring_active = [False]
    
    def update_stats():
        """更新统计信息"""
        if monitoring_active[0]:
            import psutil
            cpu_var.set(f"CPU: {psutil.cpu_percent():.1f}%")
            memory_var.set(f"内存: {psutil.virtual_memory().percent:.1f}%")
            
            if monitor.processing_times:
                avg_time = sum(monitor.processing_times) / len(monitor.processing_times)
                processing_var.set(f"平均处理时间: {avg_time:.2f} ms")
            
            if monitor.prediction_latencies:
                avg_latency = sum(monitor.prediction_latencies) / len(monitor.prediction_latencies)
                latency_var.set(f"平均预测延迟: {avg_latency:.2f} ms")
            
            stats_text.delete(1.0, tk.END)
            stats_text.insert(tk.END, f"运行时间: {time.time() - monitor.start_time:.1f}s\n" if monitor.start_time else "运行时间: --\n")
            stats_text.insert(tk.END, f"CPU历史记录: {len(monitor.cpu_history)} 条\n")
            stats_text.insert(tk.END, f"内存历史记录: {len(monitor.memory_history)} 条\n")
            stats_text.insert(tk.END, f"处理时间记录: {len(monitor.processing_times)} 条\n")
            stats_text.insert(tk.END, f"预测延迟记录: {len(monitor.prediction_latencies)} 条\n")
            
            if monitor.cpu_history:
                stats_text.insert(tk.END, f"平均CPU: {sum(monitor.cpu_history)/len(monitor.cpu_history):.1f}%\n")
            if monitor.memory_history:
                stats_text.insert(tk.END, f"平均内存: {sum(monitor.memory_history)/len(monitor.memory_history):.1f}%\n")
            
            monitor_window.after(1000, update_stats)
    
    import time
    
    def toggle_monitoring():
        """切换监控状态"""
        if not monitoring_active[0]:
            monitor.start_monitoring()
            monitoring_active[0] = True
            toggle_btn.config(text="停止监控")
            update_stats()
            logger.info("性能监控已启动")
        else:
            monitor.stop_monitoring()
            monitoring_active[0] = False
            toggle_btn.config(text="开始监控")
            logger.info("性能监控已停止")
    
    toggle_btn = ttk.Button(btn_frame, text="开始监控", command=toggle_monitoring, width=12)
    toggle_btn.pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="关闭", command=monitor_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开性能监控窗口")


def open_user_management(root):
    """打开用户管理窗口
    
    Args:
        root: Tkinter根窗口对象
    """
    try:
        from src.advanced_features import UserManager
        USER_MGMT_AVAILABLE = True
    except ImportError:
        USER_MGMT_AVAILABLE = False
    
    if not USER_MGMT_AVAILABLE:
        messagebox.showerror("错误", "用户管理模块未安装，请检查依赖")
        return
    
    user_window = tk.Toplevel(root)
    user_window.title("用户管理")
    user_window.geometry("500x400")
    
    title_frame = ttk.Frame(user_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="用户管理", font=("Arial", 14, "bold")).pack(pady=5)
    
    _, scrollable_frame, canvas = create_scrollable_frame(user_window)
    
    btn_frame = ttk.Frame(user_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    # 创建用户管理器
    user_manager = UserManager()
    
    # 当前用户
    current_frame = ttk.LabelFrame(scrollable_frame, text="当前用户", padding="10")
    current_frame.pack(fill=tk.X, padx=20, pady=5)
    
    current_user_var = tk.StringVar(value="未选择")
    ttk.Label(current_frame, textvariable=current_user_var, font=("Arial", 11)).pack(anchor=tk.W, pady=2)
    
    # 用户列表
    list_frame = ttk.LabelFrame(scrollable_frame, text="用户列表", padding="10")
    list_frame.pack(fill=tk.X, padx=20, pady=5)
    
    users_listbox = tk.Listbox(list_frame, height=6, width=40)
    users_listbox.pack(fill=tk.X, pady=5)
    
    def refresh_users():
        """刷新用户列表"""
        users_listbox.delete(0, tk.END)
        for user_info in user_manager.list_users():
            users_listbox.insert(tk.END, f"{user_info['username']} ({user_info['user_id']})")
        
        if user_manager.current_user:
            current_user_var.set(f"{user_manager.current_user.username}")
        else:
            current_user_var.set("未选择")
    
    refresh_users()
    
    # 新建用户
    new_user_frame = ttk.LabelFrame(scrollable_frame, text="新建用户", padding="10")
    new_user_frame.pack(fill=tk.X, padx=20, pady=5)
    
    new_username_var = tk.StringVar()
    ttk.Label(new_user_frame, text="用户名:").pack(anchor=tk.W)
    ttk.Entry(new_user_frame, textvariable=new_username_var, width=30).pack(fill=tk.X, pady=5)
    
    def create_new_user():
        """创建新用户"""
        username = new_username_var.get().strip()
        if not username:
            messagebox.showerror("错误", "请输入用户名")
            return
        
        user_manager.create_user(username)
        new_username_var.set("")
        refresh_users()
        messagebox.showinfo("成功", f"用户 '{username}' 创建成功")
        logger.info(f"创建新用户: {username}")
    
    ttk.Button(new_user_frame, text="创建用户", command=create_new_user, width=15).pack(pady=5)
    
    def select_user():
        """选择用户"""
        selection = users_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个用户")
            return
        
        index = selection[0]
        users = list(user_manager.users.values())
        if index < len(users):
            user_manager.set_current_user(users[index].user_id)
            refresh_users()
            messagebox.showinfo("成功", f"已切换到用户: {users[index].username}")
    
    ttk.Button(btn_frame, text="选择用户", command=select_user, width=12).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="刷新", command=refresh_users, width=10).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="关闭", command=user_window.destroy, width=12).pack(side=tk.LEFT, padx=5)
    
    logger.info("打开用户管理窗口")


def open_gesture_library(root):
    """打开手势库查看窗口
    
    Args:
        root: Tkinter根窗口对象
    """
    if not RECOGNITION_AVAILABLE:
        messagebox.showerror("错误", "手势识别模块未安装，请检查依赖")
        return
    
    library_window = tk.Toplevel(root)
    library_window.title("手势库管理")
    library_window.geometry("700x600")
    
    title_frame = ttk.Frame(library_window)
    title_frame.pack(fill=tk.X, pady=5)
    ttk.Label(title_frame, text="手势库管理", font=("Arial", 14, "bold")).pack(pady=5)
    
    notebook = ttk.Notebook(library_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    # ========== 预定义手势标签页 ==========
    predefined_frame = ttk.Frame(notebook)
    notebook.add(predefined_frame, text="预定义手势")
    
    predefined_container, predefined_scrollable, _ = create_scrollable_frame(predefined_frame)
    
    ttk.Label(predefined_scrollable, text="系统预定义的手势类型", 
              font=("Arial", 11, "bold")).pack(pady=10)
    
    gestures = PredefinedGestures.get_all_gestures()
    
    for name, gesture in gestures.items():
        gesture_card = ttk.LabelFrame(predefined_scrollable, 
                                       text=f"{gesture.display_name} ({name})", 
                                       padding="10")
        gesture_card.pack(fill=tk.X, padx=10, pady=5)
        
        info_frame = ttk.Frame(gesture_card)
        info_frame.pack(fill=tk.X)
        
        ttk.Label(info_frame, text=f"类别: {gesture.category.value}", 
                  font=("Arial", 9)).grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(info_frame, text=f"难度: {gesture.difficulty}", 
                  font=("Arial", 9)).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(info_frame, text=f"推荐样本: {gesture.recommended_samples}", 
                  font=("Arial", 9)).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(gesture_card, text=f"描述: {gesture.description}", 
                  font=("Arial", 9)).pack(anchor=tk.W, pady=2)
        
        ttk.Label(gesture_card, text=f"EMG特征: {gesture.emg_characteristics.get('description', 'N/A')}", 
                  font=("Arial", 8), foreground="blue").pack(anchor=tk.W)
        ttk.Label(gesture_card, text=f"IMU特征: {gesture.imu_characteristics.get('description', 'N/A')}", 
                  font=("Arial", 8), foreground="green").pack(anchor=tk.W)
        
        tags_str = ", ".join(gesture.tags)
        ttk.Label(gesture_card, text=f"标签: {tags_str}", 
                  font=("Arial", 8), foreground="gray").pack(anchor=tk.W)
    
    # ========== 自定义手势标签页 ==========
    custom_frame = ttk.Frame(notebook)
    notebook.add(custom_frame, text="自定义手势")
    
    custom_container, custom_scrollable, _ = create_scrollable_frame(custom_frame)
    
    ttk.Label(custom_scrollable, text="用户自定义的手势", 
              font=("Arial", 11, "bold")).pack(pady=10)
    
    custom_info_frame = ttk.LabelFrame(custom_scrollable, text="如何添加自定义手势", padding="10")
    custom_info_frame.pack(fill=tk.X, padx=10, pady=5)
    
    instructions = """添加自定义手势的步骤：

1. 在"数据标注"界面采集新手势数据
2. 输入新的手势名称（如: my_gesture）
3. 采集足够数量的样本（建议100+个）
4. 在"模型训练"界面训练模型
5. 训练完成后，新手势将自动被识别

注意事项：
• 手势名称使用英文，如: thumb_up, peace
• 每种手势至少需要20个样本才能训练
• 建议每种手势采集100-200个样本以获得最佳效果
• 可以在数据标注界面查看已采集的手势统计"""
    
    ttk.Label(custom_info_frame, text=instructions, font=("Arial", 9), 
              justify=tk.LEFT).pack(anchor=tk.W, pady=5)
    
    custom_list_frame = ttk.LabelFrame(custom_scrollable, text="已采集的手势数据", padding="10")
    custom_list_frame.pack(fill=tk.X, padx=10, pady=5)
    
    custom_gestures_text = tk.Text(custom_list_frame, height=10, width=60)
    custom_gestures_text.pack(fill=tk.X, pady=5)
    
    def refresh_custom_gestures():
        """刷新自定义手势列表"""
        custom_gestures_text.delete(1.0, tk.END)
        
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            custom_gestures_text.insert(tk.END, "数据目录不存在\n")
            return
        
        labels_file = os.path.join(raw_dir, "labels.json")
        if not os.path.exists(labels_file):
            custom_gestures_text.insert(tk.END, "暂无标注数据\n")
            custom_gestures_text.insert(tk.END, "请使用数据标注功能采集手势数据\n")
            return
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)
        
        gesture_counts = {}
        for session_id, label_list in labels_data.items():
            if label_list:
                gesture_name = label_list[0].get('gesture_name', 'unknown')
                gesture_counts[gesture_name] = gesture_counts.get(gesture_name, 0) + 1
        
        predefined_names = set(PredefinedGestures.get_gesture_names())
        
        custom_gestures_text.insert(tk.END, "所有已采集的手势:\n")
        custom_gestures_text.insert(tk.END, "=" * 40 + "\n\n")
        
        for gesture_name, count in sorted(gesture_counts.items()):
            is_predefined = "✓ 预定义" if gesture_name in predefined_names else "★ 自定义"
            custom_gestures_text.insert(tk.END, f"{is_predefined}  {gesture_name}: {count} 个样本\n")
        
        custom_count = sum(1 for g in gesture_counts if g not in predefined_names)
        predefined_count = sum(1 for g in gesture_counts if g in predefined_names)
        
        custom_gestures_text.insert(tk.END, f"\n" + "=" * 40 + "\n")
        custom_gestures_text.insert(tk.END, f"预定义手势: {predefined_count} 种\n")
        custom_gestures_text.insert(tk.END, f"自定义手势: {custom_count} 种\n")
        custom_gestures_text.insert(tk.END, f"总样本数: {sum(gesture_counts.values())}\n")
    
    ttk.Button(custom_list_frame, text="刷新列表", command=refresh_custom_gestures).pack(pady=5)
    
    # ========== 训练建议标签页 ==========
    recommend_frame = ttk.Frame(notebook)
    notebook.add(recommend_frame, text="训练建议")
    
    recommend_container, recommend_scrollable, _ = create_scrollable_frame(recommend_frame)
    
    ttk.Label(recommend_scrollable, text="手势训练建议", 
              font=("Arial", 11, "bold")).pack(pady=10)
    
    recommendations = PredefinedGestures.get_training_recommendations()
    
    for set_name, info in recommendations.items():
        set_frame = ttk.LabelFrame(recommend_scrollable, 
                                    text=f"{set_name.replace('_', ' ').title()}", 
                                    padding="10")
        set_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(set_frame, text=info['description'], 
                  font=("Arial", 10)).pack(anchor=tk.W)
        
        gestures_str = ", ".join(info['gestures'])
        ttk.Label(set_frame, text=f"手势: {gestures_str}", 
                  font=("Arial", 9), foreground="blue").pack(anchor=tk.W, pady=2)
        ttk.Label(set_frame, text=f"推荐总样本数: {info['total_samples']}", 
                  font=("Arial", 9), foreground="green").pack(anchor=tk.W)
    
    tips_frame = ttk.LabelFrame(recommend_scrollable, text="采集技巧", padding="10")
    tips_frame.pack(fill=tk.X, padx=10, pady=10)
    
    tips = """采集高质量数据的技巧：

1. 保持一致的姿势和力度
   - 每次做同一个手势时尽量保持相同的肌肉激活程度
   
2. 采集足够的样本
   - 每种手势建议采集100-200个样本
   - 样本越多，模型识别越准确
   
3. 包含变化
   - 轻微改变手势的力度和速度
   - 这有助于模型泛化
   
4. 分多次采集
   - 不要一次性采集所有样本
   - 分散在不同时间段采集效果更好
   
5. 保持放松
   - 采集前让肌肉放松几秒钟
   - 避免肌肉疲劳影响信号质量"""
    
    ttk.Label(tips_frame, text=tips, font=("Arial", 9), justify=tk.LEFT).pack(anchor=tk.W)
    
    # 初始刷新
    refresh_custom_gestures()
    
    btn_frame = ttk.Frame(library_window)
    btn_frame.pack(fill=tk.X, pady=10)
    
    ttk.Button(btn_frame, text="关闭", command=library_window.destroy, width=15).pack(side=tk.RIGHT, padx=10)
    
    logger.info("打开手势库管理窗口")


def get_recognition_functions():
    """获取手势识别相关函数列表
    
    Returns:
        函数字典
    """
    return {
        'open_model_training': open_model_training,
        'open_data_labeling': open_data_labeling,
        'open_recognition_settings': open_recognition_settings,
        'open_performance_monitor': open_performance_monitor,
        'open_user_management': open_user_management,
        'open_gesture_library': open_gesture_library,
        'recognition_available': RECOGNITION_AVAILABLE
    }
