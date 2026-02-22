#!/usr/bin/env python3
"""
主界面模块
整合所有模块，提供完整的GUI界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import src.config as config
from src.config import (
    x_scale, emg_y_min, emg_y_max, imu_y_min, imu_y_max,
    SAVE_DATA, BAUDRATE, logger, PREPROCESSING_ENABLED, 
    PREPROCESSING_FILTER_TYPE, PREPROCESSING_NORMALIZE, PREPROCESSING_DISPLAY_MODE,
    processed_emg_y_min, processed_emg_y_max, processed_imu_y_min, processed_imu_y_max
)
from src.system_control_module import (
    start_acquisition, stop_acquisition, clear_buffers,
    export_config, analyze_data, view_documentation, system_settings,
    open_model_training, open_data_labeling, open_recognition_settings,
    open_performance_monitor, open_user_management, open_gesture_library
)
from src.zoom_control import (
    quick_zoom_in, quick_zoom_out, quick_reset_zoom,
    quick_emg_zoom_in_current, quick_emg_zoom_out_current,
    quick_imu_zoom_in_current, quick_imu_zoom_out_current, zoom_control,
    update_zoom_info_display
)
from src.plot_display import update_plot, init_plot

# 手势识别相关导入
try:
    from src.realtime_recognizer import RealtimeGestureRecognizer
    from src.gesture_classifier import GestureClassifier, GestureClassifierManager
    RECOGNIZER_AVAILABLE = True
except ImportError:
    RECOGNIZER_AVAILABLE = False
    logger.warning("手势识别模块未完全加载")

# 全局变量
tree = None
axs = []
canvas = None

# 手势识别相关全局变量
recognizer = None
classifier_manager = None
recognition_enabled = False
gesture_label_var = None
confidence_label_var = None
recognition_status_var = None
ensemble_strategy_var = None
model_info_var = None


def create_main_window():
    """创建主窗口"""
    global root, port_var, save_var, status_var, status_label, start_btn, stop_btn
    global zoom_info_var, emg_zoom_info_var, imu_zoom_info_var, port_combobox, axs, canvas
    
    root = tk.Tk()
    root.title("EMG/IMU 系统控制中心")
    root.geometry("1500x900")
    root.minsize(1200, 700)
    
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 左侧控制面板 - 使用Canvas和Scrollbar实现滚动
    control_container = ttk.Frame(main_frame, width=320)
    control_container.pack(side=tk.LEFT, fill=tk.Y, padx=5)
    control_container.pack_propagate(False)
    
    # 创建Canvas和Scrollbar
    control_canvas = tk.Canvas(control_container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
    control_panel = ttk.Frame(control_canvas)
    
    control_panel.bind(
        "<Configure>",
        lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
    )
    
    control_canvas.create_window((0, 0), window=control_panel, anchor="nw")
    control_canvas.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # 绑定鼠标滚轮
    def on_mousewheel(event):
        control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    control_canvas.bind_all("<MouseWheel>", on_mousewheel)
    
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
    ttk.Button(emg_zoom_frame, text="放大", command=lambda: quick_emg_zoom_in_current(emg_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(emg_zoom_frame, text="缩小", command=lambda: quick_emg_zoom_out_current(emg_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    
    emg_zoom_info_var = tk.StringVar(value=f"EMG: {config.emg_y_min}~{config.emg_y_max} μV")
    ttk.Label(quick_zoom_frame, textvariable=emg_zoom_info_var).pack(pady=2)
    
    imu_zoom_frame = ttk.Frame(quick_zoom_frame)
    imu_zoom_frame.pack(fill=tk.X, pady=5)
    ttk.Label(imu_zoom_frame, text="IMU纵轴:", width=8).pack(side=tk.LEFT)
    ttk.Button(imu_zoom_frame, text="放大", command=lambda: quick_imu_zoom_in_current(imu_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    ttk.Button(imu_zoom_frame, text="缩小", command=lambda: quick_imu_zoom_out_current(imu_zoom_info_var), width=8).pack(side=tk.LEFT, padx=2)
    
    imu_zoom_info_var = tk.StringVar(value=f"IMU: {config.imu_y_min}~{config.imu_y_max}")
    ttk.Label(quick_zoom_frame, textvariable=imu_zoom_info_var).pack(pady=2)
    
    preprocessing_frame = ttk.LabelFrame(control_panel, text="信号预处理", padding="10")
    preprocessing_frame.pack(fill=tk.X, pady=5)
    
    # 预处理开关
    preprocessing_enable_var = tk.BooleanVar(value=config.PREPROCESSING_ENABLED)
    def toggle_preprocessing():
        config.PREPROCESSING_ENABLED = preprocessing_enable_var.get()
        from src.signal_processor import signal_processor
        signal_processor.set_enabled(config.PREPROCESSING_ENABLED)
        logger.info(f"预处理已{'启用' if config.PREPROCESSING_ENABLED else '禁用'}")
    
    ttk.Checkbutton(preprocessing_frame, text="启用预处理", variable=preprocessing_enable_var, 
                 command=toggle_preprocessing).pack(anchor=tk.W, pady=2)
    
    # 滤波器类型选择
    ttk.Label(preprocessing_frame, text="滤波器类型:").pack(anchor=tk.W, pady=2)
    filter_type_var = tk.StringVar(value=config.PREPROCESSING_FILTER_TYPE)
    def change_filter_type():
        config.PREPROCESSING_FILTER_TYPE = filter_type_var.get()
        from src.signal_processor import signal_processor
        signal_processor.set_filter_type(config.PREPROCESSING_FILTER_TYPE)
        logger.info(f"滤波器类型: {config.PREPROCESSING_FILTER_TYPE}")
    
    filter_frame = ttk.Frame(preprocessing_frame)
    filter_frame.pack(fill=tk.X, pady=2)
    ttk.Radiobutton(filter_frame, text="带通滤波", variable=filter_type_var, 
                  value='bandpass', command=change_filter_type).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(filter_frame, text="陷波滤波", variable=filter_type_var, 
                  value='notch', command=change_filter_type).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(filter_frame, text="双重滤波", variable=filter_type_var, 
                  value='both', command=change_filter_type).pack(side=tk.LEFT, padx=5)
    
    # 归一化开关
    normalize_var = tk.BooleanVar(value=config.PREPROCESSING_NORMALIZE)
    def toggle_normalize():
        config.PREPROCESSING_NORMALIZE = normalize_var.get()
        from src.signal_processor import signal_processor
        signal_processor.set_normalize(config.PREPROCESSING_NORMALIZE)
        logger.info(f"归一化已{'启用' if config.PREPROCESSING_NORMALIZE else '禁用'}")
    
    ttk.Checkbutton(preprocessing_frame, text="归一化", variable=normalize_var, 
                 command=toggle_normalize).pack(anchor=tk.W, pady=2)
    
    # 显示模式切换
    ttk.Label(preprocessing_frame, text="显示模式:").pack(anchor=tk.W, pady=2)
    display_mode_var = tk.StringVar(value=config.PREPROCESSING_DISPLAY_MODE)
    def change_display_mode():
        config.PREPROCESSING_DISPLAY_MODE = display_mode_var.get()
        from src.signal_processor import signal_processor
        
        # 当切换到预处理信号模式时，自动启用预处理
        if config.PREPROCESSING_DISPLAY_MODE == 'processed':
            config.PREPROCESSING_ENABLED = True
            preprocessing_enable_var.set(True)
            signal_processor.set_enabled(True)
            logger.info("已自动启用预处理功能")
        
        # 更新缩放信息显示
        update_zoom_info_display(emg_zoom_info_var, imu_zoom_info_var)
        
        logger.info(f"显示模式: {config.PREPROCESSING_DISPLAY_MODE}")
    
    display_frame = ttk.Frame(preprocessing_frame)
    display_frame.pack(fill=tk.X, pady=2)
    ttk.Radiobutton(display_frame, text="原始信号", variable=display_mode_var, 
                  value='raw', command=change_display_mode).pack(side=tk.LEFT, padx=5)
    ttk.Radiobutton(display_frame, text="预处理信号", variable=display_mode_var, 
                  value='processed', command=change_display_mode).pack(side=tk.LEFT, padx=5)
    
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
    ttk.Button(tools_frame, text="性能监控", command=lambda: open_performance_monitor(root), width=25).pack(pady=5)
    ttk.Button(tools_frame, text="用户管理", command=lambda: open_user_management(root), width=25).pack(pady=5)
    ttk.Button(tools_frame, text="手势库管理", command=lambda: open_gesture_library(root), width=25).pack(pady=5)
    
    # 手势识别模块
    if RECOGNIZER_AVAILABLE:
        create_recognition_panel(control_panel)
    
    plot_frame = ttk.LabelFrame(main_frame, text="实时数据", padding="10")
    plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
    
    create_plots(plot_frame)
    
    info_frame = ttk.Frame(root, padding="10")
    info_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(info_frame, text="EMG 通道: 8个 (单位: μV)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text="IMU 通道: 6个 (单位: rad/s, m/s²)").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"波特率: {config.BAUDRATE}").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"横轴缩放: {config.x_scale} 点").pack(side=tk.LEFT, padx=10)
    ttk.Label(info_frame, text=f"显示模式: {'原始' if config.PREPROCESSING_DISPLAY_MODE == 'raw' else '预处理'}").pack(side=tk.LEFT, padx=10)
    
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
    
    # 初始化图表对象（set_data模式）
    init_plot(axs)


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
    global recognizer
    if config.running:
        stop_acquisition(status_var, status_label, start_btn, stop_btn)
    if recognizer:
        recognizer.stop()
    root.destroy()


# ==================== 手势识别功能 ====================

def create_recognition_panel(parent):
    """创建手势识别面板（自动集成模式）
    
    Args:
        parent: 父容器
    """
    global gesture_label_var, confidence_label_var, recognition_status_var
    global recognizer, classifier_manager, recognition_enabled
    global ensemble_strategy_var, model_info_var
    
    recognition_frame = ttk.LabelFrame(parent, text="手势识别 (智能集成)", padding="10")
    recognition_frame.pack(fill=tk.X, pady=5)
    
    status_frame = ttk.Frame(recognition_frame)
    status_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(status_frame, text="状态:", width=8).pack(side=tk.LEFT)
    recognition_status_var = tk.StringVar(value="未初始化")
    status_label = ttk.Label(status_frame, textvariable=recognition_status_var, 
              foreground="gray", font=("Arial", 10))
    status_label.pack(side=tk.LEFT)
    
    gesture_frame = ttk.Frame(recognition_frame)
    gesture_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(gesture_frame, text="手势:", width=8).pack(side=tk.LEFT)
    gesture_label_var = tk.StringVar(value="--")
    gesture_label = ttk.Label(gesture_frame, textvariable=gesture_label_var, 
                              font=("Arial", 16, "bold"), foreground="blue")
    gesture_label.pack(side=tk.LEFT, padx=10)
    
    confidence_frame = ttk.Frame(recognition_frame)
    confidence_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(confidence_frame, text="置信度:", width=8).pack(side=tk.LEFT)
    confidence_label_var = tk.StringVar(value="--")
    ttk.Label(confidence_frame, textvariable=confidence_label_var, 
              font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
    
    model_info_frame = ttk.Frame(recognition_frame)
    model_info_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(model_info_frame, text="模型:", width=8).pack(side=tk.LEFT)
    model_info_var = tk.StringVar(value="自动加载中...")
    ttk.Label(model_info_frame, textvariable=model_info_var, 
              font=("Arial", 9), foreground="green").pack(side=tk.LEFT, padx=5)
    
    strategy_frame = ttk.Frame(recognition_frame)
    strategy_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(strategy_frame, text="策略:", width=8).pack(side=tk.LEFT)
    ensemble_strategy_var = tk.StringVar(value="weighted_voting")
    
    strategy_combo = ttk.Combobox(strategy_frame, textvariable=ensemble_strategy_var, 
                                   width=15, state="readonly")
    strategy_combo['values'] = ('weighted_voting', 'adaptive', 'best_only')
    strategy_combo.pack(side=tk.LEFT, padx=5)
    
    def on_strategy_change(event):
        if recognizer:
            recognizer.set_ensemble_strategy(ensemble_strategy_var.get())
            logger.info(f"集成策略已切换: {ensemble_strategy_var.get()}")
    
    strategy_combo.bind('<<ComboboxSelected>>', on_strategy_change)
    
    ttk.Label(strategy_frame, text="(", font=("Arial", 8)).pack(side=tk.LEFT)
    ttk.Label(strategy_frame, text="加权投票|自适应|最优", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
    ttk.Label(strategy_frame, text=")", font=("Arial", 8)).pack(side=tk.LEFT)
    
    btn_frame = ttk.Frame(recognition_frame)
    btn_frame.pack(fill=tk.X, pady=5)
    
    enable_btn_var = tk.StringVar(value="启用识别")
    
    def toggle_recognition():
        global recognition_enabled, recognizer, classifier_manager
        
        if not recognition_enabled:
            if recognizer is None:
                messagebox.showwarning("警告", "识别器未初始化")
                return
            
            if recognizer.classifier_manager is None and recognizer.classifier is None:
                messagebox.showwarning("警告", "请先训练模型或加载模型目录")
                return
            
            recognizer.start()
            recognition_enabled = True
            enable_btn_var.set("禁用识别")
            recognition_status_var.set("运行中")
            status_label.config(foreground="green")
            logger.info("手势识别已启用 (智能集成模式)")
        else:
            if recognizer:
                recognizer.stop()
            recognition_enabled = False
            enable_btn_var.set("启用识别")
            recognition_status_var.set("已停止")
            status_label.config(foreground="red")
            logger.info("手势识别已禁用")
    
    enable_btn = ttk.Button(btn_frame, textvariable=enable_btn_var, 
                           command=toggle_recognition, width=12)
    enable_btn.pack(side=tk.LEFT, padx=5)
    
    ttk.Button(btn_frame, text="刷新模型", 
               command=refresh_models, width=10).pack(side=tk.LEFT, padx=2)
    
    window_frame = ttk.Frame(recognition_frame)
    window_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(window_frame, text="窗口(ms):", width=8).pack(side=tk.LEFT)
    window_size_var = tk.StringVar(value="600")
    window_entry = ttk.Entry(window_frame, textvariable=window_size_var, width=8)
    window_entry.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(window_frame, text="阈值:", width=5).pack(side=tk.LEFT)
    threshold_var = tk.StringVar(value="0.5")
    threshold_entry = ttk.Entry(window_frame, textvariable=threshold_var, width=5)
    threshold_entry.pack(side=tk.LEFT, padx=5)
    
    def apply_settings():
        global recognizer
        try:
            window_ms = int(window_size_var.get())
            threshold = float(threshold_var.get())
            if recognizer:
                recognizer.window_size_ms = window_ms
                recognizer.emg_window_samples = int(window_ms * config.EMG_SAMPLE_RATE / 1000)
                recognizer.imu_window_samples = int(window_ms * config.IMU_SAMPLE_RATE / 1000)
                recognizer.confidence_threshold = threshold
                logger.info(f"设置已更新: 窗口={window_ms}ms, 阈值={threshold}")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
    
    ttk.Button(window_frame, text="应用", command=apply_settings, width=6).pack(side=tk.LEFT, padx=2)
    
    func_btn_frame = ttk.Frame(recognition_frame)
    func_btn_frame.pack(fill=tk.X, pady=5)
    
    ttk.Button(func_btn_frame, text="模型训练", 
               command=lambda: open_model_training(root), width=10).pack(side=tk.LEFT, padx=2)
    ttk.Button(func_btn_frame, text="数据标注", 
               command=lambda: open_data_labeling(root), width=10).pack(side=tk.LEFT, padx=2)
    ttk.Button(func_btn_frame, text="高级设置", 
               command=lambda: open_recognition_settings(root, recognizer), width=10).pack(side=tk.LEFT, padx=2)
    
    info_label = ttk.Label(recognition_frame, 
                          text="提示: 系统自动训练所有模型并选择最优，无需手动选择",
                          font=("Arial", 8), foreground="gray")
    info_label.pack(pady=2)
    
    init_recognizer()
    
    logger.info("手势识别面板已创建 (智能集成模式)")


def init_recognizer():
    """初始化识别器（自动加载集成模型）"""
    global recognizer, classifier_manager, model_info_var, recognition_status_var
    
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        slide_step=25,
        smoothing_window=5,
        models_dir='models',
        ensemble_strategy='weighted_voting',
        use_ensemble=True
    )
    recognizer.set_prediction_callback(on_prediction_result)
    
    models_dir = 'models'
    if os.path.exists(models_dir):
        pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        if pkl_files:
            try:
                recognizer.load_ensemble_models(models_dir)
                if recognizer.classifier_manager:
                    n_models = len(recognizer.classifier_manager.classifiers)
                    best_algo = recognizer.classifier_manager.best_algorithm or "未知"
                    model_info_var.set(f"{n_models}个模型, 最优: {best_algo.upper()}")
                    recognition_status_var.set("就绪")
                    logger.info(f"自动加载集成模型成功: {n_models}个模型")
                else:
                    model_info_var.set("无可用模型")
                    recognition_status_var.set("需训练")
            except Exception as e:
                model_info_var.set("加载失败")
                recognition_status_var.set("错误")
                logger.error(f"自动加载模型失败: {e}")
        else:
            model_info_var.set("无模型文件")
            recognition_status_var.set("需训练")
    else:
        model_info_var.set("模型目录不存在")
        recognition_status_var.set("需初始化")


def refresh_models():
    """刷新模型（重新加载）"""
    global recognizer, model_info_var, recognition_status_var
    
    if recognizer:
        try:
            recognizer.load_ensemble_models('models')
            if recognizer.classifier_manager:
                n_models = len(recognizer.classifier_manager.classifiers)
                best_algo = recognizer.classifier_manager.best_algorithm or "未知"
                model_info_var.set(f"{n_models}个模型, 最优: {best_algo.upper()}")
                recognition_status_var.set("就绪")
                messagebox.showinfo("成功", f"已加载 {n_models} 个模型\n最优模型: {best_algo.upper()}")
            else:
                model_info_var.set("无可用模型")
                messagebox.showwarning("警告", "没有找到可用的模型文件")
        except Exception as e:
            model_info_var.set("加载失败")
            messagebox.showerror("错误", f"模型加载失败: {e}")
            logger.error(f"刷新模型失败: {e}")


def on_prediction_result(result):
    """预测结果回调
    
    Args:
        result: 预测结果字典
    """
    global gesture_label_var, confidence_label_var
    
    gesture_label_var.set(result['gesture'])
    confidence_label_var.set(f"{result['confidence']:.1%}")
    
    if result.get('method') == 'weighted_voting':
        participating = result.get('participating_models', [])
        logger.debug(f"集成预测: {result['gesture']} ({result['confidence']:.1%}), "
                    f"参与模型: {len(participating)}个")


def update_recognition_with_data(emg_data, imu_data):
    """使用数据更新识别器（供外部调用）
    
    Args:
        emg_data: EMG数据
        imu_data: IMU数据
        
    Returns:
        预测结果或None
    """
    global recognizer, recognition_enabled
    
    if recognition_enabled and recognizer:
        return recognizer.update(emg_data, imu_data)
    return None
