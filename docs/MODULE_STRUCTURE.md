# 模块化代码结构说明

## 📁 项目结构

```
signalAnalysis/
├── main.py                    # 主程序入口
├── src/                       # 源代码目录
│   ├── config.py              # 配置和全局变量
│   ├── data_manager.py        # 数据管理模块
│   ├── data_parser.py         # 数据解析模块
│   ├── serial_comm.py         # 串口通信模块
│   ├── plot_display.py        # 绘图显示模块
│   ├── zoom_control.py        # 缩放控制模块
│   ├── system_control_module.py # 系统控制模块
│   └── main_window.py        # 主界面模块
├── docs/                      # 文档目录
├── data/                      # 数据目录
└── logs/                      # 日志目录
```

## 📦 模块说明

### 1. config.py - 配置和全局变量
**功能**：
- 系统配置参数（串口、波特率、数据目录等）
- 全局变量定义（数据缓冲区、文件句柄、图表对象等）
- 缩放参数和预设配置
- 日志配置

**主要变量**：
- `SERIAL_PORT`：串口号
- `BAUDRATE`：波特率
- `SAVE_DATA`：是否保存数据
- `DATA_DIR`：数据目录
- `emg_buffer`：EMG数据缓冲区
- `imu_buffer`：IMU数据缓冲区
- `x_scale`：横轴缩放比例
- `emg_y_min/emg_y_max`：EMG纵轴范围
- `imu_y_min/imu_y_max`：IMU纵轴范围

### 2. data_manager.py - 数据管理模块
**功能**：
- 创建数据文件
- 保存数据点
- 轮换数据文件（循环存储）
- 删除旧数据文件
- 关闭数据文件

**主要函数**：
- `init_data_files()`：初始化数据文件
- `save_data_point()`：保存单个数据点
- `rotate_data_files()`：轮换数据文件
- `delete_old_data_files()`：删除旧数据文件
- `close_data_files()`：关闭数据文件

### 3. data_parser.py - 数据解析模块
**功能**：
- 解析接收到的数据包
- 提取EMG和IMU数据
- 数据类型识别

**主要函数**：
- `parse_packet()`：解析数据包，返回(数据类型, 时间戳, 数据)

### 4. serial_comm.py - 串口通信模块
**功能**：
- 串口连接管理
- 数据接收线程
- 数据包处理

**主要函数**：
- `serial_reader()`：串口数据接收线程

### 5. plot_display.py - 绘图显示模块
**功能**：
- 实时数据可视化
- 图表更新
- EMG和IMU数据显示

**主要函数**：
- `update_plot()`：更新图表显示

### 6. zoom_control.py - 缩放控制模块
**功能**：
- 横轴缩放控制
- EMG纵轴缩放控制
- IMU纵轴缩放控制
- 预设配置管理
- 缩放控制窗口

**主要函数**：
- `quick_zoom_in()`：快速放大
- `quick_zoom_out()`：快速缩小
- `quick_reset_zoom()`：快速重置
- `quick_emg_zoom_in()`：EMG纵轴放大
- `quick_emg_zoom_out()`：EMG纵轴缩小
- `quick_imu_zoom_in()`：IMU纵轴放大
- `quick_imu_zoom_out()`：IMU纵轴缩小
- `zoom_control()`：缩放控制窗口

### 7. system_control_module.py - 系统控制模块
**功能**：
- 数据采集控制
- 系统设置
- 数据分析
- 配置导出
- 文档查看

**主要函数**：
- `start_acquisition()`：开始数据采集
- `stop_acquisition()`：停止数据采集
- `clear_buffers()`：清空数据缓冲区
- `export_config()`：导出配置
- `analyze_data()`：数据分析
- `view_documentation()`：查看文档
- `system_settings()`：系统设置

### 8. main_window.py - 主界面模块
**功能**：
- 创建主窗口
- 整合所有模块
- GUI界面管理

**主要函数**：
- `create_main_window()`：创建主窗口
- `create_plots()`：创建图表
- `refresh_ports()`：刷新串口列表
- `list_serial_ports()`：列出可用串口
- `on_closing()`：窗口关闭处理

## 🔄 模块依赖关系

```
main.py
    ↓
main_window.py
    ├── config.py
    ├── system_control_module.py
    │   ├── config.py
    │   ├── data_manager.py
    │   │   └── config.py
    │   └── serial_comm.py
    │       ├── config.py
    │       ├── data_parser.py
    │       │   └── config.py
    │       └── data_manager.py
    ├── zoom_control.py
    │   └── config.py
    └── plot_display.py
        └── config.py
```

## 🚀 使用方法

### 运行程序
```bash
python main.py
```

### 添加新功能
1. 在对应的模块中添加新函数
2. 在 `main_window.py` 中添加UI控件
3. 在 `config.py` 中添加必要的全局变量

### 修改配置
直接修改 `config.py` 中的配置参数

## 💡 优势

1. **模块化**：每个模块功能单一，职责明确
2. **可维护性**：代码结构清晰，易于理解和修改
3. **可扩展性**：添加新功能时只需修改对应模块
4. **可测试性**：每个模块可以独立测试
5. **代码复用**：模块之间可以相互调用，避免重复代码

## 📝 注意事项

1. 所有全局变量都定义在 `config.py` 中
2. 修改全局变量时需要使用 `global` 关键字
3. 模块之间的依赖关系要清晰，避免循环依赖
4. 添加新功能时，要考虑模块的职责边界
5. 修改公共接口时要注意兼容性
