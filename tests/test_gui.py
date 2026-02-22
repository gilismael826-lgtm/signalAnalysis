#!/usr/bin/env python3
"""
GUI功能测试
测试主窗口和各个GUI组件的基本功能
"""

import unittest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGUIImports(unittest.TestCase):
    """GUI模块导入测试"""
    
    def test_import_tkinter(self):
        """测试tkinter导入"""
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"tkinter不可用: {e}")
    
    def test_import_matplotlib(self):
        """测试matplotlib导入"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"matplotlib不可用: {e}")
    
    def test_import_main_window(self):
        """测试主窗口模块导入"""
        try:
            from src.main_window import MainWindow
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"主窗口模块不可用: {e}")


class TestGUIComponents(unittest.TestCase):
    """GUI组件测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        try:
            import tkinter as tk
            cls.tk = tk
            cls.root_available = True
        except ImportError:
            cls.root_available = False
    
    def setUp(self):
        """设置测试"""
        if not self.root_available:
            self.skipTest("tkinter不可用")
        
        # 创建测试用的根窗口（不显示）
        self.root = self.tk.Tk()
        self.root.withdraw()  # 隐藏窗口
    
    def tearDown(self):
        """清理测试"""
        if hasattr(self, 'root') and self.root:
            try:
                self.root.destroy()
            except:
                pass
    
    def test_button_creation(self):
        """测试按钮创建"""
        button = self.tk.Button(self.root, text="Test")
        self.assertIsNotNone(button)
    
    def test_label_creation(self):
        """测试标签创建"""
        label = self.tk.Label(self.root, text="Test Label")
        self.assertIsNotNone(label)
    
    def test_entry_creation(self):
        """测试输入框创建"""
        entry = self.tk.Entry(self.root)
        self.assertIsNotNone(entry)
    
    def test_frame_creation(self):
        """测试框架创建"""
        frame = self.tk.Frame(self.root)
        self.assertIsNotNone(frame)
    
    def test_canvas_creation(self):
        """测试画布创建"""
        canvas = self.tk.Canvas(self.root, width=400, height=300)
        self.assertIsNotNone(canvas)


class TestPlotDisplay(unittest.TestCase):
    """绘图显示测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            cls.plt = plt
            cls.np = np
            cls.matplotlib_available = True
        except ImportError:
            cls.matplotlib_available = False
    
    def test_figure_creation(self):
        """测试图形创建"""
        if not self.matplotlib_available:
            self.skipTest("matplotlib不可用")
        
        fig, axes = self.plt.subplots(2, 4)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(axes[0]), 4)
        self.plt.close(fig)
    
    def test_plot_data(self):
        """测试数据绘图"""
        if not self.matplotlib_available:
            self.skipTest("matplotlib不可用")
        
        fig, ax = self.plt.subplots()
        x = self.np.linspace(0, 10, 100)
        y = self.np.sin(x)
        line, = ax.plot(x, y)
        self.assertIsNotNone(line)
        self.plt.close(fig)
    
    def test_subplot_update(self):
        """测试子图更新"""
        if not self.matplotlib_available:
            self.skipTest("matplotlib不可用")
        
        fig, ax = self.plt.subplots()
        line, = ax.plot([], [])
        
        # 更新数据
        x = self.np.linspace(0, 10, 100)
        y = self.np.random.randn(100)
        line.set_data(x, y)
        
        self.assertEqual(len(line.get_xdata()), 100)
        self.plt.close(fig)


class TestSystemControlGUI(unittest.TestCase):
    """系统控制GUI测试"""
    
    def test_get_recognition_functions(self):
        """测试获取识别功能"""
        try:
            from src.system_control_module import get_recognition_functions
            funcs = get_recognition_functions()
            
            self.assertIn('open_model_training', funcs)
            self.assertIn('open_data_labeling', funcs)
            self.assertIn('open_recognition_settings', funcs)
            self.assertIn('recognition_available', funcs)
        except ImportError:
            self.skipTest("系统控制模块不可用")
    
    def test_model_training_window_creation(self):
        """测试模型训练窗口创建"""
        try:
            from src.system_control_module import open_model_training_window
            # 仅测试函数存在
            self.assertTrue(callable(open_model_training_window))
        except ImportError:
            self.skipTest("系统控制模块不可用")
    
    def test_data_labeling_window_creation(self):
        """测试数据标注窗口创建"""
        try:
            from src.system_control_module import open_data_labeling_window
            self.assertTrue(callable(open_data_labeling_window))
        except ImportError:
            self.skipTest("系统控制模块不可用")


class TestConfigGUI(unittest.TestCase):
    """配置GUI测试"""
    
    def test_config_values(self):
        """测试配置值"""
        try:
            import src.config as config
            
            self.assertEqual(config.EMG_SAMPLE_RATE, 250.0)
            self.assertEqual(config.IMU_SAMPLE_RATE, 104.0)
            self.assertIsInstance(config.SERIAL_PORT, str)
            self.assertIsInstance(config.BAUDRATE, int)
        except ImportError:
            self.skipTest("配置模块不可用")


class TestDataFlow(unittest.TestCase):
    """数据流测试"""
    
    def test_data_parser_flow(self):
        """测试数据解析流程"""
        try:
            from src.data_parser import parse_packet
            
            # 构造测试数据包
            header = b'\xd2\xd2\xd2'
            pkt_type = b'\xaa'
            reserved = b'\x00'
            
            emg_values = [100, -100, 500, -500, 1000, -1000, 2000, -2000]
            emg_data = b''
            for val in emg_values:
                if val < 0:
                    val = val + 0x1000000
                b0 = (val >> 16) & 0xFF
                b1 = (val >> 8) & 0xFF
                b2 = val & 0xFF
                emg_data += bytes([b0, b1, b2])
            
            padding = b'\x00' * (29 - 4 - len(emg_data))
            packet = header + pkt_type + reserved + emg_data + padding
            
            pkt_type_result, ts, data = parse_packet(packet)
            
            self.assertEqual(pkt_type_result, 'EMG')
            self.assertEqual(len(data), 8)
        except ImportError:
            self.skipTest("数据解析模块不可用")
    
    def test_feature_extraction_flow(self):
        """测试特征提取流程"""
        try:
            from src.feature_extractor import FeatureExtractor
            import numpy as np
            
            extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
            
            emg_data = np.random.randn(8, 150) * 1000
            imu_data = np.random.randn(6, 62) * 0.5
            
            features = extractor.extract_all_features(emg_data, imu_data)
            
            self.assertIsInstance(features, dict)
            self.assertGreater(len(features), 100)
        except ImportError:
            self.skipTest("特征提取模块不可用")


if __name__ == '__main__':
    unittest.main(verbosity=2)
