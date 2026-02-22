#!/usr/bin/env python3
"""
信号处理模块测试
测试滤波器设计和信号处理功能
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSignalProcessor(unittest.TestCase):
    """信号处理器测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        try:
            from src.signal_processor import SignalProcessor
            cls.SignalProcessor = SignalProcessor
            cls.module_available = True
        except ImportError:
            cls.module_available = False
    
    def setUp(self):
        """设置测试"""
        if not self.module_available:
            self.skipTest("信号处理模块不可用")
        
        self.processor = self.SignalProcessor(
            emg_sample_rate=250.0,
            imu_sample_rate=104.0,
            buffer_size=500
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.processor.emg_sample_rate, 250.0)
        self.assertEqual(self.processor.imu_sample_rate, 104.0)
        self.assertEqual(self.processor.buffer_size, 500)
        self.assertFalse(self.processor.enabled)
    
    def test_set_enabled(self):
        """测试启用/禁用预处理"""
        self.processor.set_enabled(True)
        self.assertTrue(self.processor.enabled)
        
        self.processor.set_enabled(False)
        self.assertFalse(self.processor.enabled)
    
    def test_set_filter_type(self):
        """测试设置滤波器类型"""
        for filter_type in ['bandpass', 'notch', 'both', 'none']:
            self.processor.set_filter_type(filter_type)
            self.assertEqual(self.processor.filter_type, filter_type)
    
    def test_set_normalize(self):
        """测试设置归一化"""
        self.processor.set_normalize(True)
        self.assertTrue(self.processor.normalize)
        
        self.processor.set_normalize(False)
        self.assertFalse(self.processor.normalize)
    
    def test_design_bandpass_filter(self):
        """测试带通滤波器设计"""
        b, a = self.processor.design_bandpass_filter(lowcut=20.0, highcut=100.0, order=4)
        
        self.assertIsNotNone(b)
        self.assertIsNotNone(a)
        self.assertEqual(len(b), 5)
        self.assertEqual(len(a), 5)
        
        self.assertIsNotNone(self.processor.bandpass_coeffs)
    
    def test_design_notch_filter(self):
        """测试陷波滤波器设计"""
        b, a = self.processor.design_notch_filter(notch_freq=50.0, quality_factor=30.0)
        
        self.assertIsNotNone(b)
        self.assertIsNotNone(a)
        self.assertEqual(len(b), 3)
        self.assertEqual(len(a), 3)
        
        self.assertIsNotNone(self.processor.notch_coeffs)
    
    def test_design_lowpass_filter(self):
        """测试低通滤波器设计"""
        b, a = self.processor.design_lowpass_filter(cutoff=10.0, order=4)
        
        self.assertIsNotNone(b)
        self.assertIsNotNone(a)
        self.assertEqual(len(b), 5)
        self.assertEqual(len(a), 5)
        
        self.assertIsNotNone(self.processor.lowpass_coeffs)
    
    def test_apply_bandpass_filter(self):
        """测试带通滤波"""
        t = np.linspace(0, 1, 250)
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 5 * t)
        
        filtered = self.processor.apply_bandpass_filter(signal)
        
        self.assertEqual(len(filtered), len(signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_apply_notch_filter(self):
        """测试陷波滤波"""
        t = np.linspace(0, 1, 250)
        signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 30 * t)
        
        filtered = self.processor.apply_notch_filter(signal)
        
        self.assertEqual(len(filtered), len(signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_apply_lowpass_filter(self):
        """测试低通滤波"""
        t = np.linspace(0, 1, 104)
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)
        
        filtered = self.processor.apply_lowpass_filter(signal)
        
        self.assertEqual(len(filtered), len(signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_apply_moving_average(self):
        """测试移动平均"""
        signal = np.random.randn(100)
        
        smoothed = self.processor.apply_moving_average(signal, window_size=5)
        
        self.assertEqual(len(smoothed), len(signal))
        
        signal_var = np.var(signal)
        smoothed_var = np.var(smoothed)
        self.assertLess(smoothed_var, signal_var)
    
    def test_apply_normalization(self):
        """测试归一化"""
        signal = np.array([1, 2, 3, 4, 5])
        
        normalized = self.processor.apply_normalization(signal)
        
        self.assertAlmostEqual(np.min(normalized), -1, places=5)
        self.assertAlmostEqual(np.max(normalized), 1, places=5)
    
    def test_detrend(self):
        """测试去趋势"""
        t = np.linspace(0, 1, 100)
        signal = 2 * t + np.sin(2 * np.pi * 5 * t)
        
        detrended = self.processor.detrend(signal)
        
        trend = np.polyfit(np.arange(len(detrended)), detrended, 1)[0]
        self.assertLess(abs(trend), 0.1)
    
    def test_process_emg_data(self):
        """测试EMG数据处理"""
        self.processor.set_enabled(True)
        self.processor.set_filter_type('bandpass')
        
        emg_data = [np.random.randn(100) for _ in range(8)]
        
        processed = self.processor.process_emg_data(emg_data)
        
        self.assertEqual(len(processed), 8)
        for channel in processed:
            self.assertEqual(len(channel), 100)
    
    def test_process_imu_data(self):
        """测试IMU数据处理"""
        self.processor.set_enabled(True)
        
        imu_data = [np.random.randn(100) for _ in range(6)]
        
        processed = self.processor.process_imu_data(imu_data)
        
        self.assertEqual(len(processed), 6)
        for channel in processed:
            self.assertEqual(len(channel), 100)
    
    def test_process_disabled(self):
        """测试禁用预处理时的处理"""
        self.processor.set_enabled(False)
        
        emg_data = [np.random.randn(100) for _ in range(8)]
        processed = self.processor.process_emg_data(emg_data)
        
        self.assertEqual(len(processed), 8)
        for i, channel in enumerate(processed):
            np.testing.assert_array_almost_equal(channel, emg_data[i])
    
    def test_update_buffers(self):
        """测试缓冲区更新"""
        self.processor.set_enabled(True)
        self.processor.set_filter_type('bandpass')
        
        for _ in range(50):
            emg_data = [np.random.randn() for _ in range(8)]
            imu_data = [np.random.randn() for _ in range(6)]
            self.processor.update_buffers(emg_data, imu_data)
        
        self.assertEqual(len(self.processor.emg_filter_buffers[0]), 50)
        self.assertEqual(len(self.processor.imu_filter_buffers[0]), 50)
    
    def test_buffer_size_limit(self):
        """测试缓冲区大小限制"""
        self.processor.set_enabled(True)
        
        for _ in range(600):
            emg_data = [np.random.randn() for _ in range(8)]
            self.processor.update_buffers(emg_data, [])
        
        for buffer in self.processor.emg_filter_buffers:
            self.assertLessEqual(len(buffer), self.processor.buffer_size)


class TestFilterCharacteristics(unittest.TestCase):
    """滤波器特性测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试类"""
        try:
            from src.signal_processor import SignalProcessor
            cls.SignalProcessor = SignalProcessor
            cls.module_available = True
        except ImportError:
            cls.module_available = False
    
    def setUp(self):
        """设置测试"""
        if not self.module_available:
            self.skipTest("信号处理模块不可用")
        
        self.processor = self.SignalProcessor()
    
    def test_bandpass_attenuation(self):
        """测试带通滤波器衰减特性"""
        self.processor.design_bandpass_filter(lowcut=20.0, highcut=100.0)
        
        t = np.linspace(0, 2, 500)
        
        low_freq_signal = np.sin(2 * np.pi * 5 * t)
        filtered_low = self.processor.apply_bandpass_filter(low_freq_signal)
        low_attenuation = np.std(filtered_low) / np.std(low_freq_signal)
        
        self.assertLess(low_attenuation, 0.5)
        
        high_freq_signal = np.sin(2 * np.pi * 150 * t)
        filtered_high = self.processor.apply_bandpass_filter(high_freq_signal)
        high_attenuation = np.std(filtered_high) / np.std(high_freq_signal)
        
        self.assertLess(high_attenuation, 0.5)
    
    def test_notch_filter_characteristic(self):
        """测试陷波滤波器特性"""
        self.processor.design_notch_filter(notch_freq=50.0)
        
        t = np.linspace(0, 2, 500)
        
        notch_signal = np.sin(2 * np.pi * 50 * t)
        filtered_notch = self.processor.apply_notch_filter(notch_signal)
        notch_attenuation = np.std(filtered_notch) / np.std(notch_signal)
        
        self.assertLess(notch_attenuation, 0.3)
        
        pass_signal = np.sin(2 * np.pi * 30 * t)
        filtered_pass = self.processor.apply_notch_filter(pass_signal)
        pass_ratio = np.std(filtered_pass) / np.std(pass_signal)
        
        self.assertGreater(pass_ratio, 0.7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
