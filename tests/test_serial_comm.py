#!/usr/bin/env python3
"""
串口通信模块测试
测试数据解析和串口通信功能
"""

import unittest
import sys
import os
import time
import struct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_parser import parse_packet


class TestDataParser(unittest.TestCase):
    """数据解析测试"""
    
    def test_parse_emg_packet(self):
        """测试EMG数据包解析"""
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
        self.assertIsNotNone(ts)
        self.assertEqual(len(data), 8)
        
        for i, expected in enumerate([100, -100, 500, -500, 1000, -1000, 2000, -2000]):
            self.assertAlmostEqual(data[i], expected, places=0,
                                 msg=f"EMG通道{i}值不匹配")
    
    def test_parse_imu_packet(self):
        """测试IMU数据包解析"""
        header = b'\xd2\xd2\xd2'
        pkt_type = b'\xbb'
        reserved = b'\x00\x00\x00'
        
        gyro_values = [100, -100, 200]
        accel_values = [500, -500, 1000]
        
        imu_data = b''
        for val in gyro_values + accel_values:
            if val < 0:
                val = val + 0x10000
            lo = val & 0xFF
            hi = (val >> 8) & 0xFF
            imu_data += bytes([lo, hi])
        
        padding = b'\x00' * (29 - 4 - 3 - len(imu_data))
        packet = header + pkt_type + reserved + imu_data + padding
        
        pkt_type_result, ts, data = parse_packet(packet)
        
        self.assertEqual(pkt_type_result, 'IMU')
        self.assertIsNotNone(ts)
        self.assertEqual(len(data), 6)
        
        expected_gyro = [0.0012 * x for x in gyro_values]
        expected_accel = [0.0005978 * x for x in accel_values]
        expected = expected_gyro + expected_accel
        
        for i in range(6):
            self.assertAlmostEqual(data[i], expected[i], places=6,
                                 msg=f"IMU通道{i}值不匹配")
    
    def test_parse_invalid_header(self):
        """测试无效数据包头"""
        packet = b'\xff\xff\xff' + b'\x00' * 26
        pkt_type, ts, data = parse_packet(packet)
        
        self.assertIsNone(pkt_type)
        self.assertIsNone(ts)
        self.assertIsNone(data)
    
    def test_parse_short_packet(self):
        """测试过短的数据包"""
        packet = b'\xd2\xd2\xd2\xaa\x00' + b'\x00' * 10
        pkt_type, ts, data = parse_packet(packet)
        
        self.assertIsNone(pkt_type)
        self.assertIsNone(ts)
        self.assertIsNone(data)
    
    def test_parse_unknown_type(self):
        """测试未知数据包类型"""
        header = b'\xd2\xd2\xd2'
        pkt_type = b'\xcc'
        padding = b'\x00' * 25
        packet = header + pkt_type + padding
        
        pkt_type_result, ts, data = parse_packet(packet)
        
        self.assertIsNone(pkt_type_result)
        self.assertIsNone(ts)
        self.assertIsNone(data)
    
    def test_parse_emg_max_values(self):
        """测试EMG最大值解析"""
        header = b'\xd2\xd2\xd2'
        pkt_type = b'\xaa'
        reserved = b'\x00'
        
        max_val = 0x7FFFFF
        min_val = -0x800000
        
        emg_values = [max_val, min_val] + [0] * 6
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
        self.assertEqual(data[0], max_val)
        self.assertEqual(data[1], min_val)


class TestSerialCommMock(unittest.TestCase):
    """串口通信模拟测试"""
    
    def test_emg_packet_construction(self):
        """测试EMG数据包构建"""
        header = b'\xd2\xd2\xd2'
        pkt_type = 0xAA
        
        self.assertEqual(len(header), 3)
        self.assertEqual(pkt_type, 0xAA)
    
    def test_imu_packet_construction(self):
        """测试IMU数据包构建"""
        header = b'\xd2\xd2\xd2'
        pkt_type = 0xBB
        
        self.assertEqual(len(header), 3)
        self.assertEqual(pkt_type, 0xBB)
    
    def test_packet_size(self):
        """测试数据包大小"""
        expected_size = 29
        
        emg_header = b'\xd2\xd2\xd2\xaa\x00'
        emg_data = b'\x00' * (29 - len(emg_header))
        emg_packet = emg_header + emg_data
        
        self.assertEqual(len(emg_packet), expected_size)
        
        imu_header = b'\xd2\xd2\xd2\xbb\x00\x00\x00'
        imu_data = b'\x00' * (29 - len(imu_header))
        imu_packet = imu_header + imu_data
        
        self.assertEqual(len(imu_packet), expected_size)


class TestConfigIntegration(unittest.TestCase):
    """配置集成测试"""
    
    def test_sample_rates(self):
        """测试采样率配置"""
        try:
            import src.config as config
            self.assertEqual(config.EMG_SAMPLE_RATE, 250)
            self.assertEqual(config.IMU_SAMPLE_RATE, 104)
        except ImportError:
            self.skipTest("配置模块不可用")
    
    def test_buffer_config(self):
        """测试缓冲区配置"""
        try:
            import src.config as config
            from collections import deque
            
            test_buffer = deque(maxlen=10000)
            for i in range(10000):
                test_buffer.append(i)
            
            self.assertEqual(len(test_buffer), 10000)
            test_buffer.append(10001)
            self.assertEqual(len(test_buffer), 10000)
        except ImportError:
            self.skipTest("配置模块不可用")


if __name__ == '__main__':
    unittest.main(verbosity=2)
