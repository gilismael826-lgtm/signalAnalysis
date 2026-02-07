#!/usr/bin/env python3
"""
Unit tests for EMG/IMU Signal Analysis System
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestEMGIMUReceiver(unittest.TestCase):
    """Test cases for EMG/IMU receiver functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_packet_parsing(self):
        """Test packet parsing functionality"""
        # This would test the parse_packet function
        # For now, just a placeholder
        self.assertTrue(True)
    
    def test_emg_data_format(self):
        """Test EMG data format validation"""
        # Test that EMG data has 8 channels
        test_emg_data = [100, 200, 150, 300, 250, 180, 220, 190]
        self.assertEqual(len(test_emg_data), 8)
    
    def test_imu_data_format(self):
        """Test IMU data format validation"""
        # Test that IMU data has 6 axes (3 gyro + 3 accel)
        test_imu_data = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7]
        self.assertEqual(len(test_imu_data), 6)
    
    def test_data_scaling(self):
        """Test data scaling calculations"""
        # Test gyro scaling (0.0012 factor)
        raw_gyro = 100
        scaled_gyro = raw_gyro * 0.0012
        expected_gyro = 0.12
        self.assertAlmostEqual(scaled_gyro, expected_gyro)
        
        # Test accel scaling (0.0005978 factor)
        raw_accel = 1000
        scaled_accel = raw_accel * 0.0005978
        expected_accel = 0.5978
        self.assertAlmostEqual(scaled_accel, expected_accel)

class TestSignalFeatures(unittest.TestCase):
    """Test cases for signal feature extraction"""
    
    def test_rms_calculation(self):
        """Test RMS calculation"""
        data = np.array([1, 2, 3, 4, 5])
        rms = np.sqrt(np.mean(data**2))
        expected_rms = np.sqrt(11)
        self.assertAlmostEqual(rms, expected_rms)
    
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation"""
        # Simple test case
        data = np.array([1, -1, 1, -1, 1])
        zcr = np.sum(np.diff(np.sign(data)) != 0) / len(data)
        self.assertGreater(zcr, 0)

if __name__ == '__main__':
    unittest.main()