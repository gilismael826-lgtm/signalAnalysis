# EMG/IMU Signal Analysis Package

"""
EMG/IMU Signal Analysis System

A Python package for receiving, processing, and analyzing EMG and IMU data
from wearable devices.

Modules:
    emg_imu_receiver: Basic version of the signal receiver
    emg_imu_receiver_enhanced: Enhanced version with additional features
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import emg_imu_receiver
from . import emg_imu_receiver_enhanced

__all__ = ['emg_imu_receiver', 'emg_imu_receiver_enhanced']