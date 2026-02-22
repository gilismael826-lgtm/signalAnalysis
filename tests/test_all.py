#!/usr/bin/env python3
"""
EMG/IMU 手势识别系统 - 综合测试套件
测试所有模块的功能和性能
"""

import numpy as np
import os
import sys
import time
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
        self.start_time = None
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.total_tests += 1
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        try:
            test_func()
            self.passed_tests += 1
            self.results.append((test_name, 'PASS', None))
            print(f"✓ {test_name} 通过")
        except Exception as e:
            self.failed_tests += 1
            self.results.append((test_name, 'FAIL', str(e)))
            print(f"✗ {test_name} 失败: {e}")
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("测试摘要")
        print("=" * 60)
        print(f"总测试数: {self.total_tests}")
        print(f"通过: {self.passed_tests}")
        print(f"失败: {self.failed_tests}")
        
        if self.total_tests > 0:
            print(f"通过率: {self.passed_tests/self.total_tests*100:.1f}%")
        
        if self.start_time:
            print(f"总耗时: {time.time() - self.start_time:.2f}s")
        
        if self.failed_tests > 0:
            print("\n失败的测试:")
            for name, status, error in self.results:
                if status == 'FAIL':
                    print(f"  - {name}: {error}")


def test_config():
    """测试配置模块"""
    from src import config
    
    assert hasattr(config, 'EMG_SAMPLE_RATE'), "缺少EMG_SAMPLE_RATE配置"
    assert hasattr(config, 'IMU_SAMPLE_RATE'), "缺少IMU_SAMPLE_RATE配置"
    assert config.EMG_SAMPLE_RATE == 250.0, "EMG采样率不正确"
    assert config.IMU_SAMPLE_RATE == 104.0, "IMU采样率不正确"
    
    print(f"EMG采样率: {config.EMG_SAMPLE_RATE}")
    print(f"IMU采样率: {config.IMU_SAMPLE_RATE}")
    print("配置测试通过!")


def test_feature_extractor():
    """测试特征提取模块"""
    from src.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 测试单通道特征
    data = np.random.randn(1000)
    features = extractor.extract_channel_features(data, 250.0, 'test_')
    
    assert 'test_mean' in features, f"缺少均值特征，可用特征: {list(features.keys())[:10]}"
    assert 'test_rms' in features, f"缺少RMS特征，可用特征: {list(features.keys())[:10]}"
    assert 'test_median_frequency' in features, f"缺少中值频率特征，可用特征: {list(features.keys())[:10]}"
    
    # 测试EMG特征
    emg_data = np.random.randn(8, 150) * 1000
    emg_features = extractor.extract_emg_features(emg_data)
    assert len(emg_features) > 100, f"EMG特征数量过少: {len(emg_features)}"
    
    # 测试IMU特征
    imu_data = np.random.randn(6, 62) * 0.5
    imu_features = extractor.extract_imu_features(imu_data)
    assert len(imu_features) > 50, f"IMU特征数量过少: {len(imu_features)}"
    
    # 测试全部特征
    all_features = extractor.extract_all_features(emg_data, imu_data)
    print(f"特征数量: {len(all_features)}")
    print("特征提取测试通过!")


def test_feature_selector():
    """测试特征选择模块"""
    from src.feature_selector import FeatureSelector
    
    selector = FeatureSelector(n_components=10)
    
    # 测试数据
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.array(['a'] * 50 + ['b'] * 50)
    
    # 为不同类别添加差异
    X[:50, :5] += 2
    
    # 标准化
    X_scaled = selector.fit_scaler(X)
    
    # PCA
    X_pca = selector.apply_pca(X_scaled, n_components=10)
    assert X_pca.shape == (100, 10), f"PCA形状不正确: {X_pca.shape}"
    
    # 特征重要性
    importance = selector.get_feature_importance_rf(X_scaled, y)
    assert len(importance['importance']) == 50, "特征重要性数量不正确"
    
    print("特征选择测试通过!")


def test_gesture_classifier():
    """测试手势分类模块"""
    from src.gesture_classifier import GestureClassifier
    
    # 测试数据
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.array(['fist'] * 50 + ['open'] * 50)
    
    # 添加类别差异
    X[:50, :5] += 2
    
    # 训练
    classifier = GestureClassifier(algorithm='rf')
    result = classifier.train(X, y, test_size=0.2)
    
    assert result['test_accuracy'] > 0.5, f"准确率过低: {result['test_accuracy']}"
    
    # 预测
    X_new = np.random.randn(5, 50)
    X_new[0, :5] = 2  # 类似fist
    predictions = classifier.predict(X_new)
    
    assert len(predictions) == 5, "预测数量不正确"
    
    print(f"测试准确率: {result['test_accuracy']:.4f}")
    print("手势分类测试通过!")


def test_data_labeler():
    """测试数据标注模块"""
    from src.data_labeler import GestureDataCollector, GestureLabeler, DataAugmenter
    
    test_dir = 'tests/temp_test_data'
    
    try:
        # 测试采集器
        collector = GestureDataCollector(output_dir=os.path.join(test_dir, 'raw'))
        collector.start_session(gesture_name='test', subject_id='test')
        
        for _ in range(50):
            collector.add_data(
                np.random.randn(8).tolist(),
                np.random.randn(6).tolist()
            )
        
        session = collector.stop_session()
        assert session is not None, "采集会话失败"
        assert len(session['emg_data']) == 50, "EMG数据数量不正确"
        
        # 测试标注器
        labeler = GestureLabeler(data_dir=os.path.join(test_dir, 'raw'))
        labeler.add_label('test_file', 'test_gesture')
        
        labels = labeler.get_labels('test_file')
        assert len(labels) == 1, "标签添加失败"
        
        # 测试增强器
        augmenter = DataAugmenter()
        data = np.random.randn(100, 8)
        augmented = augmenter.add_gaussian_noise(data, noise_level=0.05)
        assert augmented.shape == data.shape, "数据增强形状不正确"
        
        print("数据标注测试通过!")
        
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


def test_realtime_recognizer():
    """测试实时识别模块"""
    from src.realtime_recognizer import RealtimeGestureRecognizer
    from src.gesture_classifier import GestureClassifier
    from src.feature_extractor import FeatureExtractor
    
    # 创建并训练简单模型
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = []
    y = ['fist'] * 20 + ['open'] * 20
    
    for i, gesture in enumerate(y):
        emg = np.random.randn(8, 150) * 1000 + (i // 20) * 500
        imu = np.random.randn(6, 62) * 0.5 + (i // 20) * 0.2
        features = extractor.extract_all_features(emg, imu)
        X.append(list(features.values()))
    
    classifier.train(np.array(X), np.array(y), test_size=0.2)
    
    # 创建识别器
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        slide_step=50
    )
    recognizer.set_model(classifier)
    
    # 填充数据
    for _ in range(150):
        recognizer.update_emg(np.random.randn(8).tolist())
        recognizer.update_imu(np.random.randn(6).tolist())
    
    # 检查窗口状态
    assert recognizer.is_window_ready(), "窗口未就绪"
    
    # 预测
    result = recognizer.predict_now()
    assert result is not None, "预测失败"
    assert 'gesture' in result, "结果缺少gesture字段"
    assert 'confidence' in result, "结果缺少confidence字段"
    
    print(f"预测手势: {result['gesture']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("实时识别测试通过!")


def test_performance_optimizer():
    """测试性能优化模块"""
    from src.performance_optimizer import (
        PerformanceMonitor, MemoryManager, 
        OptimizedFeatureExtractor
    )
    
    # 测试性能监控器
    monitor = PerformanceMonitor(history_size=10)
    monitor.record_processing_time(10.5)
    monitor.record_processing_time(12.3)
    
    stats = monitor.get_statistics()
    assert stats['processing']['count'] == 2, "处理时间记录不正确"
    
    # 测试内存管理器
    mm = MemoryManager(max_memory_mb=500)
    buffer = mm.create_buffer('test', (100, 50))
    assert buffer.shape == (100, 50), "缓冲区形状不正确"
    
    usage = mm.get_memory_usage()
    assert 'total_size_mb' in usage, "内存使用信息不完整"
    
    # 测试优化特征提取器
    optimizer = OptimizedFeatureExtractor()
    data = np.random.randn(8, 1000)
    features = optimizer.extract_time_domain_features_vectorized(data)
    assert len(features) > 0, "向量化特征提取失败"
    
    print("性能优化测试通过!")


def test_data_parser():
    """测试数据解析模块"""
    from src.data_parser import parse_packet
    
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
    
    assert pkt_type_result == 'EMG', f"数据包类型不正确: {pkt_type_result}"
    assert data is not None, "数据解析失败"
    assert len(data) == 8, f"EMG通道数不正确: {len(data)}"
    
    print("数据解析测试通过!")


def test_signal_processor():
    """测试信号处理模块"""
    from src.signal_processor import SignalProcessor
    
    processor = SignalProcessor(
        emg_sample_rate=250.0,
        imu_sample_rate=104.0,
        buffer_size=500
    )
    
    assert processor.emg_sample_rate == 250.0, "EMG采样率不正确"
    assert processor.imu_sample_rate == 104.0, "IMU采样率不正确"
    
    b, a = processor.design_bandpass_filter(lowcut=20.0, highcut=100.0, order=4)
    assert b is not None and a is not None, "带通滤波器设计失败"
    
    b, a = processor.design_notch_filter(notch_freq=50.0, quality_factor=30.0)
    assert b is not None and a is not None, "陷波滤波器设计失败"
    
    test_signal = np.random.randn(250)
    filtered = processor.apply_bandpass_filter(test_signal)
    assert len(filtered) == len(test_signal), "滤波后信号长度不正确"
    
    print("信号处理测试通过!")


def test_system_control():
    """测试系统控制模块"""
    from src.system_control_module import get_recognition_functions
    
    funcs = get_recognition_functions()
    assert 'open_model_training' in funcs, "缺少open_model_training函数"
    assert 'open_data_labeling' in funcs, "缺少open_data_labeling函数"
    assert 'open_recognition_settings' in funcs, "缺少open_recognition_settings函数"
    assert 'recognition_available' in funcs, "缺少recognition_available字段"
    
    print(f"识别功能可用: {funcs['recognition_available']}")
    print("系统控制测试通过!")


def test_integration():
    """集成测试 - 完整流程"""
    print("\n运行集成测试...")
    
    from src.feature_extractor import FeatureExtractor
    from src.feature_selector import FeatureSelector
    from src.gesture_classifier import GestureClassifier
    from src.realtime_recognizer import RealtimeGestureRecognizer
    
    # 1. 特征提取
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    
    # 2. 生成训练数据
    np.random.seed(42)
    X = []
    y = []
    
    gestures = ['fist', 'open', 'wave']
    for gesture_idx, gesture_name in enumerate(gestures):
        for _ in range(30):
            emg = np.random.randn(8, 150) * 1000 + gesture_idx * 500
            imu = np.random.randn(6, 62) * 0.5 + gesture_idx * 0.2
            features = extractor.extract_all_features(emg, imu)
            X.append(list(features.values()))
            y.append(gesture_name)
    
    X = np.array(X)
    y = np.array(y)
    
    # 3. 特征选择
    selector = FeatureSelector(n_components=50)
    X_scaled = selector.fit_scaler(X)
    
    # 4. 训练分类器
    classifier = GestureClassifier(algorithm='rf')
    result = classifier.train(X_scaled, y, test_size=0.2)
    
    print(f"训练准确率: {result['train_accuracy']:.4f}")
    print(f"测试准确率: {result['test_accuracy']:.4f}")
    
    # 5. 创建实时识别器
    recognizer = RealtimeGestureRecognizer(window_size_ms=600)
    recognizer.set_model(classifier)
    
    # 6. 模拟实时预测
    for _ in range(200):
        recognizer.update(
            np.random.randn(8).tolist(),
            np.random.randn(6).tolist()
        )
    
    if recognizer.is_window_ready():
        pred_result = recognizer.predict_now()
        print(f"实时预测: {pred_result['gesture']}")
    
    print("集成测试通过!")


def run_all_tests():
    """运行所有测试"""
    runner = TestRunner()
    runner.start_time = time.time()
    
    print("\n" + "=" * 60)
    print("EMG/IMU 手势识别系统 - 综合测试套件")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 运行测试
    runner.run_test("配置模块", test_config)
    runner.run_test("特征提取模块", test_feature_extractor)
    runner.run_test("特征选择模块", test_feature_selector)
    runner.run_test("手势分类模块", test_gesture_classifier)
    runner.run_test("数据标注模块", test_data_labeler)
    runner.run_test("实时识别模块", test_realtime_recognizer)
    runner.run_test("性能优化模块", test_performance_optimizer)
    runner.run_test("数据解析模块", test_data_parser)
    runner.run_test("信号处理模块", test_signal_processor)
    runner.run_test("系统控制模块", test_system_control)
    runner.run_test("集成测试", test_integration)
    
    # 打印摘要
    runner.print_summary()
    
    return runner.failed_tests == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
