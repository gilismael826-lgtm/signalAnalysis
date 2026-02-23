#!/usr/bin/env python3
"""
实时手势识别模块测试
验证实时数据缓冲、特征提取和手势预测功能
"""

import numpy as np
import sys
import os
import time
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime_recognizer import (
    RealtimeGestureRecognizer, 
    RealtimeRecognizerManager,
    get_realtime_recognizer
)
from src.feature_extractor import FeatureExtractor
from src.gesture_classifier import GestureClassifier


def test_window_buffer():
    """测试时间窗口缓冲"""
    print("=" * 60)
    print("测试时间窗口缓冲")
    print("=" * 60)
    
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        emg_sample_rate=250.0,
        imu_sample_rate=104.0
    )
    
    print(f"EMG窗口样本数: {recognizer.emg_window_samples}")
    print(f"IMU窗口样本数: {recognizer.imu_window_samples}")
    
    # 添加EMG数据
    for _ in range(100):
        emg_data = np.random.randn(8).tolist()
        recognizer.update_emg(emg_data)
    
    # 添加IMU数据
    for _ in range(50):
        imu_data = np.random.randn(6).tolist()
        recognizer.update_imu(imu_data)
    
    status = recognizer.get_buffer_status()
    print(f"EMG缓冲区: {status['emg_samples']}/{status['emg_target']}")
    print(f"IMU缓冲区: {status['imu_samples']}/{status['imu_target']}")
    print(f"窗口就绪: {status['window_ready']}")
    
    # 填满缓冲区
    for _ in range(50):
        recognizer.update_emg(np.random.randn(8).tolist())
        recognizer.update_imu(np.random.randn(6).tolist())
    
    status = recognizer.get_buffer_status()
    print(f"填满后窗口就绪: {status['window_ready']}")
    print()


def test_window_alignment():
    """测试EMG/IMU窗口对齐"""
    print("=" * 60)
    print("测试EMG/IMU窗口对齐")
    print("=" * 60)
    
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        emg_sample_rate=250.0,
        imu_sample_rate=104.0
    )
    
    # 模拟实时数据流
    emg_samples_needed = recognizer.emg_window_samples
    imu_samples_needed = recognizer.imu_window_samples
    
    print(f"需要EMG样本: {emg_samples_needed}")
    print(f"需要IMU样本: {imu_samples_needed}")
    
    # 按比例添加数据
    emg_count = 0
    imu_count = 0
    
    while not recognizer.is_window_ready():
        # EMG:IMU比例约为 250:104 ≈ 2.4:1
        if emg_count < emg_samples_needed:
            recognizer.update_emg(np.random.randn(8).tolist())
            emg_count += 1
        
        if imu_count < imu_samples_needed and np.random.random() < 0.42:  # 104/250
            recognizer.update_imu(np.random.randn(6).tolist())
            imu_count += 1
    
    print(f"实际添加EMG样本: {emg_count}")
    print(f"实际添加IMU样本: {imu_count}")
    print(f"窗口就绪: {recognizer.is_window_ready()}")
    print()


def test_prediction_with_mock_model():
    """使用模拟模型测试预测"""
    print("=" * 60)
    print("使用模拟模型测试预测")
    print("=" * 60)
    
    # 创建并训练一个简单的模型
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    # 生成训练数据
    np.random.seed(42)
    X = []
    y = []
    
    gestures = ['fist', 'open', 'pinch', 'wave']
    for gesture_idx, gesture_name in enumerate(gestures):
        for _ in range(30):
            emg_data = np.random.randn(8, 150) * 1000 + gesture_idx * 500
            imu_data = np.random.randn(6, 62) * 0.5 + gesture_idx * 0.2
            features = extractor.extract_all_features(emg_data, imu_data)
            X.append(list(features.values()))
            y.append(gesture_name)
    
    X = np.array(X)
    y = np.array(y)
    
    # 训练模型
    classifier.train(X, y, test_size=0.2)
    print(f"模型训练完成")
    
    # 创建识别器并设置模型
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        slide_step=25
    )
    recognizer.set_model(classifier)
    
    # 填充数据并预测
    for _ in range(150):
        emg_data = np.random.randn(8).tolist()
        imu_data = np.random.randn(6).tolist()
        recognizer.update(emg_data, imu_data)
    
    # 执行预测
    result = recognizer.predict_now()
    if result:
        print(f"预测手势: {result['gesture']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"预测耗时: {result['prediction_time_ms']:.2f}ms")
    
    print()


def test_realtime_simulation():
    """模拟实时数据流测试"""
    print("=" * 60)
    print("模拟实时数据流测试")
    print("=" * 60)
    
    # 创建并训练模型
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = []
    y = []
    
    gestures = ['fist', 'open', 'wave']
    for gesture_idx, gesture_name in enumerate(gestures):
        for _ in range(20):
            emg_data = np.random.randn(8, 150) * 1000 + gesture_idx * 500
            imu_data = np.random.randn(6, 62) * 0.5 + gesture_idx * 0.2
            features = extractor.extract_all_features(emg_data, imu_data)
            X.append(list(features.values()))
            y.append(gesture_name)
    
    classifier.train(np.array(X), np.array(y), test_size=0.2)
    
    # 创建识别器
    recognizer = RealtimeGestureRecognizer(
        window_size_ms=600,
        slide_step=50,
        smoothing_window=3
    )
    recognizer.set_model(classifier)
    
    # 模拟实时数据流
    predictions = []
    
    # 模拟3秒数据（750个EMG样本，312个IMU样本）
    for i in range(750):
        emg_data = np.random.randn(8).tolist()
        imu_data = np.random.randn(6).tolist()
        
        result = recognizer.update(emg_data, imu_data)
        
        if result:
            predictions.append(result['gesture'])
    
    print(f"总预测次数: {len(predictions)}")
    if predictions:
        from collections import Counter
        pred_counts = Counter(predictions)
        print(f"预测分布: {dict(pred_counts)}")
    
    print()


def test_callback():
    """测试回调功能"""
    print("=" * 60)
    print("测试回调功能")
    print("=" * 60)
    
    # 创建模型
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = []
    y = ['fist'] * 20 + ['open'] * 20
    
    for i, gesture in enumerate(y):
        emg_data = np.random.randn(8, 150) * 1000 + (i // 20) * 500
        imu_data = np.random.randn(6, 62) * 0.5 + (i // 20) * 0.2
        features = extractor.extract_all_features(emg_data, imu_data)
        X.append(list(features.values()))
    
    classifier.train(np.array(X), np.array(y), test_size=0.2)
    
    # 创建识别器
    callback_results = []
    
    def on_prediction(result):
        callback_results.append(result)
    
    recognizer = RealtimeGestureRecognizer(slide_step=50)
    recognizer.set_model(classifier)
    recognizer.set_prediction_callback(on_prediction)
    
    # 填充数据
    for _ in range(200):
        recognizer.update(
            np.random.randn(8).tolist(),
            np.random.randn(6).tolist()
        )
    
    print(f"回调触发次数: {len(callback_results)}")
    if callback_results:
        print(f"最后一次预测: {callback_results[-1]['gesture']}")
    
    print()


def test_result_smoothing():
    """测试结果平滑"""
    print("=" * 60)
    print("测试结果平滑")
    print("=" * 60)
    
    recognizer = RealtimeGestureRecognizer(
        smoothing_window=5
    )
    
    # 手动添加预测历史
    test_predictions = ['fist', 'fist', 'open', 'fist', 'fist']
    test_confidences = [0.8, 0.7, 0.6, 0.9, 0.85]
    
    for pred, conf in zip(test_predictions, test_confidences):
        recognizer.prediction_history.append(pred)
        recognizer.confidence_history.append(conf)
    
    smoothed_pred, smoothed_conf = recognizer._smooth_results()
    
    print(f"原始预测: {test_predictions}")
    print(f"平滑后预测: {smoothed_pred}")
    print(f"原始置信度: {test_confidences}")
    print(f"平滑后置信度: {smoothed_conf:.4f}")
    print()


def test_switch_hysteresis_blocks_unstable_switch():
    """测试手势切换迟滞：低占比新手势不应立即切换"""
    recognizer = RealtimeGestureRecognizer(smoothing_window=5, switch_stability_threshold=0.8)

    recognizer.prediction_history.extend(['fist', 'fist', 'fist', 'open', 'open'])
    recognizer.confidence_history.extend([0.9, 0.88, 0.86, 0.82, 0.81])

    # 建立已有稳定状态
    recognizer._stable_prediction = 'fist'

    smoothed_prediction, _ = recognizer._smooth_results()
    stabilized_prediction, switch_blocked = recognizer._apply_switch_hysteresis(smoothed_prediction)

    assert smoothed_prediction == 'fist'
    assert stabilized_prediction == 'fist'
    assert switch_blocked is False

    # 构造接近切换边界但未达到阈值的情况
    recognizer.prediction_history.clear()
    recognizer.prediction_history.extend(['open', 'open', 'open', 'fist', 'fist'])
    smoothed_prediction, _ = recognizer._smooth_results()
    stabilized_prediction, switch_blocked = recognizer._apply_switch_hysteresis(smoothed_prediction)

    assert smoothed_prediction == 'open'
    assert stabilized_prediction == 'fist'
    assert switch_blocked is True


def test_switch_hysteresis_allows_stable_switch():
    """测试手势切换迟滞：高占比新手势应允许切换"""
    recognizer = RealtimeGestureRecognizer(smoothing_window=5, switch_stability_threshold=0.6)

    recognizer._stable_prediction = 'fist'
    recognizer.prediction_history.extend(['open', 'open', 'open', 'open', 'fist'])
    recognizer.confidence_history.extend([0.85, 0.86, 0.84, 0.83, 0.7])

    smoothed_prediction, _ = recognizer._smooth_results()
    stabilized_prediction, switch_blocked = recognizer._apply_switch_hysteresis(smoothed_prediction)

    assert smoothed_prediction == 'open'
    assert stabilized_prediction == 'open'
    assert switch_blocked is False
    assert recognizer._stable_prediction == 'open'


def test_status_management():
    """测试状态管理"""
    print("=" * 60)
    print("测试状态管理")
    print("=" * 60)
    
    recognizer = RealtimeGestureRecognizer()
    
    # 启动
    recognizer.start()
    print(f"启动后状态: is_running={recognizer.is_running}")
    
    # 添加数据
    for _ in range(100):
        recognizer.update(
            np.random.randn(8).tolist(),
            np.random.randn(6).tolist()
        )
    
    status = recognizer.get_status()
    print(f"缓冲区状态: EMG={status['buffer_status']['emg_samples']}, "
          f"IMU={status['buffer_status']['imu_samples']}")
    
    # 重置
    recognizer.reset()
    status = recognizer.get_status()
    print(f"重置后缓冲区: EMG={status['buffer_status']['emg_samples']}, "
          f"IMU={status['buffer_status']['imu_samples']}")
    
    # 停止
    recognizer.stop()
    print(f"停止后状态: is_running={recognizer.is_running}")
    
    print()


def test_feature_extraction():
    """测试实时特征提取"""
    print("=" * 60)
    print("测试实时特征提取")
    print("=" * 60)
    
    recognizer = RealtimeGestureRecognizer(window_size_ms=600)
    
    # 填充数据
    for _ in range(150):
        recognizer.update_emg(np.random.randn(8).tolist())
    
    for _ in range(62):
        recognizer.update_imu(np.random.randn(6).tolist())
    
    # 获取特征
    features = recognizer.get_current_features()
    
    if features:
        print(f"特征数量: {len(features)}")
        print(f"特征示例: {list(features.keys())[:5]}")
    
    # 获取原始数据
    emg_data, imu_data = recognizer.get_current_window_data()
    
    if emg_data is not None:
        print(f"EMG数据形状: {emg_data.shape}")
        print(f"IMU数据形状: {imu_data.shape}")
    
    print()


def test_manager():
    """测试识别器管理器"""
    print("=" * 60)
    print("测试识别器管理器")
    print("=" * 60)
    
    manager = RealtimeRecognizerManager(models_dir='tests/models')
    
    # 创建识别器
    recognizer = manager.create_recognizer(window_size_ms=600)
    print(f"识别器创建: {recognizer is not None}")
    
    # 列出模型
    models = manager.list_models()
    print(f"可用模型: {models}")
    
    # 获取识别器
    retrieved = manager.get_recognizer()
    print(f"获取识别器: {retrieved is not None}")
    
    print()


def test_global_instance():
    """测试全局实例"""
    print("=" * 60)
    print("测试全局实例")
    print("=" * 60)
    
    recognizer = get_realtime_recognizer(window_size_ms=500)
    print(f"全局实例窗口大小: {recognizer.window_size_ms}ms")
    
    recognizer2 = get_realtime_recognizer()
    print(f"再次获取同一实例: {recognizer is recognizer2}")
    
    print()


def test_performance():
    """测试性能"""
    print("=" * 60)
    print("测试性能")
    print("=" * 60)
    
    # 创建模型
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = []
    y = ['fist'] * 30 + ['open'] * 30
    
    for i, gesture in enumerate(y):
        emg_data = np.random.randn(8, 150) * 1000 + (i // 30) * 500
        imu_data = np.random.randn(6, 62) * 0.5 + (i // 30) * 0.2
        features = extractor.extract_all_features(emg_data, imu_data)
        X.append(list(features.values()))
    
    classifier.train(np.array(X), np.array(y), test_size=0.2)
    
    # 创建识别器
    recognizer = RealtimeGestureRecognizer(slide_step=25)
    recognizer.set_model(classifier)
    
    # 性能测试
    times = []
    
    for _ in range(300):  # 模拟约1.2秒数据
        start = time.time()
        recognizer.update(
            np.random.randn(8).tolist(),
            np.random.randn(6).tolist()
        )
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    max_time = np.max(times) * 1000
    
    print(f"平均处理时间: {avg_time:.2f}ms")
    print(f"最大处理时间: {max_time:.2f}ms")
    print(f"处理频率: {1000/avg_time:.1f} Hz")
    print(f"满足实时要求: {avg_time < 4.0}")  # 250Hz需要<4ms
    
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("实时手势识别模块测试")
    print("=" * 60 + "\n")
    
    test_window_buffer()
    test_window_alignment()
    test_prediction_with_mock_model()
    test_realtime_simulation()
    test_callback()
    test_result_smoothing()
    test_status_management()
    test_feature_extraction()
    test_manager()
    test_global_instance()
    test_performance()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
