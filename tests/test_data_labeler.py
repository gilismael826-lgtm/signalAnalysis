#!/usr/bin/env python3
"""
数据采集与标注工具测试
验证数据采集、标注、管理和增强功能
"""

import numpy as np
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_labeler import (
    GestureDataCollector, GestureLabeler, 
    DatasetManager, DataAugmenter, DataLabelerTool
)


def test_data_collector():
    """测试数据采集器"""
    print("=" * 60)
    print("测试数据采集器")
    print("=" * 60)
    
    collector = GestureDataCollector(output_dir='tests/test_data/raw')
    
    # 开始采集会话
    collector.start_session(gesture_name='fist', subject_id='test_subject')
    
    # 添加模拟数据
    for _ in range(100):
        emg_data = np.random.randn(8).tolist()
        imu_data = np.random.randn(6).tolist()
        collector.add_data(emg_data, imu_data)
    
    # 获取采集状态
    status = collector.get_collection_status()
    print(f"采集状态: is_collecting={status['is_collecting']}")
    print(f"EMG缓冲区大小: {status['emg_buffer_size']}")
    print(f"IMU缓冲区大小: {status['imu_buffer_size']}")
    
    # 停止采集
    session_data = collector.stop_session()
    print(f"会话ID: {session_data['session_id']}")
    print(f"EMG数据形状: {session_data['emg_data'].shape}")
    print(f"IMU数据形状: {session_data['imu_data'].shape}")
    
    # 保存数据
    filepath = collector.save_session(session_data, format='npy')
    print(f"数据已保存: {filepath}")
    
    # 清理
    if os.path.exists('tests/test_data'):
        shutil.rmtree('tests/test_data')
    
    print()


def test_data_collector_csv():
    """测试CSV格式保存"""
    print("=" * 60)
    print("测试CSV格式保存")
    print("=" * 60)
    
    collector = GestureDataCollector(output_dir='tests/test_data/raw')
    
    collector.start_session(gesture_name='open', subject_id='test')
    
    for _ in range(50):
        collector.add_data(
            np.random.randn(8).tolist(),
            np.random.randn(6).tolist()
        )
    
    session_data = collector.stop_session()
    filepath = collector.save_session(session_data, format='csv')
    print(f"CSV数据已保存: {filepath}")
    
    # 清理
    if os.path.exists('tests/test_data'):
        shutil.rmtree('tests/test_data')
    
    print()


def test_gesture_labeler():
    """测试手势标签标注工具"""
    print("=" * 60)
    print("测试手势标签标注工具")
    print("=" * 60)
    
    labeler = GestureLabeler(data_dir='tests/test_data/raw')
    
    # 添加标签
    labeler.add_label('file_001', 'fist', start_idx=0, end_idx=100, notes='测试标签1')
    labeler.add_label('file_002', 'open', start_idx=0, end_idx=100, notes='测试标签2')
    labeler.add_label('file_003', 'pinch', notes='测试标签3')
    
    # 获取标签
    labels = labeler.get_labels()
    print(f"已标注文件数: {len(labels)}")
    
    # 获取单个文件标签
    file_labels = labeler.get_labels('file_001')
    print(f"file_001标签: {file_labels}")
    
    # 更新标签
    labeler.update_label('file_001', 0, gesture_name='wave', notes='已更新')
    updated = labeler.get_labels('file_001')
    print(f"更新后标签: {updated}")
    
    # 批量标注
    labeler.batch_label(['file_004', 'file_005', 'file_006'], 'wave')
    
    # 获取统计
    stats = labeler.get_label_statistics()
    print(f"标签统计: {stats}")
    
    # 清理
    if os.path.exists('tests/test_data'):
        shutil.rmtree('tests/test_data')
    
    print()


def test_dataset_manager():
    """测试数据集管理器"""
    print("=" * 60)
    print("测试数据集管理器")
    print("=" * 60)
    
    # 首先创建一些测试数据
    raw_dir = 'tests/test_data/raw'
    os.makedirs(raw_dir, exist_ok=True)
    
    # 创建模拟数据文件
    for i, gesture in enumerate(['fist', 'open', 'pinch', 'wave']):
        for j in range(5):
            file_id = f"{gesture}_{j:03d}"
            
            # 保存EMG数据
            emg_data = np.random.randn(150, 8) * 1000 + i * 500
            np.save(os.path.join(raw_dir, f"{file_id}_emg.npy"), emg_data)
            
            # 保存IMU数据
            imu_data = np.random.randn(62, 6) * 0.5 + i * 0.2
            np.save(os.path.join(raw_dir, f"{file_id}_imu.npy"), imu_data)
            
            # 保存元数据
            metadata = {
                'session_id': file_id,
                'gesture_name': gesture,
                'subject_id': 'test'
            }
            with open(os.path.join(raw_dir, f"{file_id}_meta.json"), 'w') as f:
                import json
                json.dump(metadata, f)
    
    # 创建标签文件
    labels = {}
    for gesture in ['fist', 'open', 'pinch', 'wave']:
        for j in range(5):
            file_id = f"{gesture}_{j:03d}"
            labels[file_id] = [{
                'gesture_name': gesture,
                'start_idx': 0,
                'end_idx': 150
            }]
    
    with open(os.path.join(raw_dir, 'labels.json'), 'w') as f:
        import json
        json.dump(labels, f)
    
    # 创建数据集管理器
    manager = DatasetManager(
        data_dir='tests/test_data',
        output_dir='tests/test_data/datasets'
    )
    
    # 创建数据集
    dataset_info = manager.create_dataset(
        name='test_dataset',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    print(f"数据集名称: {dataset_info['name']}")
    print(f"手势类型: {dataset_info['gesture_names']}")
    print(f"分割信息: {dataset_info['splits']}")
    
    # 列出数据集
    datasets = manager.list_datasets()
    print(f"可用数据集: {datasets}")
    
    # 获取统计信息
    stats = manager.get_dataset_statistics('test_dataset')
    print(f"数据集统计: {stats}")
    
    # 加载数据集
    train_data = manager.load_dataset('test_dataset', 'train')
    if train_data:
        print(f"训练集样本数: {len(train_data)}")
        if train_data:
            print(f"样本示例: gesture={train_data[0].get('gesture_name')}")
    
    # 清理
    if os.path.exists('tests/test_data'):
        shutil.rmtree('tests/test_data')
    
    print()


def test_data_augmenter():
    """测试数据增强器"""
    print("=" * 60)
    print("测试数据增强器")
    print("=" * 60)
    
    augmenter = DataAugmenter()
    
    # 创建测试数据
    emg_data = np.random.randn(150, 8) * 1000
    imu_data = np.random.randn(62, 6) * 0.5
    
    print(f"原始EMG数据形状: {emg_data.shape}")
    print(f"原始IMU数据形状: {imu_data.shape}")
    
    # 测试高斯噪声
    noisy_emg = augmenter.add_gaussian_noise(emg_data, noise_level=0.05)
    print(f"添加噪声后EMG标准差变化: {np.std(emg_data):.2f} -> {np.std(noisy_emg):.2f}")
    
    # 测试时间缩放
    scaled_emg = augmenter.time_scale(emg_data[:, 0], scale_factor=1.2)
    print(f"时间缩放后形状: {scaled_emg.shape}")
    
    # 测试幅度缩放
    amp_scaled = augmenter.amplitude_scale(emg_data, scale_factor=1.5)
    print(f"幅度缩放后均值变化: {np.mean(emg_data):.2f} -> {np.mean(amp_scaled):.2f}")
    
    # 测试时间平移
    shifted = augmenter.time_shift(emg_data[:, 0], shift_samples=10)
    print(f"时间平移后形状: {shifted.shape}")
    
    # 测试EMG增强
    aug_emg = augmenter.augment_emg(emg_data)
    print(f"增强后EMG形状: {aug_emg.shape}")
    
    # 测试IMU增强
    aug_imu = augmenter.augment_imu(imu_data)
    print(f"增强后IMU形状: {aug_imu.shape}")
    
    # 测试样本增强
    sample = {
        'file_id': 'test_sample',
        'emg_data': emg_data,
        'imu_data': imu_data,
        'gesture_name': 'fist'
    }
    
    aug_samples = augmenter.augment_sample(sample, num_augmentations=3)
    print(f"生成增强样本数: {len(aug_samples)}")
    
    # 测试数据集增强
    samples = [sample] * 5
    aug_dataset = augmenter.augment_dataset(samples, augment_factor=2)
    print(f"数据集增强: {len(samples)} -> {len(aug_dataset)} 样本")
    
    print()


def test_data_labeler_tool():
    """测试数据标注综合工具"""
    print("=" * 60)
    print("测试数据标注综合工具")
    print("=" * 60)
    
    tool = DataLabelerTool(data_dir='tests/test_data')
    
    # 测试采集接口
    result = tool.collect_gesture('fist', duration_ms=3000)
    print(f"采集状态: {result['status']}")
    
    # 测试快速标注
    tool.quick_label(['file_001', 'file_002'], 'fist')
    
    # 获取统计信息
    stats = tool.get_statistics()
    print(f"标签统计: {stats['labels']}")
    print(f"数据集列表: {stats['datasets']}")
    
    # 清理
    if os.path.exists('tests/test_data'):
        shutil.rmtree('tests/test_data')
    
    print()


def test_full_workflow():
    """测试完整工作流"""
    print("=" * 60)
    print("测试完整工作流")
    print("=" * 60)
    
    # 创建测试目录
    data_dir = 'tests/test_workflow'
    
    # 1. 数据采集
    print("\n1. 数据采集...")
    collector = GestureDataCollector(output_dir=os.path.join(data_dir, 'raw'))
    
    gestures = ['fist', 'open', 'pinch', 'wave']
    for gesture in gestures:
        for _ in range(3):  # 每个手势采集3次
            collector.start_session(gesture_name=gesture, subject_id='test')
            
            # 添加模拟数据
            for _ in range(150):
                collector.add_data(
                    np.random.randn(8).tolist(),
                    np.random.randn(6).tolist()
                )
            
            session_data = collector.stop_session()
            collector.save_session(session_data, format='npy')
    
    # 2. 数据标注
    print("\n2. 数据标注...")
    labeler = GestureLabeler(data_dir=os.path.join(data_dir, 'raw'))
    
    # 为所有文件添加标签
    raw_dir = os.path.join(data_dir, 'raw')
    for f in os.listdir(raw_dir):
        if f.endswith('_meta.json'):
            file_id = f.replace('_meta.json', '')
            with open(os.path.join(raw_dir, f), 'r') as fp:
                import json
                meta = json.load(fp)
            labeler.add_label(file_id, meta['gesture_name'])
    
    stats = labeler.get_label_statistics()
    print(f"标注统计: {stats}")
    
    # 3. 创建数据集
    print("\n3. 创建数据集...")
    manager = DatasetManager(
        data_dir=data_dir,
        output_dir=os.path.join(data_dir, 'datasets')
    )
    
    dataset_info = manager.create_dataset(
        name='gesture_dataset',
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    print(f"数据集创建完成: {dataset_info['name']}")
    
    # 4. 加载并增强数据
    print("\n4. 数据增强...")
    augmenter = DataAugmenter()
    
    train_data = manager.load_dataset('gesture_dataset', 'train')
    if train_data:
        augmented_data = augmenter.augment_dataset(train_data, augment_factor=2)
        print(f"训练集增强: {len(train_data)} -> {len(augmented_data)}")
    
    # 清理
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    
    print("\n完整工作流测试完成!")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("数据采集与标注工具测试")
    print("=" * 60 + "\n")
    
    test_data_collector()
    test_data_collector_csv()
    test_gesture_labeler()
    test_dataset_manager()
    test_data_augmenter()
    test_data_labeler_tool()
    test_full_workflow()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
