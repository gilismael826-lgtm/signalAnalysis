#!/usr/bin/env python3
"""
数据采集与标注工具
负责手势数据采集、标注、数据集管理和数据增强
"""

import numpy as np
import os
import sys
import json
import csv
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger


class GestureDataCollector:
    """手势数据采集器
    
    负责采集和保存手势数据
    """
    
    def __init__(self, output_dir='data/raw'):
        """初始化数据采集器
        
        Args:
            output_dir: 数据输出目录
        """
        self.output_dir = output_dir
        self.current_session = None
        self.emg_buffer = []
        self.imu_buffer = []
        self.is_collecting = False
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"数据采集器初始化: 输出目录={output_dir}")
    
    def start_session(self, gesture_name: str, subject_id: str = 'default'):
        """开始新的采集会话
        
        Args:
            gesture_name: 手势名称
            subject_id: 受试者ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"{gesture_name}_{subject_id}_{timestamp}"
        
        self.current_session = {
            'session_id': session_id,
            'gesture_name': gesture_name,
            'subject_id': subject_id,
            'start_time': datetime.now().isoformat(),
            'emg_sample_rate': config.EMG_SAMPLE_RATE,
            'imu_sample_rate': config.IMU_SAMPLE_RATE
        }
        
        self.emg_buffer = []
        self.imu_buffer = []
        self.is_collecting = True
        
        logger.info(f"开始采集会话: {session_id}")
    
    def add_data(self, emg_data: List[float], imu_data: List[float]):
        """添加采集数据
        
        Args:
            emg_data: EMG数据 (8通道)
            imu_data: IMU数据 (6通道)
        """
        if not self.is_collecting:
            return
        
        self.emg_buffer.append(emg_data)
        self.imu_buffer.append(imu_data)
    
    def stop_session(self) -> Dict:
        """停止采集并返回会话数据
        
        Returns:
            会话数据字典
        """
        if not self.is_collecting:
            return None
        
        self.is_collecting = False
        
        session_data = {
            **self.current_session,
            'end_time': datetime.now().isoformat(),
            'emg_samples': len(self.emg_buffer),
            'imu_samples': len(self.imu_buffer),
            'emg_data': np.array(self.emg_buffer),
            'imu_data': np.array(self.imu_buffer)
        }
        
        logger.info(f"采集会话结束: {self.current_session['session_id']}, "
                   f"EMG样本数={len(self.emg_buffer)}, IMU样本数={len(self.imu_buffer)}")
        
        return session_data
    
    def save_session(self, session_data: Dict = None, format: str = 'npy') -> str:
        """保存会话数据
        
        Args:
            session_data: 会话数据，如果为None则使用当前会话
            format: 保存格式 ('npy', 'csv', 'json')
            
        Returns:
            保存的文件路径
        """
        if session_data is None:
            session_data = self.stop_session()
        
        if session_data is None:
            return None
        
        session_id = session_data['session_id']
        
        if format == 'npy':
            return self._save_npy(session_data)
        elif format == 'csv':
            return self._save_csv(session_data)
        elif format == 'json':
            return self._save_json(session_data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _save_npy(self, session_data: Dict) -> str:
        """保存为NPY格式"""
        session_id = session_data['session_id']
        base_path = os.path.join(self.output_dir, session_id)
        
        # 保存数据
        np.save(f"{base_path}_emg.npy", session_data['emg_data'])
        np.save(f"{base_path}_imu.npy", session_data['imu_data'])
        
        # 保存元数据
        metadata = {k: v for k, v in session_data.items() 
                   if k not in ['emg_data', 'imu_data']}
        with open(f"{base_path}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存: {base_path}")
        return base_path
    
    def _save_csv(self, session_data: Dict) -> str:
        """保存为CSV格式"""
        session_id = session_data['session_id']
        base_path = os.path.join(self.output_dir, session_id)
        
        # 保存EMG数据
        with open(f"{base_path}_emg.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f'emg_{i}' for i in range(8)])
            writer.writerows(session_data['emg_data'])
        
        # 保存IMU数据
        with open(f"{base_path}_imu.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([f'accel_{i}' for i in range(3)] + [f'gyro_{i}' for i in range(3)])
            writer.writerows(session_data['imu_data'])
        
        logger.info(f"CSV数据已保存: {base_path}")
        return base_path
    
    def _save_json(self, session_data: Dict) -> str:
        """保存为JSON格式"""
        session_id = session_data['session_id']
        filepath = os.path.join(self.output_dir, f"{session_id}.json")
        
        # 转换numpy数组为列表
        save_data = {
            **session_data,
            'emg_data': session_data['emg_data'].tolist(),
            'imu_data': session_data['imu_data'].tolist()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON数据已保存: {filepath}")
        return filepath
    
    def get_collection_status(self) -> Dict:
        """获取采集状态
        
        Returns:
            状态字典
        """
        return {
            'is_collecting': self.is_collecting,
            'current_session': self.current_session,
            'emg_buffer_size': len(self.emg_buffer),
            'imu_buffer_size': len(self.imu_buffer),
            'emg_duration': len(self.emg_buffer) / config.EMG_SAMPLE_RATE if self.emg_buffer else 0,
            'imu_duration': len(self.imu_buffer) / config.IMU_SAMPLE_RATE if self.imu_buffer else 0
        }


class GestureLabeler:
    """手势标签标注工具
    
    负责数据标注和管理
    """
    
    def __init__(self, data_dir='data/raw'):
        """初始化标注工具
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir
        self.labels = {}
        self.current_file = None
        self.annotations = []
        
        os.makedirs(data_dir, exist_ok=True)
        
        self._load_labels()
        
        logger.info(f"标注工具初始化: 数据目录={data_dir}")
    
    def _load_labels(self):
        """加载已有标签"""
        labels_file = os.path.join(self.data_dir, 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)
            logger.info(f"已加载 {len(self.labels)} 个标签")
    
    def _save_labels(self):
        """保存标签"""
        labels_file = os.path.join(self.data_dir, 'labels.json')
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=2)
    
    def add_label(self, file_id: str, gesture_name: str, 
                  start_idx: int = None, end_idx: int = None,
                  notes: str = ''):
        """添加标签
        
        Args:
            file_id: 文件ID
            gesture_name: 手势名称
            start_idx: 起始索引（可选）
            end_idx: 结束索引（可选）
            notes: 备注
        """
        if file_id not in self.labels:
            self.labels[file_id] = []
        
        annotation = {
            'gesture_name': gesture_name,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        self.labels[file_id].append(annotation)
        self._save_labels()
        
        logger.info(f"添加标签: {file_id} -> {gesture_name}")
    
    def remove_label(self, file_id: str, annotation_idx: int):
        """删除标签
        
        Args:
            file_id: 文件ID
            annotation_idx: 标注索引
        """
        if file_id in self.labels and annotation_idx < len(self.labels[file_id]):
            removed = self.labels[file_id].pop(annotation_idx)
            self._save_labels()
            logger.info(f"删除标签: {file_id}[{annotation_idx}]")
            return removed
        return None
    
    def update_label(self, file_id: str, annotation_idx: int, 
                     gesture_name: str = None, start_idx: int = None,
                     end_idx: int = None, notes: str = None):
        """更新标签
        
        Args:
            file_id: 文件ID
            annotation_idx: 标注索引
            gesture_name: 新手势名称
            start_idx: 新起始索引
            end_idx: 新结束索引
            notes: 新备注
        """
        if file_id in self.labels and annotation_idx < len(self.labels[file_id]):
            annotation = self.labels[file_id][annotation_idx]
            
            if gesture_name is not None:
                annotation['gesture_name'] = gesture_name
            if start_idx is not None:
                annotation['start_idx'] = start_idx
            if end_idx is not None:
                annotation['end_idx'] = end_idx
            if notes is not None:
                annotation['notes'] = notes
            
            annotation['updated_at'] = datetime.now().isoformat()
            self._save_labels()
            logger.info(f"更新标签: {file_id}[{annotation_idx}]")
            return annotation
        return None
    
    def get_labels(self, file_id: str = None) -> Dict:
        """获取标签
        
        Args:
            file_id: 文件ID，如果为None则返回所有标签
            
        Returns:
            标签字典
        """
        if file_id:
            return self.labels.get(file_id, [])
        return self.labels
    
    def batch_label(self, file_ids: List[str], gesture_name: str):
        """批量标注
        
        Args:
            file_ids: 文件ID列表
            gesture_name: 手势名称
        """
        for file_id in file_ids:
            self.add_label(file_id, gesture_name)
        
        logger.info(f"批量标注完成: {len(file_ids)} 个文件 -> {gesture_name}")
    
    def get_label_statistics(self) -> Dict:
        """获取标签统计
        
        Returns:
            统计字典
        """
        gesture_counts = {}
        for file_id, annotations in self.labels.items():
            for annotation in annotations:
                gesture = annotation['gesture_name']
                gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        return {
            'total_files': len(self.labels),
            'total_annotations': sum(len(v) for v in self.labels.values()),
            'gesture_counts': gesture_counts
        }


class DatasetManager:
    """数据集管理器
    
    负责数据集的创建、分割和管理
    """
    
    def __init__(self, data_dir='data', output_dir='datasets'):
        """初始化数据集管理器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 数据集输出目录
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"数据集管理器初始化: 数据目录={data_dir}, 输出目录={output_dir}")
    
    def create_dataset(self, name: str, gesture_names: List[str] = None,
                       train_ratio: float = 0.7, val_ratio: float = 0.15,
                       test_ratio: float = 0.15, random_seed: int = 42):
        """创建数据集
        
        Args:
            name: 数据集名称
            gesture_names: 包含的手势名称列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            
        Returns:
            数据集信息字典
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 加载标签
        labels_file = os.path.join(self.data_dir, 'raw', 'labels.json')
        if not os.path.exists(labels_file):
            logger.error("标签文件不存在")
            return None
        
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        # 按手势分组
        gesture_files = {}
        for file_id, annotations in labels.items():
            for annotation in annotations:
                gesture = annotation['gesture_name']
                if gesture_names is None or gesture in gesture_names:
                    if gesture not in gesture_files:
                        gesture_files[gesture] = []
                    gesture_files[gesture].append(file_id)
        
        # 分割数据集
        np.random.seed(random_seed)
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for gesture, files in gesture_files.items():
            np.random.shuffle(files)
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            splits['train'][gesture] = files[:n_train]
            splits['val'][gesture] = files[n_train:n_train + n_val]
            splits['test'][gesture] = files[n_train + n_val:]
        
        # 保存数据集配置
        dataset_dir = os.path.join(self.output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        dataset_info = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'gesture_names': list(gesture_files.keys()),
            'splits': {
                split: {
                    gesture: len(files) for gesture, files in gestures.items()
                } for split, gestures in splits.items()
            }
        }
        
        with open(os.path.join(dataset_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        # 保存分割文件列表
        for split_name, gestures in splits.items():
            split_dir = os.path.join(dataset_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for gesture, files in gestures.items():
                gesture_dir = os.path.join(split_dir, gesture)
                os.makedirs(gesture_dir, exist_ok=True)
                
                # 复制或链接文件
                for file_id in files:
                    self._copy_data_files(file_id, gesture_dir)
        
        logger.info(f"数据集创建完成: {name}")
        logger.info(f"训练集: {sum(len(files) for files in splits['train'].values())} 样本")
        logger.info(f"验证集: {sum(len(files) for files in splits['val'].values())} 样本")
        logger.info(f"测试集: {sum(len(files) for files in splits['test'].values())} 样本")
        
        return dataset_info
    
    def _copy_data_files(self, file_id: str, target_dir: str):
        """复制数据文件到目标目录"""
        raw_dir = os.path.join(self.data_dir, 'raw')
        
        # 复制所有相关文件
        for ext in ['_emg.npy', '_imu.npy', '_meta.json', '_emg.csv', '_imu.csv']:
            src = os.path.join(raw_dir, f"{file_id}{ext}")
            if os.path.exists(src):
                dst = os.path.join(target_dir, f"{file_id}{ext}")
                shutil.copy2(src, dst)
    
    def load_dataset(self, name: str, split: str = 'train'):
        """加载数据集
        
        Args:
            name: 数据集名称
            split: 数据分割 ('train', 'val', 'test')
            
        Returns:
            数据列表
        """
        dataset_dir = os.path.join(self.output_dir, name, split)
        
        if not os.path.exists(dataset_dir):
            logger.error(f"数据集不存在: {name}/{split}")
            return None
        
        data = []
        
        for gesture_name in os.listdir(dataset_dir):
            gesture_dir = os.path.join(dataset_dir, gesture_name)
            if not os.path.isdir(gesture_dir):
                continue
            
            for file_id in self._get_file_ids(gesture_dir):
                sample = self._load_sample(gesture_dir, file_id)
                if sample:
                    sample['gesture_name'] = gesture_name
                    data.append(sample)
        
        logger.info(f"加载数据集: {name}/{split}, {len(data)} 样本")
        return data
    
    def _get_file_ids(self, directory: str) -> List[str]:
        """获取目录中的文件ID列表"""
        file_ids = set()
        for f in os.listdir(directory):
            if f.endswith('_emg.npy'):
                file_ids.add(f.replace('_emg.npy', ''))
        return list(file_ids)
    
    def _load_sample(self, directory: str, file_id: str) -> Dict:
        """加载单个样本"""
        emg_path = os.path.join(directory, f"{file_id}_emg.npy")
        imu_path = os.path.join(directory, f"{file_id}_imu.npy")
        meta_path = os.path.join(directory, f"{file_id}_meta.json")
        
        if not os.path.exists(emg_path) or not os.path.exists(imu_path):
            return None
        
        sample = {
            'file_id': file_id,
            'emg_data': np.load(emg_path),
            'imu_data': np.load(imu_path)
        }
        
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                sample['metadata'] = json.load(f)
        
        return sample
    
    def get_dataset_statistics(self, name: str) -> Dict:
        """获取数据集统计信息
        
        Args:
            name: 数据集名称
            
        Returns:
            统计信息字典
        """
        info_path = os.path.join(self.output_dir, name, 'dataset_info.json')
        
        if not os.path.exists(info_path):
            return None
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        return info
    
    def list_datasets(self) -> List[str]:
        """列出所有数据集
        
        Returns:
            数据集名称列表
        """
        if not os.path.exists(self.output_dir):
            return []
        
        return [d for d in os.listdir(self.output_dir) 
                if os.path.isdir(os.path.join(self.output_dir, d))]
    
    def delete_dataset(self, name: str):
        """删除数据集
        
        Args:
            name: 数据集名称
        """
        dataset_dir = os.path.join(self.output_dir, name)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            logger.info(f"数据集已删除: {name}")


class DataAugmenter:
    """数据增强器
    
    提供多种数据增强方法
    """
    
    def __init__(self):
        """初始化数据增强器"""
        logger.info("数据增强器初始化")
    
    def add_gaussian_noise(self, data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """添加高斯噪声
        
        Args:
            data: 输入数据
            noise_level: 噪声水平（相对于标准差）
            
        Returns:
            增强后的数据
        """
        noise = np.random.normal(0, noise_level * np.std(data), data.shape)
        return data + noise
    
    def time_scale(self, data: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """时间缩放
        
        Args:
            data: 输入数据
            scale_factor: 缩放因子
            
        Returns:
            缩放后的数据
        """
        from scipy import signal as sp_signal
        
        if scale_factor == 1.0:
            return data
        
        original_length = len(data)
        new_length = int(original_length * scale_factor)
        
        scaled = sp_signal.resample(data, new_length)
        
        # 如果需要保持原始长度，进行裁剪或填充
        if len(scaled) > original_length:
            scaled = scaled[:original_length]
        elif len(scaled) < original_length:
            pad_width = original_length - len(scaled)
            scaled = np.pad(scaled, (0, pad_width), mode='edge')
        
        return scaled
    
    def amplitude_scale(self, data: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """幅度缩放
        
        Args:
            data: 输入数据
            scale_factor: 缩放因子
            
        Returns:
            缩放后的数据
        """
        return data * scale_factor
    
    def time_shift(self, data: np.ndarray, shift_samples: int = 0) -> np.ndarray:
        """时间平移
        
        Args:
            data: 输入数据
            shift_samples: 平移样本数（正数向右，负数向左）
            
        Returns:
            平移后的数据
        """
        if shift_samples == 0:
            return data
        
        shifted = np.zeros_like(data)
        
        if shift_samples > 0:
            shifted[shift_samples:] = data[:-shift_samples]
            shifted[:shift_samples] = data[0]
        else:
            shifted[:shift_samples] = data[-shift_samples:]
            shifted[shift_samples:] = data[-1]
        
        return shifted
    
    def augment_emg(self, emg_data: np.ndarray, 
                    noise_level: float = 0.05,
                    time_scale_range: Tuple[float, float] = (0.9, 1.1),
                    amp_scale_range: Tuple[float, float] = (0.9, 1.1),
                    time_shift_range: Tuple[int, int] = (-10, 10)) -> np.ndarray:
        """增强EMG数据
        
        Args:
            emg_data: EMG数据 (samples, channels)
            noise_level: 噪声水平
            time_scale_range: 时间缩放范围
            amp_scale_range: 幅度缩放范围
            time_shift_range: 时间平移范围
            
        Returns:
            增强后的数据
        """
        augmented = emg_data.copy()
        
        # 随机应用增强
        if np.random.random() < 0.5:
            augmented = self.add_gaussian_noise(augmented, noise_level)
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(*time_scale_range)
            # 对每个通道分别应用时间缩放
            for i in range(augmented.shape[1]):
                augmented[:, i] = self.time_scale(augmented[:, i], scale)
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(*amp_scale_range)
            augmented = self.amplitude_scale(augmented, scale)
        
        if np.random.random() < 0.3:
            shift = np.random.randint(*time_shift_range)
            for i in range(augmented.shape[1]):
                augmented[:, i] = self.time_shift(augmented[:, i], shift)
        
        return augmented
    
    def augment_imu(self, imu_data: np.ndarray,
                    noise_level: float = 0.02,
                    time_scale_range: Tuple[float, float] = (0.95, 1.05),
                    amp_scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """增强IMU数据
        
        Args:
            imu_data: IMU数据 (samples, channels)
            noise_level: 噪声水平
            time_scale_range: 时间缩放范围
            amp_scale_range: 幅度缩放范围
            
        Returns:
            增强后的数据
        """
        augmented = imu_data.copy()
        
        if np.random.random() < 0.5:
            augmented = self.add_gaussian_noise(augmented, noise_level)
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(*time_scale_range)
            for i in range(augmented.shape[1]):
                augmented[:, i] = self.time_scale(augmented[:, i], scale)
        
        if np.random.random() < 0.3:
            scale = np.random.uniform(*amp_scale_range)
            augmented = self.amplitude_scale(augmented, scale)
        
        return augmented
    
    def augment_sample(self, sample: Dict, num_augmentations: int = 1) -> List[Dict]:
        """增强样本
        
        Args:
            sample: 原始样本字典
            num_augmentations: 增强数量
            
        Returns:
            增强后的样本列表
        """
        augmented_samples = []
        
        for i in range(num_augmentations):
            aug_sample = {
                'file_id': f"{sample.get('file_id', 'sample')}_aug_{i}",
                'emg_data': self.augment_emg(sample['emg_data']),
                'imu_data': self.augment_imu(sample['imu_data']),
                'gesture_name': sample.get('gesture_name'),
                'is_augmented': True,
                'augmentation_seed': np.random.get_state()[1][0]
            }
            augmented_samples.append(aug_sample)
        
        return augmented_samples
    
    def augment_dataset(self, samples: List[Dict], 
                        target_samples_per_gesture: int = None,
                        augment_factor: int = 2) -> List[Dict]:
        """增强整个数据集
        
        Args:
            samples: 样本列表
            target_samples_per_gesture: 每个手势的目标样本数
            augment_factor: 增强倍数
            
        Returns:
            增强后的样本列表
        """
        # 按手势分组
        gesture_samples = {}
        for sample in samples:
            gesture = sample.get('gesture_name', 'unknown')
            if gesture not in gesture_samples:
                gesture_samples[gesture] = []
            gesture_samples[gesture].append(sample)
        
        augmented_data = list(samples)  # 保留原始数据
        
        if target_samples_per_gesture:
            # 按目标数量增强
            for gesture, gesture_data in gesture_samples.items():
                current_count = len(gesture_data)
                needed = target_samples_per_gesture - current_count
                
                if needed > 0:
                    samples_to_augment = np.random.choice(
                        gesture_data, 
                        size=min(needed, current_count),
                        replace=True
                    )
                    
                    for sample in samples_to_augment:
                        aug_samples = self.augment_sample(sample, num_augmentations=1)
                        augmented_data.extend(aug_samples)
        else:
            # 按固定倍数增强
            for gesture, gesture_data in gesture_samples.items():
                for sample in gesture_data:
                    aug_samples = self.augment_sample(sample, num_augmentations=augment_factor - 1)
                    augmented_data.extend(aug_samples)
        
        logger.info(f"数据集增强完成: {len(samples)} -> {len(augmented_data)} 样本")
        return augmented_data


class DataLabelerTool:
    """数据标注综合工具
    
    整合采集、标注、管理和增强功能
    """
    
    def __init__(self, data_dir='data'):
        """初始化数据标注工具
        
        Args:
            data_dir: 数据根目录
        """
        self.data_dir = data_dir
        
        self.collector = GestureDataCollector(
            output_dir=os.path.join(data_dir, 'raw')
        )
        self.labeler = GestureLabeler(
            data_dir=os.path.join(data_dir, 'raw')
        )
        self.dataset_manager = DatasetManager(
            data_dir=data_dir,
            output_dir=os.path.join(data_dir, 'datasets')
        )
        self.augmenter = DataAugmenter()
        
        logger.info(f"数据标注工具初始化完成: {data_dir}")
    
    def collect_gesture(self, gesture_name: str, duration_ms: int = 3000,
                        subject_id: str = 'default') -> Dict:
        """采集手势数据（模拟接口，实际采集需要外部数据源）
        
        Args:
            gesture_name: 手势名称
            duration_ms: 采集时长（毫秒）
            subject_id: 受试者ID
            
        Returns:
            采集结果
        """
        self.collector.start_session(gesture_name, subject_id)
        
        # 注意：实际数据需要通过add_data方法添加
        # 这里只返回状态信息
        return {
            'status': 'started',
            'gesture_name': gesture_name,
            'duration_ms': duration_ms,
            'message': '请通过add_data方法添加数据'
        }
    
    def quick_label(self, file_ids: List[str], gesture_name: str):
        """快速标注
        
        Args:
            file_ids: 文件ID列表
            gesture_name: 手势名称
        """
        self.labeler.batch_label(file_ids, gesture_name)
    
    def prepare_training_data(self, dataset_name: str, 
                              gestures: List[str] = None,
                              augment: bool = True,
                              augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据
        
        Args:
            dataset_name: 数据集名称
            gestures: 手势列表
            augment: 是否增强
            augment_factor: 增强倍数
            
        Returns:
            特征矩阵和标签向量
        """
        from src.feature_extractor import FeatureExtractor
        
        # 加载数据集
        train_data = self.dataset_manager.load_dataset(dataset_name, 'train')
        
        if train_data is None:
            return None, None
        
        # 过滤手势
        if gestures:
            train_data = [s for s in train_data if s.get('gesture_name') in gestures]
        
        # 数据增强
        if augment:
            train_data = self.augmenter.augment_dataset(
                train_data, augment_factor=augment_factor
            )
        
        # 提取特征
        extractor = FeatureExtractor()
        X = []
        y = []
        
        for sample in train_data:
            features = extractor.extract_all_features(
                sample['emg_data'].T,  # 转置为 (channels, samples)
                sample['imu_data'].T
            )
            X.append(list(features.values()))
            y.append(sample['gesture_name'])
        
        return np.array(X), np.array(y)
    
    def get_statistics(self) -> Dict:
        """获取整体统计信息
        
        Returns:
            统计信息字典
        """
        label_stats = self.labeler.get_label_statistics()
        datasets = self.dataset_manager.list_datasets()
        
        return {
            'labels': label_stats,
            'datasets': datasets,
            'collection_status': self.collector.get_collection_status()
        }


class DataQualityChecker:
    """数据质量检查器
    
    检查采集数据的质量，包括信号完整性、噪声水平、运动伪影等
    """
    
    def __init__(self, emg_sample_rate=250.0, imu_sample_rate=104.0):
        """初始化数据质量检查器
        
        Args:
            emg_sample_rate: EMG采样率
            imu_sample_rate: IMU采样率
        """
        self.emg_sample_rate = emg_sample_rate
        self.imu_sample_rate = imu_sample_rate
        
        logger.info("数据质量检查器初始化完成")
    
    def check_signal_completeness(self, emg_data: np.ndarray, imu_data: np.ndarray,
                                   expected_emg_samples: int = None,
                                   expected_imu_samples: int = None) -> Dict:
        """检查信号完整性
        
        Args:
            emg_data: EMG数据 (samples, channels)
            imu_data: IMU数据 (samples, channels)
            expected_emg_samples: 期望的EMG样本数
            expected_imu_samples: 期望的IMU样本数
            
        Returns:
            完整性检查结果
        """
        results = {
            'emg_complete': True,
            'imu_complete': True,
            'emg_missing_ratio': 0.0,
            'imu_missing_ratio': 0.0,
            'issues': []
        }
        
        if emg_data is not None and len(emg_data) > 0:
            if expected_emg_samples:
                missing_ratio = 1 - len(emg_data) / expected_emg_samples
                results['emg_missing_ratio'] = missing_ratio
                if missing_ratio > 0.1:
                    results['emg_complete'] = False
                    results['issues'].append(f"EMG数据缺失 {missing_ratio*100:.1f}%")
            
            nan_count = np.sum(np.isnan(emg_data))
            if nan_count > 0:
                results['emg_complete'] = False
                results['issues'].append(f"EMG数据包含 {nan_count} 个NaN值")
        else:
            results['emg_complete'] = False
            results['issues'].append("EMG数据为空")
        
        if imu_data is not None and len(imu_data) > 0:
            if expected_imu_samples:
                missing_ratio = 1 - len(imu_data) / expected_imu_samples
                results['imu_missing_ratio'] = missing_ratio
                if missing_ratio > 0.1:
                    results['imu_complete'] = False
                    results['issues'].append(f"IMU数据缺失 {missing_ratio*100:.1f}%")
            
            nan_count = np.sum(np.isnan(imu_data))
            if nan_count > 0:
                results['imu_complete'] = False
                results['issues'].append(f"IMU数据包含 {nan_count} 个NaN值")
        else:
            results['imu_complete'] = False
            results['issues'].append("IMU数据为空")
        
        return results
    
    def check_noise_level(self, emg_data: np.ndarray, 
                          noise_threshold: float = 50.0) -> Dict:
        """检查噪声水平
        
        Args:
            emg_data: EMG数据 (samples, channels)
            noise_threshold: 噪声阈值
            
        Returns:
            噪声检查结果
        """
        results = {
            'noise_level_ok': True,
            'channel_noise_levels': [],
            'noisy_channels': [],
            'overall_snr': None
        }
        
        if emg_data is None or len(emg_data) == 0:
            return results
        
        for ch in range(emg_data.shape[1]):
            channel_data = emg_data[:, ch]
            
            noise_level = np.std(channel_data)
            signal_level = np.max(np.abs(channel_data)) - np.min(np.abs(channel_data))
            
            if signal_level > 0:
                snr = signal_level / (noise_level + 1e-10)
            else:
                snr = 0
            
            results['channel_noise_levels'].append({
                'channel': ch,
                'noise_level': float(noise_level),
                'signal_range': float(signal_level),
                'snr': float(snr)
            })
            
            if noise_level > noise_threshold:
                results['noisy_channels'].append(ch)
                results['noise_level_ok'] = False
        
        if results['channel_noise_levels']:
            results['overall_snr'] = float(np.mean([
                c['snr'] for c in results['channel_noise_levels']
            ]))
        
        return results
    
    def check_motion_artifact(self, emg_data: np.ndarray,
                               imu_data: np.ndarray,
                               threshold: float = 100.0) -> Dict:
        """检查运动伪影
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            threshold: 运动伪影阈值
            
        Returns:
            运动伪影检查结果
        """
        results = {
            'has_motion_artifact': False,
            'artifact_indices': [],
            'artifact_severity': 'none'
        }
        
        if emg_data is None or len(emg_data) == 0:
            return results
        
        for ch in range(emg_data.shape[1]):
            channel_data = emg_data[:, ch]
            
            diff = np.abs(np.diff(channel_data))
            artifact_indices = np.where(diff > threshold)[0]
            
            if len(artifact_indices) > 0:
                results['has_motion_artifact'] = True
                results['artifact_indices'].extend(artifact_indices.tolist())
        
        if results['has_motion_artifact']:
            artifact_ratio = len(set(results['artifact_indices'])) / len(emg_data)
            if artifact_ratio > 0.1:
                results['artifact_severity'] = 'high'
            elif artifact_ratio > 0.05:
                results['artifact_severity'] = 'medium'
            else:
                results['artifact_severity'] = 'low'
        
        return results
    
    def check_signal_saturation(self, emg_data: np.ndarray,
                                 max_value: float = 0x7FFFFF) -> Dict:
        """检查信号饱和
        
        Args:
            emg_data: EMG数据
            max_value: 最大值阈值
            
        Returns:
            饱和检查结果
        """
        results = {
            'has_saturation': False,
            'saturated_channels': [],
            'saturation_ratio': 0.0
        }
        
        if emg_data is None or len(emg_data) == 0:
            return results
        
        total_samples = emg_data.shape[0] * emg_data.shape[1]
        saturated_count = 0
        
        for ch in range(emg_data.shape[1]):
            channel_data = emg_data[:, ch]
            sat_count = np.sum(np.abs(channel_data) >= max_value * 0.99)
            
            if sat_count > 0:
                results['has_saturation'] = True
                results['saturated_channels'].append({
                    'channel': ch,
                    'saturation_count': int(sat_count)
                })
                saturated_count += sat_count
        
        results['saturation_ratio'] = saturated_count / total_samples if total_samples > 0 else 0
        
        return results
    
    def check_data_quality(self, emg_data: np.ndarray, imu_data: np.ndarray,
                           expected_duration_ms: int = None) -> Dict:
        """综合数据质量检查
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            expected_duration_ms: 期望时长（毫秒）
            
        Returns:
            综合质量检查结果
        """
        expected_emg = int(expected_duration_ms * self.emg_sample_rate / 1000) if expected_duration_ms else None
        expected_imu = int(expected_duration_ms * self.imu_sample_rate / 1000) if expected_duration_ms else None
        
        completeness = self.check_signal_completeness(emg_data, imu_data, expected_emg, expected_imu)
        noise = self.check_noise_level(emg_data)
        motion = self.check_motion_artifact(emg_data, imu_data)
        saturation = self.check_signal_saturation(emg_data)
        
        quality_score = 100.0
        issues = []
        
        if not completeness['emg_complete']:
            quality_score -= 20
            issues.extend(completeness['issues'])
        
        if not completeness['imu_complete']:
            quality_score -= 10
            issues.extend(completeness['issues'])
        
        if not noise['noise_level_ok']:
            quality_score -= 15
            issues.append(f"噪声水平过高，受影响通道: {noise['noisy_channels']}")
        
        if motion['has_motion_artifact']:
            if motion['artifact_severity'] == 'high':
                quality_score -= 25
            elif motion['artifact_severity'] == 'medium':
                quality_score -= 15
            else:
                quality_score -= 5
            issues.append(f"检测到运动伪影，严重程度: {motion['artifact_severity']}")
        
        if saturation['has_saturation']:
            quality_score -= 20
            issues.append(f"信号饱和，受影响通道: {[c['channel'] for c in saturation['saturated_channels']]}")
        
        quality_score = max(0, quality_score)
        
        quality_level = 'excellent'
        if quality_score < 60:
            quality_level = 'poor'
        elif quality_score < 75:
            quality_level = 'fair'
        elif quality_score < 90:
            quality_level = 'good'
        
        return {
            'quality_score': quality_score,
            'quality_level': quality_level,
            'issues': issues,
            'details': {
                'completeness': completeness,
                'noise': noise,
                'motion_artifact': motion,
                'saturation': saturation
            }
        }


class DataVisualizer:
    """数据可视化工具
    
    提供数据波形的可视化功能
    """
    
    def __init__(self, figsize=(12, 8)):
        """初始化可视化工具
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        logger.info("数据可视化工具初始化完成")
    
    def plot_emg_waveform(self, emg_data: np.ndarray, 
                          title: str = "EMG Signal",
                          channel_names: List[str] = None,
                          time_axis: np.ndarray = None) -> Dict:
        """绘制EMG波形
        
        Args:
            emg_data: EMG数据 (samples, channels) 或 (channels, samples)
            title: 图标题
            channel_names: 通道名称列表
            time_axis: 时间轴
            
        Returns:
            绘图数据字典
        """
        if emg_data is None or len(emg_data) == 0:
            return {'error': 'EMG数据为空'}
        
        if emg_data.shape[0] == 8 and emg_data.shape[1] != 8:
            emg_data = emg_data.T
        
        n_channels = emg_data.shape[1]
        
        if channel_names is None:
            channel_names = [f'Channel {i+1}' for i in range(n_channels)]
        
        if time_axis is None:
            time_axis = np.arange(emg_data.shape[0])
        
        plot_data = {
            'type': 'emg_waveform',
            'title': title,
            'n_channels': n_channels,
            'time_axis': time_axis.tolist(),
            'channels': []
        }
        
        for i in range(n_channels):
            channel_data = emg_data[:, i]
            plot_data['channels'].append({
                'name': channel_names[i] if i < len(channel_names) else f'Channel {i+1}',
                'data': channel_data.tolist(),
                'stats': {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'min': float(np.min(channel_data)),
                    'max': float(np.max(channel_data))
                }
            })
        
        return plot_data
    
    def plot_imu_waveform(self, imu_data: np.ndarray,
                         title: str = "IMU Signal",
                         time_axis: np.ndarray = None) -> Dict:
        """绘制IMU波形
        
        Args:
            imu_data: IMU数据 (samples, 6) 或 (6, samples)
            title: 图标题
            time_axis: 时间轴
            
        Returns:
            绘图数据字典
        """
        if imu_data is None or len(imu_data) == 0:
            return {'error': 'IMU数据为空'}
        
        if imu_data.shape[0] == 6 and imu_data.shape[1] != 6:
            imu_data = imu_data.T
        
        if time_axis is None:
            time_axis = np.arange(imu_data.shape[0])
        
        gyro_names = ['Gyro X', 'Gyro Y', 'Gyro Z']
        accel_names = ['Accel X', 'Accel Y', 'Accel Z']
        
        plot_data = {
            'type': 'imu_waveform',
            'title': title,
            'time_axis': time_axis.tolist(),
            'gyro': [],
            'accel': []
        }
        
        for i in range(3):
            gyro_data = imu_data[:, i]
            accel_data = imu_data[:, i + 3]
            
            plot_data['gyro'].append({
                'name': gyro_names[i],
                'data': gyro_data.tolist(),
                'stats': {
                    'mean': float(np.mean(gyro_data)),
                    'std': float(np.std(gyro_data)),
                    'range': float(np.max(gyro_data) - np.min(gyro_data))
                }
            })
            
            plot_data['accel'].append({
                'name': accel_names[i],
                'data': accel_data.tolist(),
                'stats': {
                    'mean': float(np.mean(accel_data)),
                    'std': float(np.std(accel_data)),
                    'range': float(np.max(accel_data) - np.min(accel_data))
                }
            })
        
        return plot_data
    
    def plot_spectrogram(self, signal: np.ndarray, sample_rate: float,
                        title: str = "Spectrogram") -> Dict:
        """绘制频谱图
        
        Args:
            signal: 信号数据
            sample_rate: 采样率
            title: 图标题
            
        Returns:
            频谱图数据字典
        """
        if signal is None or len(signal) == 0:
            return {'error': '信号数据为空'}
        
        try:
            from scipy import signal as sp_signal
            
            f, t, Sxx = sp_signal.spectrogram(signal, fs=sample_rate)
            
            return {
                'type': 'spectrogram',
                'title': title,
                'frequencies': f.tolist(),
                'times': t.tolist(),
                'spectrogram': Sxx.tolist()
            }
        except ImportError:
            return {'error': 'scipy未安装，无法计算频谱图'}
    
    def create_realtime_preview_data(self, emg_buffer: List, imu_buffer: List,
                                     window_size: int = 500) -> Dict:
        """创建实时预览数据
        
        Args:
            emg_buffer: EMG缓冲区数据
            imu_buffer: IMU缓冲区数据
            window_size: 显示窗口大小
            
        Returns:
            实时预览数据
        """
        preview_data = {
            'type': 'realtime_preview',
            'timestamp': datetime.now().isoformat(),
            'emg': None,
            'imu': None
        }
        
        if emg_buffer and len(emg_buffer) > 0:
            emg_array = np.array(emg_buffer[-window_size:]) if len(emg_buffer) > window_size else np.array(emg_buffer)
            if len(emg_array.shape) == 1:
                emg_array = emg_array.reshape(-1, 1)
            preview_data['emg'] = {
                'data': emg_array.tolist(),
                'samples': len(emg_array),
                'channels': emg_array.shape[1] if len(emg_array.shape) > 1 else 1
            }
        
        if imu_buffer and len(imu_buffer) > 0:
            imu_array = np.array(imu_buffer[-window_size:]) if len(imu_buffer) > window_size else np.array(imu_buffer)
            if len(imu_array.shape) == 1:
                imu_array = imu_array.reshape(-1, 1)
            preview_data['imu'] = {
                'data': imu_array.tolist(),
                'samples': len(imu_array),
                'channels': imu_array.shape[1] if len(imu_array.shape) > 1 else 1
            }
        
        return preview_data
