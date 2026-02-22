#!/usr/bin/env python3
"""
高级功能模块
提供自定义手势训练、多用户支持、手势序列识别等功能
"""

import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import deque

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger
from src.feature_extractor import FeatureExtractor
from src.gesture_classifier import GestureClassifier
from src.gesture_library import PredefinedGestures, GestureTemplate


@dataclass
class UserProfile:
    """用户配置数据类"""
    user_id: str
    username: str
    created_at: str
    last_active: str
    trained_gestures: List[str]
    model_path: Optional[str]
    settings: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        return cls(**data)


class UserManager:
    """用户管理器
    
    管理多用户配置和个性化模型
    """
    
    def __init__(self, users_dir: str = 'data/users'):
        """初始化用户管理器
        
        Args:
            users_dir: 用户数据目录
        """
        self.users_dir = users_dir
        self.users: Dict[str, UserProfile] = {}
        self.current_user: Optional[UserProfile] = None
        
        os.makedirs(users_dir, exist_ok=True)
        
        self._load_users()
        
        logger.info(f"用户管理器初始化: {users_dir}")
    
    def _load_users(self):
        """加载已有用户"""
        for filename in os.listdir(self.users_dir):
            if filename.endswith('_profile.json'):
                filepath = os.path.join(self.users_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    user = UserProfile.from_dict(data)
                    self.users[user.user_id] = user
                except Exception as e:
                    logger.error(f"加载用户失败 {filename}: {e}")
    
    def create_user(self, username: str) -> UserProfile:
        """创建新用户
        
        Args:
            username: 用户名
            
        Returns:
            用户配置
        """
        user_id = f"user_{int(time.time() * 1000)}"
        
        user = UserProfile(
            user_id=user_id,
            username=username,
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            trained_gestures=[],
            model_path=None,
            settings={
                'window_size_ms': 600,
                'confidence_threshold': 0.5,
                'preferred_algorithm': 'rf'
            }
        )
        
        self.users[user_id] = user
        self._save_user(user)
        
        logger.info(f"创建用户: {username} ({user_id})")
        
        return user
    
    def _save_user(self, user: UserProfile):
        """保存用户配置
        
        Args:
            user: 用户配置
        """
        filepath = os.path.join(self.users_dir, f"{user.user_id}_profile.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user.to_dict(), f, ensure_ascii=False, indent=2)
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户配置
        """
        return self.users.get(user_id)
    
    def set_current_user(self, user_id: str) -> bool:
        """设置当前用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        user = self.users.get(user_id)
        if user:
            self.current_user = user
            user.last_active = datetime.now().isoformat()
            self._save_user(user)
            logger.info(f"切换用户: {user.username}")
            return True
        return False
    
    def update_user_settings(self, user_id: str, settings: Dict):
        """更新用户设置
        
        Args:
            user_id: 用户ID
            settings: 设置字典
        """
        user = self.users.get(user_id)
        if user:
            user.settings.update(settings)
            self._save_user(user)
            logger.info(f"更新用户设置: {user_id}")
    
    def add_trained_gesture(self, user_id: str, gesture_name: str):
        """添加已训练手势
        
        Args:
            user_id: 用户ID
            gesture_name: 手势名称
        """
        user = self.users.get(user_id)
        if user and gesture_name not in user.trained_gestures:
            user.trained_gestures.append(gesture_name)
            self._save_user(user)
    
    def set_user_model(self, user_id: str, model_path: str):
        """设置用户模型路径
        
        Args:
            user_id: 用户ID
            model_path: 模型路径
        """
        user = self.users.get(user_id)
        if user:
            user.model_path = model_path
            self._save_user(user)
    
    def list_users(self) -> List[Dict]:
        """列出所有用户
        
        Returns:
            用户信息列表
        """
        return [
            {
                'user_id': u.user_id,
                'username': u.username,
                'trained_gestures': len(u.trained_gestures),
                'last_active': u.last_active
            }
            for u in self.users.values()
        ]
    
    def delete_user(self, user_id: str):
        """删除用户
        
        Args:
            user_id: 用户ID
        """
        if user_id in self.users:
            # 删除用户文件
            profile_path = os.path.join(self.users_dir, f"{user_id}_profile.json")
            if os.path.exists(profile_path):
                os.remove(profile_path)
            
            # 删除用户模型
            user = self.users[user_id]
            if user.model_path and os.path.exists(user.model_path):
                os.remove(user.model_path)
            
            del self.users[user_id]
            
            if self.current_user and self.current_user.user_id == user_id:
                self.current_user = None
            
            logger.info(f"删除用户: {user_id}")


class CustomGestureTrainer:
    """自定义手势训练器
    
    引导用户采集数据并训练自定义手势
    """
    
    def __init__(self, user_manager: UserManager = None):
        """初始化训练器
        
        Args:
            user_manager: 用户管理器
        """
        self.user_manager = user_manager or UserManager()
        self.feature_extractor = FeatureExtractor()
        
        # 训练状态
        self.is_training = False
        self.current_gesture = None
        self.collected_samples = []
        
        # 回调
        self._progress_callback = None
        self._completion_callback = None
        
        logger.info("自定义手势训练器初始化")
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """设置进度回调
        
        Args:
            callback: 回调函数(current, total, message)
        """
        self._progress_callback = callback
    
    def set_completion_callback(self, callback: Callable[[Dict], None]):
        """设置完成回调
        
        Args:
            callback: 回调函数(result)
        """
        self._completion_callback = callback
    
    def start_collection(self, gesture_name: str, display_name: str = None):
        """开始采集
        
        Args:
            gesture_name: 手势名称
            display_name: 显示名称
        """
        self.current_gesture = {
            'name': gesture_name,
            'display_name': display_name or gesture_name,
            'start_time': datetime.now().isoformat()
        }
        self.collected_samples = []
        self.is_training = True
        
        logger.info(f"开始采集手势: {gesture_name}")
    
    def add_sample(self, emg_data: np.ndarray, imu_data: np.ndarray):
        """添加样本
        
        Args:
            emg_data: EMG数据 (8, samples)
            imu_data: IMU数据 (6, samples)
        """
        if not self.is_training:
            return
        
        # 提取特征
        features = self.feature_extractor.extract_all_features(emg_data, imu_data)
        self.collected_samples.append({
            'features': features,
            'timestamp': datetime.now().isoformat()
        })
        
        # 进度回调
        if self._progress_callback:
            self._progress_callback(
                len(self.collected_samples),
                100,  # 目标样本数
                f"采集样本 {len(self.collected_samples)}"
            )
    
    def stop_collection(self) -> Dict:
        """停止采集
        
        Returns:
            采集结果
        """
        self.is_training = False
        
        result = {
            'gesture': self.current_gesture,
            'sample_count': len(self.collected_samples),
            'samples': self.collected_samples
        }
        
        logger.info(f"停止采集: {self.current_gesture['name']}, 样本数={len(self.collected_samples)}")
        
        return result
    
    def train_model(self, gesture_data: Dict[str, List], 
                    algorithm: str = 'rf',
                    test_size: float = 0.2) -> Dict:
        """训练模型
        
        Args:
            gesture_data: 手势数据 {gesture_name: [features_list]}
            algorithm: 算法
            test_size: 测试集比例
            
        Returns:
            训练结果
        """
        # 准备数据
        X = []
        y = []
        
        for gesture_name, samples in gesture_data.items():
            for sample in samples:
                X.append(list(sample.values()) if isinstance(sample, dict) else sample)
                y.append(gesture_name)
        
        X = np.array(X)
        y = np.array(y)
        
        # 训练
        classifier = GestureClassifier(algorithm=algorithm)
        result = classifier.train(X, y, test_size=test_size)
        
        # 保存模型
        model_path = os.path.join('models', f"custom_{int(time.time())}.pkl")
        os.makedirs('models', exist_ok=True)
        classifier.save(model_path)
        
        # 更新用户
        if self.user_manager and self.user_manager.current_user:
            for gesture_name in gesture_data.keys():
                self.user_manager.add_trained_gesture(
                    self.user_manager.current_user.user_id,
                    gesture_name
                )
            self.user_manager.set_user_model(
                self.user_manager.current_user.user_id,
                model_path
            )
        
        result['model_path'] = model_path
        result['gestures'] = list(gesture_data.keys())
        
        # 完成回调
        if self._completion_callback:
            self._completion_callback(result)
        
        logger.info(f"模型训练完成: {model_path}")
        
        return result


class GestureSequenceRecognizer:
    """手势序列识别器
    
    识别手势组合和序列
    """
    
    def __init__(self, sequence_window: int = 10, sequence_timeout_ms: int = 2000):
        """初始化序列识别器
        
        Args:
            sequence_window: 序列窗口大小
            sequence_timeout_ms: 序列超时时间（毫秒）
        """
        self.sequence_window = sequence_window
        self.sequence_timeout_ms = sequence_timeout_ms
        
        # 预测历史
        self.prediction_history = deque(maxlen=sequence_window)
        self.timestamp_history = deque(maxlen=sequence_window)
        
        # 定义的序列
        self.sequences: Dict[str, List[str]] = {}
        
        # 回调
        self._sequence_callback = None
        
        logger.info(f"手势序列识别器初始化: 窗口={sequence_window}")
    
    def define_sequence(self, name: str, gestures: List[str], action: str = None):
        """定义手势序列
        
        Args:
            name: 序列名称
            gestures: 手势列表
            action: 触发动作
        """
        self.sequences[name] = {
            'gestures': gestures,
            'action': action,
            'triggered_count': 0
        }
        logger.info(f"定义手势序列: {name} = {gestures}")
    
    def add_prediction(self, gesture: str, confidence: float):
        """添加预测结果
        
        Args:
            gesture: 预测的手势
            confidence: 置信度
        """
        current_time = time.time()
        
        # 检查超时
        if self.timestamp_history:
            last_time = self.timestamp_history[-1]
            if (current_time - last_time) * 1000 > self.sequence_timeout_ms:
                # 超时，清空历史
                self.prediction_history.clear()
                self.timestamp_history.clear()
        
        self.prediction_history.append(gesture)
        self.timestamp_history.append(current_time)
        
        # 检测序列
        self._detect_sequences()
    
    def _detect_sequences(self):
        """检测手势序列"""
        if len(self.prediction_history) < 2:
            return
        
        current_sequence = list(self.prediction_history)
        
        for seq_name, seq_info in self.sequences.items():
            target_gestures = seq_info['gestures']
            
            # 检查是否匹配
            if len(current_sequence) >= len(target_gestures):
                recent = current_sequence[-len(target_gestures):]
                if recent == target_gestures:
                    # 匹配成功
                    seq_info['triggered_count'] += 1
                    
                    if self._sequence_callback:
                        self._sequence_callback({
                            'sequence_name': seq_name,
                            'gestures': target_gestures,
                            'action': seq_info['action'],
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    logger.info(f"检测到手势序列: {seq_name}")
    
    def set_sequence_callback(self, callback: Callable[[Dict], None]):
        """设置序列检测回调
        
        Args:
            callback: 回调函数
        """
        self._sequence_callback = callback
    
    def get_recent_sequence(self) -> List[str]:
        """获取最近的序列
        
        Returns:
            最近的手势列表
        """
        return list(self.prediction_history)
    
    def clear_history(self):
        """清空历史"""
        self.prediction_history.clear()
        self.timestamp_history.clear()
    
    def get_sequence_stats(self) -> Dict:
        """获取序列统计
        
        Returns:
            统计信息
        """
        return {
            name: {
                'gestures': info['gestures'],
                'triggered_count': info['triggered_count']
            }
            for name, info in self.sequences.items()
        }


class ContinuousGestureRecognizer:
    """连续手势识别器
    
    实现手势分割和连续识别
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.1,
                 min_gesture_duration_ms: int = 300,
                 silence_threshold_ms: int = 200):
        """初始化连续手势识别器
        
        Args:
            energy_threshold: 能量阈值
            min_gesture_duration_ms: 最小手势持续时间
            silence_threshold_ms: 静默阈值
        """
        self.energy_threshold = energy_threshold
        self.min_gesture_duration_ms = min_gesture_duration_ms
        self.silence_threshold_ms = silence_threshold_ms
        
        # 状态
        self.is_in_gesture = False
        self.gesture_start_time = None
        self.last_active_time = None
        
        # 缓冲
        self.gesture_buffer = []
        
        # 回调
        self._gesture_callback = None
        
        logger.info("连续手势识别器初始化")
    
    def process_frame(self, emg_data: np.ndarray, imu_data: np.ndarray) -> Optional[str]:
        """处理数据帧
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            
        Returns:
            检测到的手势或None
        """
        # 计算能量
        emg_energy = np.mean(np.abs(emg_data))
        current_time = time.time()
        
        # 状态机
        if emg_energy > self.energy_threshold:
            if not self.is_in_gesture:
                # 开始新手势
                self.is_in_gesture = True
                self.gesture_start_time = current_time
                self.gesture_buffer = []
            
            self.last_active_time = current_time
            self.gesture_buffer.append((emg_data.copy(), imu_data.copy()))
        
        else:
            if self.is_in_gesture and self.last_active_time:
                silence_duration = (current_time - self.last_active_time) * 1000
                
                if silence_duration > self.silence_threshold_ms:
                    # 手势结束
                    gesture_duration = (current_time - self.gesture_start_time) * 1000
                    
                    if gesture_duration >= self.min_gesture_duration_ms:
                        # 有效手势
                        result = self._process_gesture()
                        self.is_in_gesture = False
                        self.gesture_buffer = []
                        return result
                    else:
                        # 太短，忽略
                        self.is_in_gesture = False
                        self.gesture_buffer = []
        
        return None
    
    def _process_gesture(self) -> str:
        """处理检测到的手势
        
        Returns:
            手势标签
        """
        # 这里可以调用分类器
        # 简化版本：返回缓冲区大小
        gesture_info = {
            'duration_ms': (time.time() - self.gesture_start_time) * 1000,
            'frame_count': len(self.gesture_buffer),
            'timestamp': datetime.now().isoformat()
        }
        
        if self._gesture_callback:
            self._gesture_callback(gesture_info)
        
        logger.debug(f"检测到手势: {gesture_info}")
        
        return "detected"
    
    def set_gesture_callback(self, callback: Callable[[Dict], None]):
        """设置手势检测回调
        
        Args:
            callback: 回调函数
        """
        self._gesture_callback = callback
    
    def get_status(self) -> Dict:
        """获取状态
        
        Returns:
            状态字典
        """
        return {
            'is_in_gesture': self.is_in_gesture,
            'buffer_size': len(self.gesture_buffer),
            'gesture_duration_ms': (time.time() - self.gesture_start_time) * 1000 if self.gesture_start_time else 0
        }


class AdvancedFeaturesManager:
    """高级功能管理器
    
    整合所有高级功能
    """
    
    def __init__(self):
        """初始化高级功能管理器"""
        self.user_manager = UserManager()
        self.custom_trainer = CustomGestureTrainer(self.user_manager)
        self.sequence_recognizer = GestureSequenceRecognizer()
        self.continuous_recognizer = ContinuousGestureRecognizer()
        
        logger.info("高级功能管理器初始化完成")
    
    def get_feature_status(self) -> Dict:
        """获取功能状态
        
        Returns:
            状态字典
        """
        return {
            'users': len(self.user_manager.users),
            'current_user': self.user_manager.current_user.username if self.user_manager.current_user else None,
            'defined_sequences': len(self.sequence_recognizer.sequences),
            'continuous_status': self.continuous_recognizer.get_status()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("高级功能模块测试")
    print("=" * 60)
    
    # 测试用户管理
    print("\n1. 用户管理测试")
    user_manager = UserManager()
    user = user_manager.create_user("测试用户")
    print(f"创建用户: {user.username} ({user.user_id})")
    
    # 测试自定义手势训练
    print("\n2. 自定义手势训练测试")
    trainer = CustomGestureTrainer(user_manager)
    trainer.start_collection("custom_wave")
    print("开始采集...")
    
    # 模拟采集
    for _ in range(10):
        emg = np.random.randn(8, 150) * 1000
        imu = np.random.randn(6, 62) * 0.5
        trainer.add_sample(emg, imu)
    
    result = trainer.stop_collection()
    print(f"采集完成: {result['sample_count']} 个样本")
    
    # 测试序列识别
    print("\n3. 序列识别测试")
    seq_recognizer = GestureSequenceRecognizer()
    seq_recognizer.define_sequence("double_fist", ["fist", "fist"])
    
    def on_sequence(result):
        print(f"检测到序列: {result['sequence_name']}")
    
    seq_recognizer.set_sequence_callback(on_sequence)
    
    # 模拟预测
    seq_recognizer.add_prediction("fist", 0.9)
    seq_recognizer.add_prediction("open", 0.8)
    seq_recognizer.add_prediction("fist", 0.9)
    seq_recognizer.add_prediction("fist", 0.9)
    
    print("\n高级功能模块测试完成!")
