#!/usr/bin/env python3
"""
实时手势识别模块
负责实时数据缓冲、特征提取和手势预测
"""

import numpy as np
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger
from src.feature_extractor import FeatureExtractor
from src.gesture_classifier import GestureClassifier, GestureClassifierManager


class RealtimeGestureRecognizer:
    """实时手势识别器
    
    实现基于时间窗口的实时手势识别
    支持EMG (250Hz) 和 IMU (104Hz) 的不同采样率
    支持集成预测和智能模型选择
    """
    
    def __init__(self, 
                 window_size_ms: int = 600,
                 emg_sample_rate: float = None,
                 imu_sample_rate: float = None,
                 slide_step: int = 25,
                 confidence_threshold: float = 0.5,
                 smoothing_window: int = 5,
                 switch_stability_threshold: float = 0.65,
                 model_path: str = None,
                 models_dir: str = 'models',
                 ensemble_strategy: str = 'weighted_voting',
                 use_ensemble: bool = True):
        """初始化实时手势识别器
        
        Args:
            window_size_ms: 时间窗口大小（毫秒）
            emg_sample_rate: EMG采样率（Hz）
            imu_sample_rate: IMU采样率（Hz）
            slide_step: 滑动步长（EMG样本数）
            confidence_threshold: 置信度阈值
            smoothing_window: 结果平滑窗口大小
            switch_stability_threshold: 手势切换稳定性阈值（0-1，越高越不易抖动切换）
            model_path: 预训练模型路径（单模型模式）
            models_dir: 模型目录（集成模式）
            ensemble_strategy: 集成策略 ('weighted_voting', 'adaptive', 'best_only')
            use_ensemble: 是否使用集成预测
        """
        self.emg_sample_rate = emg_sample_rate or config.EMG_SAMPLE_RATE
        self.imu_sample_rate = imu_sample_rate or config.IMU_SAMPLE_RATE
        self.window_size_ms = window_size_ms
        self.slide_step = slide_step
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.switch_stability_threshold = np.clip(switch_stability_threshold, 0.0, 1.0)
        self.ensemble_strategy = ensemble_strategy
        self.use_ensemble = use_ensemble
        
        self.emg_window_samples = int(window_size_ms * self.emg_sample_rate / 1000)
        self.imu_window_samples = int(window_size_ms * self.imu_sample_rate / 1000)
        
        self.emg_buffer = [deque(maxlen=self.emg_window_samples) for _ in range(8)]
        self.imu_buffer = [deque(maxlen=self.imu_window_samples) for _ in range(6)]
        
        self.emg_counter = 0
        
        self.prediction_history = deque(maxlen=smoothing_window)
        self.confidence_history = deque(maxlen=smoothing_window)
        
        self.feature_cache = {}
        self.feature_cache_enabled = True
        self.feature_cache_max_size = 10
        
        self._incremental_stats = {}
        self._last_window_hash = None
        
        self.feature_extractor = FeatureExtractor(
            emg_sample_rate=self.emg_sample_rate,
            imu_sample_rate=self.imu_sample_rate
        )
        
        self.classifier = None
        self.classifier_manager = None
        self.models_dir = models_dir
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists(models_dir):
            self.load_ensemble_models(models_dir)
        
        self.is_running = False
        self.prediction_count = 0
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_prediction_time = None
        self._stable_prediction = None
        
        self._prediction_callback = None
        self._window_ready_callback = None
        
        logger.info(f"实时手势识别器初始化: "
                   f"窗口={window_size_ms}ms, "
                   f"EMG样本={self.emg_window_samples}, "
                   f"IMU样本={self.imu_window_samples}, "
                   f"集成模式={'启用' if use_ensemble else '禁用'}")
    
    # ==================== 模型管理 ====================
    
    def load_model(self, model_path: str):
        """加载预训练模型（单模型模式）
        
        Args:
            model_path: 模型文件路径
        """
        try:
            self.classifier = GestureClassifier(algorithm='svm')
            self.classifier.load(model_path)
            self.use_ensemble = False
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.classifier = None
    
    def load_ensemble_models(self, models_dir: str = None):
        """加载集成模型
        
        Args:
            models_dir: 模型目录
        """
        models_dir = models_dir or self.models_dir
        
        try:
            self.classifier_manager = GestureClassifierManager(models_dir=models_dir)
            loaded_count = self.classifier_manager.load_all_classifiers()
            
            if loaded_count > 0:
                self.use_ensemble = True
                self.classifier = self.classifier_manager.best_classifier
                logger.info(f"集成模型加载成功: {loaded_count} 个模型, "
                           f"最优模型: {self.classifier_manager.best_algorithm}")
            else:
                logger.warning("没有可用的集成模型")
        except Exception as e:
            logger.error(f"集成模型加载失败: {e}")
            self.classifier_manager = None
    
    def set_model(self, classifier: GestureClassifier):
        """设置分类器（单模型模式）
        
        Args:
            classifier: 已训练的分类器实例
        """
        self.classifier = classifier
        self.use_ensemble = False
        logger.info("分类器已设置")
    
    def set_classifier_manager(self, manager: GestureClassifierManager):
        """设置分类器管理器（集成模式）
        
        Args:
            manager: 已训练的分类器管理器实例
        """
        self.classifier_manager = manager
        self.classifier = manager.best_classifier
        self.use_ensemble = True
        logger.info(f"分类器管理器已设置, 最优模型: {manager.best_algorithm}")
    
    def set_ensemble_strategy(self, strategy: str):
        """设置集成策略
        
        Args:
            strategy: 集成策略 ('weighted_voting', 'adaptive', 'best_only')
        """
        self.ensemble_strategy = strategy
        logger.info(f"集成策略已设置为: {strategy}")
    
    def enable_ensemble(self, enabled: bool):
        """启用/禁用集成模式
        
        Args:
            enabled: 是否启用
        """
        self.use_ensemble = enabled and self.classifier_manager is not None
        logger.info(f"集成模式: {'启用' if self.use_ensemble else '禁用'}")
    
    # ==================== 数据更新 ====================
    
    def update_emg(self, emg_data: List[float]) -> Optional[Dict]:
        """更新EMG数据
        
        Args:
            emg_data: 8通道EMG数据
            
        Returns:
            预测结果（如果窗口就绪且达到滑动步长）
        """
        if len(emg_data) != 8:
            logger.warning(f"EMG数据通道数不正确: {len(emg_data)}")
            return None
        
        # 更新缓冲区
        for i in range(8):
            self.emg_buffer[i].append(emg_data[i])
        
        self.emg_counter += 1
        
        # 检查是否需要预测
        if self.is_window_ready() and self.emg_counter >= self.slide_step:
            self.emg_counter = 0
            return self._predict()
        
        return None
    
    def update_imu(self, imu_data: List[float]):
        """更新IMU数据
        
        Args:
            imu_data: 6通道IMU数据
        """
        if len(imu_data) != 6:
            logger.warning(f"IMU数据通道数不正确: {len(imu_data)}")
            return
        
        for i in range(6):
            self.imu_buffer[i].append(imu_data[i])
    
    def update(self, emg_data: List[float], imu_data: List[float]) -> Optional[Dict]:
        """同时更新EMG和IMU数据
        
        Args:
            emg_data: 8通道EMG数据
            imu_data: 6通道IMU数据
            
        Returns:
            预测结果
        """
        self.update_imu(imu_data)
        return self.update_emg(emg_data)
    
    # ==================== 窗口状态 ====================
    
    def is_window_ready(self) -> bool:
        """检查数据窗口是否就绪
        
        Returns:
            窗口是否就绪
        """
        emg_ready = len(self.emg_buffer[0]) >= self.emg_window_samples
        imu_ready = len(self.imu_buffer[0]) >= self.imu_window_samples
        return emg_ready and imu_ready
    
    def get_buffer_status(self) -> Dict:
        """获取缓冲区状态
        
        Returns:
            状态字典
        """
        return {
            'emg_samples': len(self.emg_buffer[0]),
            'imu_samples': len(self.imu_buffer[0]),
            'emg_target': self.emg_window_samples,
            'imu_target': self.imu_window_samples,
            'emg_ready': len(self.emg_buffer[0]) >= self.emg_window_samples,
            'imu_ready': len(self.imu_buffer[0]) >= self.imu_window_samples,
            'window_ready': self.is_window_ready()
        }
    
    def clear_buffer(self):
        """清空数据缓冲区"""
        for i in range(8):
            self.emg_buffer[i].clear()
        for i in range(6):
            self.imu_buffer[i].clear()
        self.emg_counter = 0
        self.prediction_history.clear()
        self.confidence_history.clear()
        self._stable_prediction = None
        logger.info("缓冲区已清空")
    
    # ==================== 预测功能 ====================
    
    def _predict(self) -> Optional[Dict]:
        """执行预测（支持集成预测）
        
        Returns:
            预测结果字典
        """
        if self.use_ensemble and self.classifier_manager:
            return self._predict_ensemble()
        else:
            return self._predict_single()
    
    def _predict_single(self) -> Optional[Dict]:
        """单模型预测
        
        Returns:
            预测结果字典
        """
        if self.classifier is None or not self.classifier.is_trained:
            logger.warning("分类器未加载或未训练")
            return None
        
        if not self.is_window_ready():
            return None
        
        start_time = time.time()
        
        emg_data = np.array([list(self.emg_buffer[i]) for i in range(8)])
        imu_data = np.array([list(self.imu_buffer[i]) for i in range(6)])
        
        features = self._extract_features_with_cache(emg_data, imu_data)
        
        try:
            result = self.classifier.predict_single(features)
            prediction = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            
            confidence_passed = confidence >= self.confidence_threshold
            if not confidence_passed:
                logger.debug(f"置信度 {confidence:.2f} 低于阈值 {self.confidence_threshold}, "
                           f"预测结果 '{prediction}' 被过滤")
            
            if confidence_passed:
                self.prediction_history.append(prediction)
            else:
                self.prediction_history.append(None)
            self.confidence_history.append(confidence)
            
            smoothed_prediction, smoothed_confidence = self._smooth_results()
            stabilized_prediction, switch_blocked = self._apply_switch_hysteresis(
                smoothed_prediction
            )
            
            stability = self._calculate_stability()
            
            self.prediction_count += 1
            self.last_prediction = stabilized_prediction
            self.last_confidence = smoothed_confidence
            self.last_prediction_time = datetime.now()
            
            prediction_time = (time.time() - start_time) * 1000
            
            result_dict = {
                'gesture': stabilized_prediction,
                'confidence': smoothed_confidence,
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'probabilities': probabilities,
                'labels': result['labels'],
                'prediction_time_ms': prediction_time,
                'timestamp': datetime.now().isoformat(),
                'prediction_count': self.prediction_count,
                'confidence_passed': confidence_passed,
                'confidence_threshold': self.confidence_threshold,
                'stability': stability,
                'switch_blocked': switch_blocked,
                'method': 'single',
                'algorithm': self.classifier.algorithm if self.classifier else 'unknown',
                'window_info': {
                    'emg_samples': self.emg_window_samples,
                    'imu_samples': self.imu_window_samples,
                    'window_ms': self.window_size_ms
                }
            }
            
            if self._prediction_callback:
                self._prediction_callback(result_dict)
            
            logger.debug(f"预测: {stabilized_prediction}, 置信度: {smoothed_confidence:.2f}, "
                        f"稳定性: {stability:.2f}, 耗时: {prediction_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None
    
    def _predict_ensemble(self) -> Optional[Dict]:
        """集成预测：综合多个模型
        
        Returns:
            预测结果字典
        """
        if self.classifier_manager is None:
            logger.warning("分类器管理器未设置")
            return self._predict_single()
        
        if not self.is_window_ready():
            return None
        
        start_time = time.time()
        
        emg_data = np.array([list(self.emg_buffer[i]) for i in range(8)])
        imu_data = np.array([list(self.imu_buffer[i]) for i in range(6)])
        
        features = self._extract_features_with_cache(emg_data, imu_data)
        
        try:
            result = self.classifier_manager.predict_single_ensemble(
                features, 
                strategy=self.ensemble_strategy
            )
            
            prediction = result['prediction']
            confidence = result['confidence']
            probabilities = result['probabilities']
            method = result.get('method', 'unknown')
            participating_models = result.get('participating_models', [])
            
            confidence_passed = confidence >= self.confidence_threshold
            
            if confidence_passed:
                self.prediction_history.append(prediction)
            else:
                self.prediction_history.append(None)
            self.confidence_history.append(confidence)
            
            smoothed_prediction, smoothed_confidence = self._smooth_results()
            stabilized_prediction, switch_blocked = self._apply_switch_hysteresis(
                smoothed_prediction
            )
            
            stability = self._calculate_stability()
            
            self.prediction_count += 1
            self.last_prediction = stabilized_prediction
            self.last_confidence = smoothed_confidence
            self.last_prediction_time = datetime.now()
            
            prediction_time = (time.time() - start_time) * 1000
            
            result_dict = {
                'gesture': stabilized_prediction,
                'confidence': smoothed_confidence,
                'raw_prediction': prediction,
                'raw_confidence': confidence,
                'probabilities': probabilities,
                'labels': result.get('labels'),
                'prediction_time_ms': prediction_time,
                'timestamp': datetime.now().isoformat(),
                'prediction_count': self.prediction_count,
                'confidence_passed': confidence_passed,
                'confidence_threshold': self.confidence_threshold,
                'stability': stability,
                'switch_blocked': switch_blocked,
                'method': method,
                'ensemble_strategy': self.ensemble_strategy,
                'participating_models': participating_models,
                'model_weights': self.classifier_manager.model_weights if self.classifier_manager else {},
                'best_algorithm': self.classifier_manager.best_algorithm if self.classifier_manager else None,
                'window_info': {
                    'emg_samples': self.emg_window_samples,
                    'imu_samples': self.imu_window_samples,
                    'window_ms': self.window_size_ms
                }
            }
            
            if self._prediction_callback:
                self._prediction_callback(result_dict)
            
            logger.debug(f"集成预测: {stabilized_prediction}, 置信度: {smoothed_confidence:.2f}, "
                        f"策略: {method}, 稳定性: {stability:.2f}, 耗时: {prediction_time:.1f}ms")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            return self._predict_single()
    
    def _smooth_results(self) -> Tuple[str, float]:
        """平滑预测结果
        
        Returns:
            平滑后的预测和置信度
        """
        if len(self.prediction_history) == 0:
            return None, 0.0
        
        # 过滤掉None值（低置信度标记）
        valid_predictions = [p for p in self.prediction_history if p is not None]
        
        if len(valid_predictions) == 0:
            return None, 0.0
        
        # 多数投票
        from collections import Counter
        vote_counts = Counter(valid_predictions)
        most_common = vote_counts.most_common(1)[0]
        smoothed_prediction = most_common[0]
        
        # 计算平滑后的置信度（加权平均）
        if len(self.confidence_history) > 0:
            weights = np.exp(np.linspace(0, 1, len(self.confidence_history)))
            weights = weights / weights.sum()
            smoothed_confidence = float(np.average(
                list(self.confidence_history), 
                weights=weights
            ))
        else:
            smoothed_confidence = 0.0
        
        return smoothed_prediction, smoothed_confidence
    
    def _calculate_stability(self) -> float:
        """计算预测结果的稳定性
        
        基于投票比例评估结果稳定性
        
        Returns:
            稳定性值 (0-1)，越高越稳定
        """
        if len(self.prediction_history) == 0:
            return 0.0
        
        # 过滤掉None值
        valid_predictions = [p for p in self.prediction_history if p is not None]
        
        if len(valid_predictions) == 0:
            return 0.0
        
        # 计算投票比例
        from collections import Counter
        vote_counts = Counter(valid_predictions)
        most_common_count = vote_counts.most_common(1)[0][1]
        
        # 稳定性 = 最高票数 / 总有效票数
        stability = most_common_count / len(valid_predictions)
        
        # 考虑置信度方差
        if len(self.confidence_history) > 1:
            confidence_std = np.std(list(self.confidence_history))
            # 置信度方差越小，稳定性越高
            confidence_factor = max(0, 1 - confidence_std)
            stability = stability * 0.7 + confidence_factor * 0.3
        
        return stability

    def _apply_switch_hysteresis(self, prediction: Optional[str]) -> Tuple[Optional[str], bool]:
        """应用手势切换迟滞，减少边界抖动造成的误切换。

        Args:
            prediction: 平滑后的候选预测

        Returns:
            (稳定后的预测, 是否阻止了切换)
        """
        if prediction is None:
            self._stable_prediction = None
            return None, False

        valid_predictions = [p for p in self.prediction_history if p is not None]
        if not valid_predictions:
            return prediction, False

        # 首次预测直接建立稳定状态
        if self._stable_prediction is None:
            self._stable_prediction = prediction
            return prediction, False

        # 未发生切换，直接接受
        if prediction == self._stable_prediction:
            return prediction, False

        # 只有当新手势在窗口中占比足够高时才允许切换
        switch_ratio = valid_predictions.count(prediction) / len(valid_predictions)
        if switch_ratio >= self.switch_stability_threshold:
            self._stable_prediction = prediction
            return prediction, False

        return self._stable_prediction, True
    
    def _extract_features_with_cache(self, emg_data: np.ndarray, imu_data: np.ndarray) -> Dict:
        """使用缓存优化的特征提取
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            
        Returns:
            特征字典
        """
        if not self.feature_cache_enabled:
            return self.feature_extractor.extract_all_features(emg_data, imu_data)
        
        # 计算数据哈希（用于缓存键）
        window_hash = self._compute_window_hash(emg_data, imu_data)
        
        # 检查缓存
        if window_hash in self.feature_cache:
            logger.debug("使用缓存的特征")
            return self.feature_cache[window_hash]
        
        # 提取特征
        features = self.feature_extractor.extract_all_features(emg_data, imu_data)
        
        # 更新缓存
        self._update_feature_cache(window_hash, features)
        
        return features
    
    def _compute_window_hash(self, emg_data: np.ndarray, imu_data: np.ndarray) -> str:
        """计算窗口数据哈希
        
        Args:
            emg_data: EMG数据
            imu_data: IMU数据
            
        Returns:
            哈希字符串
        """
        # 使用数据的统计特征作为哈希（比完整数据更高效）
        emg_hash = hash((
            round(np.mean(emg_data), 4),
            round(np.std(emg_data), 4),
            round(np.min(emg_data), 4),
            round(np.max(emg_data), 4),
            emg_data.shape
        ))
        imu_hash = hash((
            round(np.mean(imu_data), 4),
            round(np.std(imu_data), 4),
            round(np.min(imu_data), 4),
            round(np.max(imu_data), 4),
            imu_data.shape
        ))
        return f"{emg_hash}_{imu_hash}"
    
    def _update_feature_cache(self, window_hash: str, features: Dict):
        """更新特征缓存
        
        Args:
            window_hash: 窗口哈希
            features: 特征字典
        """
        # 清理过期缓存
        if len(self.feature_cache) >= self.feature_cache_max_size:
            # 删除最旧的缓存项
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        self.feature_cache[window_hash] = features
    
    def enable_feature_cache(self, enabled: bool = True):
        """启用/禁用特征缓存
        
        Args:
            enabled: 是否启用
        """
        self.feature_cache_enabled = enabled
        if not enabled:
            self.feature_cache.clear()
        logger.info(f"特征缓存已{'启用' if enabled else '禁用'}")
    
    def clear_feature_cache(self):
        """清空特征缓存"""
        self.feature_cache.clear()
        self._incremental_stats.clear()
        self._last_window_hash = None
        logger.info("特征缓存已清空")
    
    def predict_now(self) -> Optional[Dict]:
        """立即执行预测（不考虑滑动步长）
        
        Returns:
            预测结果
        """
        if not self.is_window_ready():
            logger.warning("窗口未就绪，无法预测")
            return None
        return self._predict()
    
    # ==================== 回调设置 ====================
    
    def set_prediction_callback(self, callback: Callable[[Dict], None]):
        """设置预测回调函数
        
        Args:
            callback: 回调函数，接收预测结果字典
        """
        self._prediction_callback = callback
    
    def set_window_ready_callback(self, callback: Callable[[Dict], None]):
        """设置窗口就绪回调函数
        
        Args:
            callback: 回调函数，接收缓冲区状态
        """
        self._window_ready_callback = callback
    
    # ==================== 状态管理 ====================
    
    def start(self):
        """启动识别器"""
        self.is_running = True
        self.clear_buffer()
        logger.info("实时识别器已启动")
    
    def stop(self):
        """停止识别器"""
        self.is_running = False
        logger.info("实时识别器已停止")
    
    def reset(self):
        """重置识别器状态"""
        self.clear_buffer()
        self.prediction_count = 0
        self.last_prediction = None
        self.last_confidence = 0.0
        self.last_prediction_time = None
        self._stable_prediction = None
        logger.info("识别器已重置")
    
    def get_status(self) -> Dict:
        """获取识别器状态
        
        Returns:
            状态字典
        """
        status = {
            'is_running': self.is_running,
            'model_loaded': self.classifier is not None and self.classifier.is_trained,
            'window_ready': self.is_window_ready(),
            'buffer_status': self.get_buffer_status(),
            'prediction_count': self.prediction_count,
            'last_prediction': self.last_prediction,
            'last_confidence': self.last_confidence,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'ensemble_mode': self.use_ensemble,
            'ensemble_strategy': self.ensemble_strategy if self.use_ensemble else None,
            'config': {
                'window_size_ms': self.window_size_ms,
                'emg_sample_rate': self.emg_sample_rate,
                'imu_sample_rate': self.imu_sample_rate,
                'slide_step': self.slide_step,
                'confidence_threshold': self.confidence_threshold,
                'smoothing_window': self.smoothing_window,
                'switch_stability_threshold': self.switch_stability_threshold
            }
        }
        
        if self.classifier_manager:
            status['ensemble_info'] = {
                'total_models': len(self.classifier_manager.classifiers),
                'best_algorithm': self.classifier_manager.best_algorithm,
                'model_weights': self.classifier_manager.model_weights,
                'available_algorithms': list(self.classifier_manager.classifiers.keys())
            }
        
        return status
    
    # ==================== 特征获取 ====================
    
    def get_current_features(self) -> Optional[Dict]:
        """获取当前窗口的特征
        
        Returns:
            特征字典
        """
        if not self.is_window_ready():
            return None
        
        emg_data = np.array([list(self.emg_buffer[i]) for i in range(8)])
        imu_data = np.array([list(self.imu_buffer[i]) for i in range(6)])
        
        return self.feature_extractor.extract_all_features(emg_data, imu_data)
    
    def get_current_window_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取当前窗口的原始数据
        
        Returns:
            EMG数据和IMU数据
        """
        if not self.is_window_ready():
            return None, None
        
        emg_data = np.array([list(self.emg_buffer[i]) for i in range(8)])
        imu_data = np.array([list(self.imu_buffer[i]) for i in range(6)])
        
        return emg_data, imu_data


class RealtimeRecognizerManager:
    """实时识别器管理器
    
    管理识别器的生命周期和配置
    """
    
    def __init__(self, models_dir: str = 'models'):
        """初始化管理器
        
        Args:
            models_dir: 模型目录
        """
        self.models_dir = models_dir
        self.recognizer = None
        self.available_models = []
        
        self._scan_models()
        
        logger.info(f"识别器管理器初始化: 模型目录={models_dir}")
    
    def _scan_models(self):
        """扫描可用模型"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return
        
        self.available_models = [
            f for f in os.listdir(self.models_dir) 
            if f.endswith('.pkl')
        ]
    
    def create_recognizer(self, **kwargs) -> RealtimeGestureRecognizer:
        """创建识别器
        
        Args:
            **kwargs: 识别器参数
            
        Returns:
            识别器实例
        """
        self.recognizer = RealtimeGestureRecognizer(**kwargs)
        return self.recognizer
    
    def get_recognizer(self) -> Optional[RealtimeGestureRecognizer]:
        """获取当前识别器
        
        Returns:
            识别器实例
        """
        return self.recognizer
    
    def list_models(self) -> List[str]:
        """列出可用模型
        
        Returns:
            模型文件列表
        """
        self._scan_models()
        return self.available_models
    
    def load_model(self, model_name: str) -> bool:
        """加载模型
        
        Args:
            model_name: 模型文件名
            
        Returns:
            是否成功
        """
        if self.recognizer is None:
            logger.warning("识别器未创建")
            return False
        
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        self.recognizer.load_model(model_path)
        return self.recognizer.classifier is not None


# 全局识别器实例
_realtime_recognizer = None

def get_realtime_recognizer(**kwargs) -> RealtimeGestureRecognizer:
    """获取全局实时识别器实例
    
    Args:
        **kwargs: 识别器参数
        
    Returns:
        RealtimeGestureRecognizer实例
    """
    global _realtime_recognizer
    if _realtime_recognizer is None:
        _realtime_recognizer = RealtimeGestureRecognizer(**kwargs)
    return _realtime_recognizer
