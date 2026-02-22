#!/usr/bin/env python3
"""
性能优化模块
提供计算效率优化、内存管理和性能监控功能
"""

import numpy as np
import time
import os
import sys
import psutil
import threading
from functools import wraps
from typing import Callable, Dict, List, Optional
from collections import deque

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger


class PerformanceMonitor:
    """性能监控器
    
    监控系统资源使用和性能指标
    """
    
    def __init__(self, history_size: int = 100):
        """初始化性能监控器
        
        Args:
            history_size: 历史记录大小
        """
        self.history_size = history_size
        
        # CPU使用率历史
        self.cpu_history = deque(maxlen=history_size)
        
        # 内存使用历史
        self.memory_history = deque(maxlen=history_size)
        
        # 处理时间历史
        self.processing_times = deque(maxlen=history_size)
        
        # 预测延迟历史
        self.prediction_latencies = deque(maxlen=history_size)
        
        # 开始时间
        self.start_time = None
        
        # 监控线程
        self._monitor_thread = None
        self._monitoring = False
        
        logger.info("性能监控器初始化完成")
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        self.start_time = time.time()
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                self.cpu_history.append(psutil.cpu_percent())
                self.memory_history.append(psutil.virtual_memory().percent)
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info(f"性能监控已启动，间隔: {interval}s")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("性能监控已停止")
    
    def record_processing_time(self, duration_ms: float):
        """记录处理时间
        
        Args:
            duration_ms: 处理时间（毫秒）
        """
        self.processing_times.append(duration_ms)
    
    def record_prediction_latency(self, latency_ms: float):
        """记录预测延迟
        
        Args:
            latency_ms: 预测延迟（毫秒）
        """
        self.prediction_latencies.append(latency_ms)
    
    def get_statistics(self) -> Dict:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'cpu': {
                'current': self.cpu_history[-1] if self.cpu_history else 0,
                'avg': np.mean(self.cpu_history) if self.cpu_history else 0,
                'max': np.max(self.cpu_history) if self.cpu_history else 0
            },
            'memory': {
                'current': self.memory_history[-1] if self.memory_history else 0,
                'avg': np.mean(self.memory_history) if self.memory_history else 0,
                'max': np.max(self.memory_history) if self.memory_history else 0
            },
            'processing': {
                'count': len(self.processing_times),
                'avg_ms': np.mean(self.processing_times) if self.processing_times else 0,
                'max_ms': np.max(self.processing_times) if self.processing_times else 0,
                'min_ms': np.min(self.processing_times) if self.processing_times else 0
            },
            'prediction': {
                'count': len(self.prediction_latencies),
                'avg_ms': np.mean(self.prediction_latencies) if self.prediction_latencies else 0,
                'max_ms': np.max(self.prediction_latencies) if self.prediction_latencies else 0,
                'min_ms': np.min(self.prediction_latencies) if self.prediction_latencies else 0
            }
        }
        
        return stats
    
    def get_status(self) -> Dict:
        """获取当前状态
        
        Returns:
            状态字典
        """
        return {
            'is_monitoring': self._monitoring,
            'cpu_percent': self.cpu_history[-1] if self.cpu_history else 0,
            'memory_percent': self.memory_history[-1] if self.memory_history else 0,
            'process_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }


def timing_decorator(func: Callable) -> Callable:
    """计时装饰器
    
    用于测量函数执行时间
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start) * 1000
        logger.debug(f"{func.__name__} 执行时间: {duration:.2f}ms")
        return result
    return wrapper


class OptimizedFeatureExtractor:
    """优化的特征提取器
    
    使用NumPy向量化计算提高效率
    """
    
    def __init__(self):
        """初始化优化特征提取器"""
        self._cache = {}
        logger.info("优化特征提取器初始化")
    
    def extract_time_domain_features_vectorized(self, data: np.ndarray) -> Dict:
        """向量化时域特征提取
        
        Args:
            data: 数据数组 (channels, samples)
            
        Returns:
            特征字典
        """
        n_channels, n_samples = data.shape
        
        # 使用NumPy向量化计算
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        var = np.var(data, axis=1)
        rms = np.sqrt(np.mean(data ** 2, axis=1))
        mav = np.mean(np.abs(data), axis=1)
        peak_to_peak = np.max(data, axis=1) - np.min(data, axis=1)
        iemg = np.sum(np.abs(data), axis=1)
        wl = np.sum(np.abs(np.diff(data, axis=1)), axis=1)
        
        features = {}
        for i in range(n_channels):
            prefix = f'ch_{i}'
            features.update({
                f'{prefix}_mean': mean[i],
                f'{prefix}_std': std[i],
                f'{prefix}_var': var[i],
                f'{prefix}_rms': rms[i],
                f'{prefix}_mav': mav[i],
                f'{prefix}_p2p': peak_to_peak[i],
                f'{prefix}_iemg': iemg[i],
                f'{prefix}_wl': wl[i]
            })
        
        return features
    
    def extract_fft_features_vectorized(self, data: np.ndarray, sample_rate: float) -> Dict:
        """向量化FFT特征提取
        
        Args:
            data: 数据数组 (channels, samples)
            sample_rate: 采样率
            
        Returns:
            特征字典
        """
        n_channels, n_samples = data.shape
        
        # 批量FFT
        fft_results = np.fft.fft(data, axis=1)
        frequencies = np.fft.fftfreq(n_samples, 1.0 / sample_rate)[:n_samples // 2]
        power_spectrum = np.abs(fft_results[:, :n_samples // 2]) ** 2 / n_samples
        
        # 向量化计算频域特征
        total_power = np.sum(power_spectrum, axis=1)
        
        features = {}
        for i in range(n_channels):
            prefix = f'ch_{i}'
            ps = power_spectrum[i]
            tp = total_power[i]
            
            if tp > 0:
                # 中值频率
                cumsum = np.cumsum(ps)
                mf_idx = np.searchsorted(cumsum, tp / 2)
                mf = frequencies[min(mf_idx, len(frequencies) - 1)]
                
                # 平均功率频率
                mpf = np.sum(frequencies * ps) / tp
                
                # 峰值频率
                pf = frequencies[np.argmax(ps)]
                
                # 频谱能量
                energy = tp
                
                features.update({
                    f'{prefix}_median_freq': mf,
                    f'{prefix}_mean_power_freq': mpf,
                    f'{prefix}_peak_freq': pf,
                    f'{prefix}_spectral_energy': energy
                })
            else:
                features.update({
                    f'{prefix}_median_freq': 0,
                    f'{prefix}_mean_power_freq': 0,
                    f'{prefix}_peak_freq': 0,
                    f'{prefix}_spectral_energy': 0
                })
        
        return features


class MemoryManager:
    """内存管理器
    
    管理固定大小缓冲区和内存释放
    """
    
    def __init__(self, max_memory_mb: float = 500):
        """初始化内存管理器
        
        Args:
            max_memory_mb: 最大内存使用（MB）
        """
        self.max_memory_mb = max_memory_mb
        self.buffers = {}
        
        logger.info(f"内存管理器初始化，最大内存: {max_memory_mb}MB")
    
    def create_buffer(self, name: str, shape: tuple, dtype: np.dtype = np.float64) -> np.ndarray:
        """创建固定大小缓冲区
        
        Args:
            name: 缓冲区名称
            shape: 缓冲区形状
            dtype: 数据类型
            
        Returns:
            缓冲区数组
        """
        buffer = np.zeros(shape, dtype=dtype)
        self.buffers[name] = buffer
        
        size_mb = buffer.nbytes / 1024 / 1024
        logger.info(f"创建缓冲区 '{name}': shape={shape}, size={size_mb:.2f}MB")
        
        return buffer
    
    def get_buffer(self, name: str) -> Optional[np.ndarray]:
        """获取缓冲区
        
        Args:
            name: 缓冲区名称
            
        Returns:
            缓冲区数组
        """
        return self.buffers.get(name)
    
    def clear_buffer(self, name: str):
        """清空缓冲区
        
        Args:
            name: 缓冲区名称
        """
        if name in self.buffers:
            self.buffers[name].fill(0)
    
    def release_buffer(self, name: str):
        """释放缓冲区
        
        Args:
            name: 缓冲区名称
        """
        if name in self.buffers:
            del self.buffers[name]
            logger.info(f"释放缓冲区 '{name}'")
    
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况
        
        Returns:
            内存使用字典
        """
        total_size = sum(buf.nbytes for buf in self.buffers.values())
        
        return {
            'total_buffers': len(self.buffers),
            'total_size_mb': total_size / 1024 / 1024,
            'buffers': {name: buf.nbytes / 1024 / 1024 for name, buf in self.buffers.items()},
            'process_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
    
    def check_memory(self) -> bool:
        """检查内存使用是否超限
        
        Returns:
            是否在限制内
        """
        current_mb = psutil.Process().memory_info().rss / 1024 / 1024
        return current_mb < self.max_memory_mb


class ThreadPool:
    """简单线程池
    
    管理后台任务执行
    """
    
    def __init__(self, max_workers: int = 4):
        """初始化线程池
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.workers = []
        self.tasks = []
        self.lock = threading.Lock()
        
        logger.info(f"线程池初始化，最大工作线程: {max_workers}")
    
    def submit(self, func: Callable, *args, **kwargs):
        """提交任务
        
        Args:
            func: 任务函数
            *args: 位置参数
            **kwargs: 关键字参数
        """
        def task_wrapper():
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.error(f"任务执行错误: {e}")
        
        thread = threading.Thread(target=task_wrapper, daemon=True)
        thread.start()
        self.workers.append(thread)
        
        # 清理已完成的工作线程
        self.workers = [w for w in self.workers if w.is_alive()]


class DataAcquisitionThread:
    """数据采集专用线程
    
    负责从数据源持续采集数据
    """
    
    def __init__(self, callback: Callable = None, buffer_size: int = 10000):
        """初始化数据采集线程
        
        Args:
            callback: 数据回调函数
            buffer_size: 缓冲区大小
        """
        self.callback = callback
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        
        self.samples_collected = 0
        self.last_sample_time = None
        
        logger.info("数据采集线程初始化")
    
    def start(self):
        """启动采集线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()
        logger.info("数据采集线程已启动")
    
    def stop(self):
        """停止采集线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("数据采集线程已停止")
    
    def _acquisition_loop(self):
        """采集循环"""
        while self.running:
            try:
                if self.callback:
                    data = self.callback()
                    if data is not None:
                        with self.lock:
                            self.buffer.append(data)
                            self.samples_collected += 1
                            self.last_sample_time = time.time()
            except Exception as e:
                logger.error(f"数据采集错误: {e}")
                time.sleep(0.01)
    
    def get_data(self, n_samples: int = None) -> List:
        """获取采集的数据
        
        Args:
            n_samples: 获取的样本数，None表示全部
            
        Returns:
            数据列表
        """
        with self.lock:
            if n_samples is None:
                return list(self.buffer)
            return list(self.buffer)[-n_samples:]
    
    def clear_buffer(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            'running': self.running,
            'buffer_size': len(self.buffer),
            'samples_collected': self.samples_collected,
            'last_sample_time': self.last_sample_time
        }


class FeatureExtractionThread:
    """特征提取专用线程
    
    负责从数据缓冲区提取特征
    """
    
    def __init__(self, feature_callback: Callable = None):
        """初始化特征提取线程
        
        Args:
            feature_callback: 特征提取回调函数
        """
        self.feature_callback = feature_callback
        self.thread = None
        self.running = False
        self.data_queue = deque(maxlen=100)
        self.feature_queue = deque(maxlen=100)
        self.lock = threading.Lock()
        
        self.features_extracted = 0
        self.extraction_times = deque(maxlen=100)
        
        logger.info("特征提取线程初始化")
    
    def start(self):
        """启动特征提取线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._extraction_loop, daemon=True)
        self.thread.start()
        logger.info("特征提取线程已启动")
    
    def stop(self):
        """停止特征提取线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("特征提取线程已停止")
    
    def submit_data(self, data):
        """提交数据用于特征提取
        
        Args:
            data: 待处理的数据
        """
        with self.lock:
            self.data_queue.append(data)
    
    def _extraction_loop(self):
        """特征提取循环"""
        while self.running:
            try:
                data = None
                with self.lock:
                    if self.data_queue:
                        data = self.data_queue.popleft()
                
                if data is not None and self.feature_callback:
                    start_time = time.time()
                    features = self.feature_callback(data)
                    extraction_time = (time.time() - start_time) * 1000
                    
                    with self.lock:
                        self.feature_queue.append(features)
                        self.features_extracted += 1
                        self.extraction_times.append(extraction_time)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"特征提取错误: {e}")
                time.sleep(0.01)
    
    def get_features(self) -> Optional[Dict]:
        """获取提取的特征
        
        Returns:
            特征字典
        """
        with self.lock:
            if self.feature_queue:
                return self.feature_queue.popleft()
            return None
    
    def get_status(self) -> Dict:
        """获取状态"""
        with self.lock:
            avg_time = np.mean(self.extraction_times) if self.extraction_times else 0
            return {
                'running': self.running,
                'pending_data': len(self.data_queue),
                'pending_features': len(self.feature_queue),
                'features_extracted': self.features_extracted,
                'avg_extraction_time_ms': avg_time
            }


class PredictionThread:
    """预测专用线程
    
    负责执行模型预测
    """
    
    def __init__(self, predict_callback: Callable = None):
        """初始化预测线程
        
        Args:
            predict_callback: 预测回调函数
        """
        self.predict_callback = predict_callback
        self.thread = None
        self.running = False
        self.feature_queue = deque(maxlen=50)
        self.result_queue = deque(maxlen=50)
        self.lock = threading.Lock()
        
        self.predictions_made = 0
        self.prediction_times = deque(maxlen=100)
        
        logger.info("预测线程初始化")
    
    def set_predict_callback(self, callback: Callable):
        """设置预测回调函数
        
        Args:
            callback: 预测回调函数
        """
        self.predict_callback = callback
    
    def start(self):
        """启动预测线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.thread.start()
        logger.info("预测线程已启动")
    
    def stop(self):
        """停止预测线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("预测线程已停止")
    
    def submit_features(self, features: Dict):
        """提交特征用于预测
        
        Args:
            features: 特征字典
        """
        with self.lock:
            self.feature_queue.append(features)
    
    def _prediction_loop(self):
        """预测循环"""
        while self.running:
            try:
                features = None
                with self.lock:
                    if self.feature_queue:
                        features = self.feature_queue.popleft()
                
                if features is not None and self.predict_callback:
                    start_time = time.time()
                    result = self.predict_callback(features)
                    prediction_time = (time.time() - start_time) * 1000
                    
                    with self.lock:
                        self.result_queue.append(result)
                        self.predictions_made += 1
                        self.prediction_times.append(prediction_time)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"预测错误: {e}")
                time.sleep(0.01)
    
    def get_result(self) -> Optional[Dict]:
        """获取预测结果
        
        Returns:
            预测结果字典
        """
        with self.lock:
            if self.result_queue:
                return self.result_queue.popleft()
            return None
    
    def get_status(self) -> Dict:
        """获取状态"""
        with self.lock:
            avg_time = np.mean(self.prediction_times) if self.prediction_times else 0
            return {
                'running': self.running,
                'pending_features': len(self.feature_queue),
                'pending_results': len(self.result_queue),
                'predictions_made': self.predictions_made,
                'avg_prediction_time_ms': avg_time
            }


class UIUpdateThread:
    """UI更新专用线程
    
    负责管理UI更新频率，避免频繁更新
    """
    
    def __init__(self, update_interval_ms: float = 100):
        """初始化UI更新线程
        
        Args:
            update_interval_ms: 更新间隔（毫秒）
        """
        self.update_interval_ms = update_interval_ms
        self.thread = None
        self.running = False
        self.update_callbacks = {}
        self.pending_updates = {}
        self.lock = threading.Lock()
        
        self.updates_performed = 0
        self.updates_skipped = 0
        
        logger.info(f"UI更新线程初始化，更新间隔: {update_interval_ms}ms")
    
    def register_callback(self, name: str, callback: Callable):
        """注册更新回调
        
        Args:
            name: 回调名称
            callback: 回调函数
        """
        with self.lock:
            self.update_callbacks[name] = callback
    
    def unregister_callback(self, name: str):
        """注销更新回调
        
        Args:
            name: 回调名称
        """
        with self.lock:
            if name in self.update_callbacks:
                del self.update_callbacks[name]
    
    def request_update(self, name: str, data=None):
        """请求UI更新
        
        Args:
            name: 更新名称
            data: 更新数据
        """
        with self.lock:
            if name in self.pending_updates:
                self.updates_skipped += 1
            self.pending_updates[name] = data
    
    def start(self):
        """启动UI更新线程"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        logger.info("UI更新线程已启动")
    
    def stop(self):
        """停止UI更新线程"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("UI更新线程已停止")
    
    def _update_loop(self):
        """更新循环"""
        while self.running:
            try:
                updates_to_process = {}
                with self.lock:
                    if self.pending_updates:
                        updates_to_process = self.pending_updates.copy()
                        self.pending_updates.clear()
                
                for name, data in updates_to_process.items():
                    callback = self.update_callbacks.get(name)
                    if callback:
                        try:
                            callback(data)
                            self.updates_performed += 1
                        except Exception as e:
                            logger.error(f"UI更新错误 [{name}]: {e}")
                
                time.sleep(self.update_interval_ms / 1000)
            except Exception as e:
                logger.error(f"UI更新循环错误: {e}")
                time.sleep(0.01)
    
    def get_status(self) -> Dict:
        """获取状态"""
        with self.lock:
            return {
                'running': self.running,
                'registered_callbacks': list(self.update_callbacks.keys()),
                'pending_updates': len(self.pending_updates),
                'updates_performed': self.updates_performed,
                'updates_skipped': self.updates_skipped
            }


class DedicatedThreadManager:
    """专用线程管理器
    
    统一管理所有专用线程
    """
    
    def __init__(self):
        """初始化专用线程管理器"""
        self.acquisition_thread = DataAcquisitionThread()
        self.feature_thread = FeatureExtractionThread()
        self.prediction_thread = PredictionThread()
        self.ui_thread = UIUpdateThread()
        
        self.is_running = False
        
        logger.info("专用线程管理器初始化")
    
    def start_all(self):
        """启动所有线程"""
        self.acquisition_thread.start()
        self.feature_thread.start()
        self.prediction_thread.start()
        self.ui_thread.start()
        self.is_running = True
        logger.info("所有专用线程已启动")
    
    def stop_all(self):
        """停止所有线程"""
        self.acquisition_thread.stop()
        self.feature_thread.stop()
        self.prediction_thread.stop()
        self.ui_thread.stop()
        self.is_running = False
        logger.info("所有专用线程已停止")
    
    def get_status(self) -> Dict:
        """获取所有线程状态"""
        return {
            'is_running': self.is_running,
            'acquisition': self.acquisition_thread.get_status(),
            'feature_extraction': self.feature_thread.get_status(),
            'prediction': self.prediction_thread.get_status(),
            'ui_update': self.ui_thread.get_status()
        }
    
    def set_acquisition_callback(self, callback: Callable):
        """设置数据采集回调"""
        self.acquisition_thread.callback = callback
    
    def set_feature_callback(self, callback: Callable):
        """设置特征提取回调"""
        self.feature_thread.feature_callback = callback
    
    def set_prediction_callback(self, callback: Callable):
        """设置预测回调"""
        self.prediction_thread.set_predict_callback(callback)
    
    def register_ui_callback(self, name: str, callback: Callable):
        """注册UI更新回调"""
        self.ui_thread.register_callback(name, callback)


# 全局性能监控器
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器
    
    Returns:
        PerformanceMonitor实例
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def benchmark_feature_extraction(n_samples: int = 1000, n_channels: int = 8, 
                                  n_iterations: int = 100) -> Dict:
    """特征提取性能基准测试
    
    Args:
        n_samples: 样本数
        n_channels: 通道数
        n_iterations: 迭代次数
        
    Returns:
        基准测试结果
    """
    from src.feature_extractor import FeatureExtractor
    
    extractor = FeatureExtractor()
    optimizer = OptimizedFeatureExtractor()
    
    # 生成测试数据
    data = np.random.randn(n_channels, n_samples)
    
    # 测试原始方法
    times_original = []
    for _ in range(n_iterations):
        start = time.time()
        for i in range(n_channels):
            extractor.extract_channel_features(data[i], 250.0, f'ch_{i}')
        times_original.append((time.time() - start) * 1000)
    
    # 测试优化方法
    times_optimized = []
    for _ in range(n_iterations):
        start = time.time()
        optimizer.extract_time_domain_features_vectorized(data)
        optimizer.extract_fft_features_vectorized(data, 250.0)
        times_optimized.append((time.time() - start) * 1000)
    
    return {
        'original': {
            'avg_ms': np.mean(times_original),
            'std_ms': np.std(times_original),
            'max_ms': np.max(times_original),
            'min_ms': np.min(times_original)
        },
        'optimized': {
            'avg_ms': np.mean(times_optimized),
            'std_ms': np.std(times_optimized),
            'max_ms': np.max(times_optimized),
            'min_ms': np.min(times_optimized)
        },
        'speedup': np.mean(times_original) / np.mean(times_optimized)
    }


if __name__ == "__main__":
    # 运行基准测试
    print("=" * 60)
    print("性能基准测试")
    print("=" * 60)
    
    print("\n特征提取性能测试...")
    results = benchmark_feature_extraction()
    
    print(f"\n原始方法:")
    print(f"  平均: {results['original']['avg_ms']:.2f}ms")
    print(f"  标准差: {results['original']['std_ms']:.2f}ms")
    
    print(f"\n优化方法:")
    print(f"  平均: {results['optimized']['avg_ms']:.2f}ms")
    print(f"  标准差: {results['optimized']['std_ms']:.2f}ms")
    
    print(f"\n加速比: {results['speedup']:.2f}x")
    
    print("\n内存管理器测试...")
    mm = MemoryManager()
    mm.create_buffer('emg', (8, 1000))
    mm.create_buffer('imu', (6, 500))
    print(f"内存使用: {mm.get_memory_usage()}")
