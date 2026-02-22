#!/usr/bin/env python3
"""
预定义手势库模块
定义常见手势、特征模板和默认配置
"""

import numpy as np
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger


class GestureCategory(Enum):
    """手势类别枚举"""
    STATIC = "static"      # 静态手势
    DYNAMIC = "dynamic"    # 动态手势
    SEQUENCE = "sequence"  # 序列手势


@dataclass
class GestureDefinition:
    """手势定义数据类"""
    name: str                          # 手势名称
    display_name: str                  # 显示名称
    category: GestureCategory          # 类别
    description: str                   # 描述
    emg_characteristics: Dict          # EMG特征描述
    imu_characteristics: Dict          # IMU特征描述
    difficulty: str                    # 难度：easy/medium/hard
    recommended_samples: int           # 推荐样本数
    tags: List[str]                    # 标签


class PredefinedGestures:
    """预定义手势库
    
    包含常见手势的定义和特征模板
    """
    
    GESTURES = {
        # 静态手势
        'fist': GestureDefinition(
            name='fist',
            display_name='握拳',
            category=GestureCategory.STATIC,
            description='手指握紧成拳',
            emg_characteristics={
                'pattern': 'high_amplitude',
                'channels': [0, 1, 2, 3, 4, 5, 6, 7],
                'description': '所有EMG通道高幅度激活'
            },
            imu_characteristics={
                'pattern': 'stable',
                'accel': 'low_variance',
                'gyro': 'low_variance',
                'description': 'IMU相对稳定'
            },
            difficulty='easy',
            recommended_samples=100,
            tags=['basic', 'static', 'grasp']
        ),
        
        'open': GestureDefinition(
            name='open',
            display_name='张开',
            category=GestureCategory.STATIC,
            description='手指全部张开',
            emg_characteristics={
                'pattern': 'medium_amplitude',
                'channels': [0, 2, 4, 6],
                'description': '伸肌群中等幅度激活'
            },
            imu_characteristics={
                'pattern': 'stable',
                'accel': 'low_variance',
                'gyro': 'low_variance',
                'description': 'IMU相对稳定'
            },
            difficulty='easy',
            recommended_samples=100,
            tags=['basic', 'static', 'extend']
        ),
        
        'pinch': GestureDefinition(
            name='pinch',
            display_name='捏合',
            category=GestureCategory.STATIC,
            description='拇指与食指捏合',
            emg_characteristics={
                'pattern': 'localized',
                'channels': [0, 1],
                'description': '特定通道高幅度激活'
            },
            imu_characteristics={
                'pattern': 'stable',
                'accel': 'low_variance',
                'gyro': 'low_variance',
                'description': 'IMU相对稳定'
            },
            difficulty='medium',
            recommended_samples=150,
            tags=['precision', 'static', 'grasp']
        ),
        
        'point': GestureDefinition(
            name='point',
            display_name='指向',
            category=GestureCategory.STATIC,
            description='食指伸出指向',
            emg_characteristics={
                'pattern': 'selective',
                'channels': [2, 3],
                'description': '食指相关通道激活'
            },
            imu_characteristics={
                'pattern': 'stable',
                'accel': 'low_variance',
                'gyro': 'low_variance',
                'description': 'IMU相对稳定'
            },
            difficulty='medium',
            recommended_samples=150,
            tags=['precision', 'static']
        ),
        
        'rest': GestureDefinition(
            name='rest',
            display_name='静止',
            category=GestureCategory.STATIC,
            description='手部自然放松',
            emg_characteristics={
                'pattern': 'baseline',
                'channels': [],
                'description': '所有通道低幅度'
            },
            imu_characteristics={
                'pattern': 'stable',
                'accel': 'low_variance',
                'gyro': 'low_variance',
                'description': 'IMU稳定'
            },
            difficulty='easy',
            recommended_samples=50,
            tags=['basic', 'static', 'neutral']
        ),
        
        # 动态手势
        'wave': GestureDefinition(
            name='wave',
            display_name='挥手',
            category=GestureCategory.DYNAMIC,
            description='手腕左右摆动',
            emg_characteristics={
                'pattern': 'periodic',
                'channels': [0, 1, 2, 3],
                'description': '周期性EMG变化'
            },
            imu_characteristics={
                'pattern': 'oscillating',
                'accel': 'x_axis_oscillation',
                'gyro': 'z_axis_rotation',
                'description': '加速度X轴振荡，陀螺仪Z轴旋转'
            },
            difficulty='medium',
            recommended_samples=150,
            tags=['dynamic', 'wrist']
        ),
        
        'flex': GestureDefinition(
            name='flex',
            display_name='手腕弯曲',
            category=GestureCategory.DYNAMIC,
            description='手腕向内弯曲',
            emg_characteristics={
                'pattern': 'transient',
                'channels': [0, 1],
                'description': '屈肌群激活'
            },
            imu_characteristics={
                'pattern': 'single_axis',
                'accel': 'y_axis_change',
                'gyro': 'x_axis_rotation',
                'description': '加速度Y轴变化'
            },
            difficulty='easy',
            recommended_samples=100,
            tags=['dynamic', 'wrist', 'flex']
        ),
        
        'extend': GestureDefinition(
            name='extend',
            display_name='手腕伸展',
            category=GestureCategory.DYNAMIC,
            description='手腕向外伸展',
            emg_characteristics={
                'pattern': 'transient',
                'channels': [2, 3],
                'description': '伸肌群激活'
            },
            imu_characteristics={
                'pattern': 'single_axis',
                'accel': 'y_axis_change_reverse',
                'gyro': 'x_axis_rotation_reverse',
                'description': '加速度Y轴反向变化'
            },
            difficulty='easy',
            recommended_samples=100,
            tags=['dynamic', 'wrist', 'extend']
        ),
        
        'pronation': GestureDefinition(
            name='pronation',
            display_name='前臂内旋',
            category=GestureCategory.DYNAMIC,
            description='前臂向内旋转',
            emg_characteristics={
                'pattern': 'transient',
                'channels': [4, 5],
                'description': '旋前肌群激活'
            },
            imu_characteristics={
                'pattern': 'rotation',
                'accel': 'stable',
                'gyro': 'z_axis_rotation',
                'description': '陀螺仪Z轴旋转'
            },
            difficulty='medium',
            recommended_samples=150,
            tags=['dynamic', 'forearm', 'rotation']
        ),
        
        'supination': GestureDefinition(
            name='supination',
            display_name='前臂外旋',
            category=GestureCategory.DYNAMIC,
            description='前臂向外旋转',
            emg_characteristics={
                'pattern': 'transient',
                'channels': [6, 7],
                'description': '旋后肌群激活'
            },
            imu_characteristics={
                'pattern': 'rotation',
                'accel': 'stable',
                'gyro': 'z_axis_rotation_reverse',
                'description': '陀螺仪Z轴反向旋转'
            },
            difficulty='medium',
            recommended_samples=150,
            tags=['dynamic', 'forearm', 'rotation']
        ),
        
        # 复杂手势
        'grab': GestureDefinition(
            name='grab',
            display_name='抓取',
            category=GestureCategory.SEQUENCE,
            description='从张开到握拳的抓取动作',
            emg_characteristics={
                'pattern': 'sequence',
                'channels': [0, 1, 2, 3, 4, 5, 6, 7],
                'description': '先伸肌后屈肌激活'
            },
            imu_characteristics={
                'pattern': 'transient',
                'accel': 'minor_change',
                'gyro': 'minor_change',
                'description': '轻微运动'
            },
            difficulty='hard',
            recommended_samples=200,
            tags=['complex', 'sequence', 'grasp']
        ),
        
        'release': GestureDefinition(
            name='release',
            display_name='释放',
            category=GestureCategory.SEQUENCE,
            description='从握拳到张开的释放动作',
            emg_characteristics={
                'pattern': 'sequence',
                'channels': [0, 1, 2, 3, 4, 5, 6, 7],
                'description': '先屈肌后伸肌激活'
            },
            imu_characteristics={
                'pattern': 'transient',
                'accel': 'minor_change',
                'gyro': 'minor_change',
                'description': '轻微运动'
            },
            difficulty='hard',
            recommended_samples=200,
            tags=['complex', 'sequence', 'release']
        )
    }
    
    @classmethod
    def get_gesture(cls, name: str) -> Optional[GestureDefinition]:
        """获取手势定义
        
        Args:
            name: 手势名称
            
        Returns:
            手势定义，如果不存在返回None
        """
        return cls.GESTURES.get(name)
    
    @classmethod
    def get_all_gestures(cls) -> Dict[str, GestureDefinition]:
        """获取所有手势定义
        
        Returns:
            手势定义字典
        """
        return cls.GESTURES
    
    @classmethod
    def get_gestures_by_category(cls, category: GestureCategory) -> List[GestureDefinition]:
        """按类别获取手势
        
        Args:
            category: 手势类别
            
        Returns:
            手势列表
        """
        return [g for g in cls.GESTURES.values() if g.category == category]
    
    @classmethod
    def get_gestures_by_difficulty(cls, difficulty: str) -> List[GestureDefinition]:
        """按难度获取手势
        
        Args:
            difficulty: 难度级别
            
        Returns:
            手势列表
        """
        return [g for g in cls.GESTURES.values() if g.difficulty == difficulty]
    
    @classmethod
    def get_gesture_names(cls) -> List[str]:
        """获取所有手势名称
        
        Returns:
            手势名称列表
        """
        return list(cls.GESTURES.keys())
    
    @classmethod
    def get_basic_gestures(cls) -> List[str]:
        """获取基础手势列表
        
        Returns:
            基础手势名称列表
        """
        return ['fist', 'open', 'rest', 'flex', 'extend']
    
    @classmethod
    def to_dict(cls) -> Dict:
        """转换为字典格式
        
        Returns:
            字典格式的手势库
        """
        return {
            name: {
                'name': g.name,
                'display_name': g.display_name,
                'category': g.category.value,
                'description': g.description,
                'emg_characteristics': g.emg_characteristics,
                'imu_characteristics': g.imu_characteristics,
                'difficulty': g.difficulty,
                'recommended_samples': g.recommended_samples,
                'tags': g.tags
            }
            for name, g in cls.GESTURES.items()
        }
    
    @classmethod
    def save_to_file(cls, filepath: str):
        """保存手势库到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cls.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"手势库已保存: {filepath}")
    
    @classmethod
    def get_training_recommendations(cls) -> Dict:
        """获取训练建议
        
        Returns:
            训练建议字典
        """
        return {
            'basic_set': {
                'gestures': cls.get_basic_gestures(),
                'description': '基础手势集，适合初学者',
                'total_samples': 450
            },
            'standard_set': {
                'gestures': ['fist', 'open', 'pinch', 'point', 'rest', 'flex', 'extend'],
                'description': '标准手势集，包含常用手势',
                'total_samples': 750
            },
            'full_set': {
                'gestures': cls.get_gesture_names(),
                'description': '完整手势集，包含所有预定义手势',
                'total_samples': sum(g.recommended_samples for g in cls.GESTURES.values())
            }
        }


class GestureTemplate:
    """手势特征模板
    
    存储手势的特征统计信息，用于快速匹配
    """
    
    def __init__(self, name: str):
        """初始化手势模板
        
        Args:
            name: 手势名称
        """
        self.name = name
        self.definition = PredefinedGestures.get_gesture(name)
        
        # 特征统计
        self.feature_mean = None
        self.feature_std = None
        self.feature_ranges = None
        
        # 样本数
        self.sample_count = 0
        
        logger.info(f"手势模板初始化: {name}")
    
    def update_from_features(self, features: np.ndarray):
        """从特征更新模板
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
        """
        self.sample_count = len(features)
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0)
        self.feature_ranges = {
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        }
        
        logger.info(f"模板更新: {self.name}, 样本数={self.sample_count}")
    
    def match_score(self, features: np.ndarray) -> float:
        """计算匹配分数
        
        Args:
            features: 特征向量
            
        Returns:
            匹配分数 (0-1)
        """
        if self.feature_mean is None:
            return 0.0
        
        # 计算马氏距离的简化版本
        diff = np.abs(features - self.feature_mean)
        normalized_diff = diff / (self.feature_std + 1e-10)
        distance = np.mean(normalized_diff)
        
        # 转换为分数
        score = np.exp(-distance)
        return float(score)
    
    def to_dict(self) -> Dict:
        """转换为字典
        
        Returns:
            模板字典
        """
        return {
            'name': self.name,
            'sample_count': self.sample_count,
            'feature_mean': self.feature_mean.tolist() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.tolist() if self.feature_std is not None else None,
            'feature_ranges': {
                k: v.tolist() for k, v in self.feature_ranges.items()
            } if self.feature_ranges else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GestureTemplate':
        """从字典创建模板
        
        Args:
            data: 模板字典
            
        Returns:
            GestureTemplate实例
        """
        template = cls(data['name'])
        template.sample_count = data['sample_count']
        template.feature_mean = np.array(data['feature_mean']) if data['feature_mean'] else None
        template.feature_std = np.array(data['feature_std']) if data['feature_std'] else None
        if data['feature_ranges']:
            template.feature_ranges = {
                k: np.array(v) for k, v in data['feature_ranges'].items()
            }
        return template


class GestureLibraryManager:
    """手势库管理器
    
    管理手势定义和模板
    """
    
    def __init__(self, library_path: str = 'data/gesture_library'):
        """初始化手势库管理器
        
        Args:
            library_path: 手势库路径
        """
        self.library_path = library_path
        self.templates: Dict[str, GestureTemplate] = {}
        
        os.makedirs(library_path, exist_ok=True)
        
        self._load_templates()
        
        logger.info(f"手势库管理器初始化: {library_path}")
    
    def _load_templates(self):
        """加载已保存的模板"""
        template_dir = os.path.join(self.library_path, 'templates')
        if not os.path.exists(template_dir):
            return
        
        for filename in os.listdir(template_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(template_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    template = GestureTemplate.from_dict(data)
                    self.templates[template.name] = template
                except Exception as e:
                    logger.error(f"加载模板失败 {filename}: {e}")
    
    def save_template(self, template: GestureTemplate):
        """保存模板
        
        Args:
            template: 手势模板
        """
        template_dir = os.path.join(self.library_path, 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        filepath = os.path.join(template_dir, f"{template.name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)
        
        self.templates[template.name] = template
        logger.info(f"模板已保存: {template.name}")
    
    def get_template(self, name: str) -> Optional[GestureTemplate]:
        """获取模板
        
        Args:
            name: 手势名称
            
        Returns:
            手势模板
        """
        return self.templates.get(name)
    
    def get_all_templates(self) -> Dict[str, GestureTemplate]:
        """获取所有模板
        
        Returns:
            模板字典
        """
        return self.templates
    
    def match_gesture(self, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """匹配手势
        
        Args:
            features: 特征向量
            top_k: 返回前k个匹配
            
        Returns:
            匹配结果列表 [(手势名, 分数), ...]
        """
        scores = []
        for name, template in self.templates.items():
            score = template.match_score(features)
            scores.append((name, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get_library_info(self) -> Dict:
        """获取手势库信息
        
        Returns:
            信息字典
        """
        return {
            'predefined_gestures': len(PredefinedGestures.GESTURES),
            'saved_templates': len(self.templates),
            'template_details': {
                name: {
                    'sample_count': t.sample_count,
                    'has_features': t.feature_mean is not None
                }
                for name, t in self.templates.items()
            }
        }


# 全局手势库管理器
_gesture_library = None

def get_gesture_library() -> GestureLibraryManager:
    """获取全局手势库管理器
    
    Returns:
        GestureLibraryManager实例
    """
    global _gesture_library
    if _gesture_library is None:
        _gesture_library = GestureLibraryManager()
    return _gesture_library


if __name__ == "__main__":
    # 测试预定义手势库
    print("=" * 60)
    print("预定义手势库测试")
    print("=" * 60)
    
    # 获取所有手势
    print("\n所有预定义手势:")
    for name, gesture in PredefinedGestures.get_all_gestures().items():
        print(f"  {name}: {gesture.display_name} ({gesture.category.value})")
    
    # 获取基础手势
    print("\n基础手势集:")
    print(f"  {PredefinedGestures.get_basic_gestures()}")
    
    # 获取训练建议
    print("\n训练建议:")
    recommendations = PredefinedGestures.get_training_recommendations()
    for set_name, info in recommendations.items():
        print(f"  {set_name}: {info['description']}")
        print(f"    手势数: {len(info['gestures'])}, 总样本: {info['total_samples']}")
    
    # 保存手势库
    PredefinedGestures.save_to_file('data/gesture_library/gestures.json')
    
    print("\n手势库测试完成!")
