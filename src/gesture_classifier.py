#!/usr/bin/env python3
"""
手势分类模块
负责手势分类器的训练、评估和预测
支持SVM、KNN、随机森林、MLP和集成学习分类器
"""

import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger

try:
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn库未安装，手势分类功能将不可用")


class ModelVersionManager:
    """模型版本管理器
    
    管理模型的版本号、版本历史和版本比较
    """
    
    def __init__(self, models_dir='models'):
        """初始化版本管理器
        
        Args:
            models_dir: 模型存储目录
        """
        self.models_dir = models_dir
        self.version_file = os.path.join(models_dir, 'versions.json')
        self.versions = {}
        
        os.makedirs(models_dir, exist_ok=True)
        self._load_versions()
        
        logger.info(f"模型版本管理器初始化: {models_dir}")
    
    def _load_versions(self):
        """加载版本历史"""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    self.versions = json.load(f)
                logger.info(f"已加载 {len(self.versions)} 个模型版本记录")
            except Exception as e:
                logger.warning(f"加载版本文件失败: {e}")
                self.versions = {}
    
    def _save_versions(self):
        """保存版本历史"""
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(self.versions, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存版本文件失败: {e}")
    
    def generate_version(self, algorithm: str, accuracy: float) -> str:
        """生成新版本号
        
        Args:
            algorithm: 算法名称
            accuracy: 模型准确率
            
        Returns:
            版本号字符串
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f"v{timestamp}_{algorithm}_acc{accuracy:.4f}"
        return version
    
    def register_version(self, version: str, model_info: Dict):
        """注册新版本
        
        Args:
            version: 版本号
            model_info: 模型信息字典
        """
        self.versions[version] = {
            **model_info,
            'registered_at': datetime.now().isoformat()
        }
        self._save_versions()
        logger.info(f"注册模型版本: {version}")
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """获取版本信息
        
        Args:
            version: 版本号
            
        Returns:
            版本信息字典
        """
        return self.versions.get(version)
    
    def list_versions(self, algorithm: str = None, limit: int = 10) -> List[Dict]:
        """列出所有版本
        
        Args:
            algorithm: 筛选算法类型
            limit: 返回数量限制
            
        Returns:
            版本列表
        """
        versions_list = []
        
        for version, info in self.versions.items():
            if algorithm is None or algorithm in version:
                versions_list.append({
                    'version': version,
                    **info
                })
        
        versions_list.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        
        return versions_list[:limit]
    
    def get_best_version(self, algorithm: str = None) -> Optional[str]:
        """获取最佳版本
        
        Args:
            algorithm: 算法类型
            
        Returns:
            最佳版本号
        """
        versions = self.list_versions(algorithm=algorithm, limit=100)
        
        if not versions:
            return None
        
        best = max(versions, key=lambda x: x.get('test_accuracy', 0))
        return best.get('version')
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """比较两个版本
        
        Args:
            version1: 版本1
            version2: 版本2
            
        Returns:
            比较结果字典
        """
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            return {'error': '版本不存在'}
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'accuracy_diff': info1.get('test_accuracy', 0) - info2.get('test_accuracy', 0),
            'training_time_diff': info1.get('training_time', 0) - info2.get('training_time', 0),
            'algorithm1': info1.get('algorithm'),
            'algorithm2': info2.get('algorithm'),
            'better_version': version1 if info1.get('test_accuracy', 0) >= info2.get('test_accuracy', 0) else version2
        }
        
        return comparison
    
    def delete_version(self, version: str) -> bool:
        """删除版本记录
        
        Args:
            version: 版本号
            
        Returns:
            是否成功
        """
        if version in self.versions:
            del self.versions[version]
            self._save_versions()
            logger.info(f"删除版本记录: {version}")
            return True
        return False
    
    def get_version_statistics(self) -> Dict:
        """获取版本统计信息
        
        Returns:
            统计信息字典
        """
        if not self.versions:
            return {
                'total_versions': 0,
                'algorithms': {},
                'best_accuracy': None,
                'best_version': None
            }
        
        algorithms = {}
        best_accuracy = 0
        best_version = None
        
        for version, info in self.versions.items():
            algo = info.get('algorithm', 'unknown')
            algorithms[algo] = algorithms.get(algo, 0) + 1
            
            acc = info.get('test_accuracy', 0)
            if acc > best_accuracy:
                best_accuracy = acc
                best_version = version
        
        return {
            'total_versions': len(self.versions),
            'algorithms': algorithms,
            'best_accuracy': best_accuracy,
            'best_version': best_version
        }


class GestureClassifier:
    """手势分类器
    
    支持多种分类算法：SVM、KNN、随机森林、MLP、集成学习
    """
    
    ALGORITHMS = ['svm', 'knn', 'rf', 'mlp', 'ensemble', 'gb']
    
    def __init__(self, algorithm='svm'):
        """初始化分类器
        
        Args:
            algorithm: 分类算法 ('svm', 'knn', 'rf', 'mlp', 'ensemble', 'gb')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn库未安装，无法使用分类功能")
        
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"未知的算法: {algorithm}，支持的算法: {self.ALGORITHMS}")
        
        self.algorithm = algorithm
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.labels = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = {}
        self.version = None
        
        logger.info(f"手势分类器初始化: 算法={algorithm}")
    
    # ==================== 分类器创建 ====================
    
    def _create_classifier(self, params=None):
        """创建分类器实例
        
        Args:
            params: 分类器参数字典
            
        Returns:
            分类器实例
        """
        params = params or {}
        
        if self.algorithm == 'svm':
            default_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            }
            default_params.update(params)
            return SVC(**default_params)
        
        elif self.algorithm == 'knn':
            default_params = {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'minkowski',
                'p': 2
            }
            default_params.update(params)
            return KNeighborsClassifier(**default_params)
        
        elif self.algorithm == 'rf':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
        elif self.algorithm == 'mlp':
            default_params = {
                'hidden_layer_sizes': (256, 128, 64),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate': 'adaptive',
                'max_iter': 500,
                'random_state': 42,
                'early_stopping': False,  # 禁用早停以支持小样本
                'validation_fraction': 0.1
            }
            default_params.update(params)
            return MLPClassifier(**default_params)
        
        elif self.algorithm == 'ensemble':
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            return VotingClassifier(
                estimators=[('svm', svm), ('knn', knn), ('rf', rf)],
                voting='soft',
                n_jobs=-1
            )
        
        elif self.algorithm == 'gb':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            default_params.update(params)
            return GradientBoostingClassifier(**default_params)
        
        else:
            raise ValueError(f"未知的算法: {self.algorithm}")
    
    # ==================== 数据预处理 ====================
    
    def _preprocess_data(self, X, y=None, fit=True):
        """数据预处理
        
        Args:
            X: 特征矩阵
            y: 标签向量
            fit: 是否拟合预处理器
            
        Returns:
            处理后的X和y
        """
        # 标准化
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("预处理器未拟合")
            X_scaled = self.scaler.transform(X)
        
        # 标签编码
        if y is not None:
            if fit:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                self.labels = self.label_encoder.classes_
            else:
                if self.label_encoder is None:
                    raise ValueError("标签编码器未拟合")
                y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = None
        
        return X_scaled, y_encoded
    
    # ==================== 模型训练 ====================
    
    def train(self, X, y, test_size=0.2, params=None, preprocess=True, random_state=42):
        """训练分类器
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量
            test_size: 测试集比例
            params: 分类器参数
            preprocess: 是否进行数据预处理
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        start_time = time.time()
        
        logger.info(f"开始训练 {self.algorithm} 分类器...")
        logger.info(f"数据形状: X={X.shape}, y={y.shape}")
        logger.info(f"类别数: {len(np.unique(y))}")
        
        # 数据预处理
        if preprocess:
            X_processed, y_processed = self._preprocess_data(X, y, fit=True)
        else:
            X_processed, y_processed = X, y
            self.labels = np.unique(y)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=test_size,
            random_state=random_state,
            stratify=y_processed
        )
        
        logger.info(f"训练集: {X_train.shape[0]} 样本")
        logger.info(f"测试集: {X_test.shape[0]} 样本")
        
        # 创建并训练分类器
        self.classifier = self._create_classifier(params)
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # 评估
        train_pred = self.classifier.predict(X_train)
        test_pred = self.classifier.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # 计算更多评估指标
        precision = precision_score(y_test, test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, test_pred)
        
        # 训练历史
        training_time = time.time() - start_time
        self.training_history = {
            'algorithm': self.algorithm,
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'n_features': X_train.shape[1],
            'n_classes': len(self.labels),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"训练完成! 耗时: {training_time:.2f}s")
        logger.info(f"训练集准确率: {train_acc:.4f}")
        logger.info(f"测试集准确率: {test_acc:.4f}")
        logger.info(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'training_time': training_time,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': test_pred
        }
    
    def train_with_cv(self, X, y, n_splits=5, params=None, preprocess=True):
        """使用交叉验证训练
        
        Args:
            X: 特征矩阵
            y: 标签向量
            n_splits: 交叉验证折数
            params: 分类器参数
            preprocess: 是否进行数据预处理
            
        Returns:
            交叉验证结果
        """
        start_time = time.time()
        
        logger.info(f"开始{n_splits}折交叉验证训练...")
        
        # 数据预处理
        if preprocess:
            X_processed, y_processed = self._preprocess_data(X, y, fit=True)
        else:
            X_processed, y_processed = X, y
            self.labels = np.unique(y)
        
        # 创建分类器
        self.classifier = self._create_classifier(params)
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_scores = cross_val_score(
            self.classifier, X_processed, y_processed,
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        # 在全部数据上训练最终模型
        self.classifier.fit(X_processed, y_processed)
        self.is_trained = True
        
        training_time = time.time() - start_time
        
        logger.info(f"交叉验证准确率: {cv_scores}")
        logger.info(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.training_history = {
            'algorithm': self.algorithm,
            'n_features': X_processed.shape[1],
            'n_classes': len(self.labels),
            'cv_scores': cv_scores.tolist(),
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'training_time': training_time
        }
    
    def grid_search(self, X, y, param_grid, cv=5, preprocess=True):
        """网格搜索超参数调优
        
        Args:
            X: 特征矩阵
            y: 标签向量
            param_grid: 参数网格
            cv: 交叉验证折数
            preprocess: 是否进行数据预处理
            
        Returns:
            最佳参数和结果
        """
        logger.info(f"开始网格搜索超参数调优...")
        logger.info(f"参数网格: {param_grid}")
        
        # 数据预处理
        if preprocess:
            X_processed, y_processed = self._preprocess_data(X, y, fit=True)
        else:
            X_processed, y_processed = X, y
            self.labels = np.unique(y)
        
        # 创建基础分类器
        base_classifier = self._create_classifier()
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_classifier,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_processed, y_processed)
        
        # 使用最佳参数
        self.classifier = grid_search.best_estimator_
        self.is_trained = True
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳分数: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    # ==================== 模型预测 ====================
    
    def predict(self, X, preprocess=True):
        """预测手势类别
        
        Args:
            X: 特征矩阵
            preprocess: 是否进行数据预处理
            
        Returns:
            预测的类别标签
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 数据预处理
        if preprocess:
            X_processed, _ = self._preprocess_data(X, fit=False)
        else:
            X_processed = X
        
        # 预测
        y_pred_encoded = self.classifier.predict(X_processed)
        
        # 解码标签
        if self.label_encoder is not None:
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = y_pred_encoded
        
        return y_pred
    
    def predict_proba(self, X, preprocess=True):
        """预测类别概率
        
        Args:
            X: 特征矩阵
            preprocess: 是否进行数据预处理
            
        Returns:
            类别概率矩阵
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 数据预处理
        if preprocess:
            X_processed, _ = self._preprocess_data(X, fit=False)
        else:
            X_processed = X
        
        # 预测概率
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X_processed)
        else:
            raise ValueError(f"{self.algorithm} 分类器不支持概率预测")
        
        return proba
    
    def predict_single(self, features, preprocess=True):
        """预测单个样本
        
        Args:
            features: 特征向量或字典
            preprocess: 是否进行数据预处理
            
        Returns:
            预测结果字典
        """
        if isinstance(features, dict):
            if self.feature_names is None:
                self.feature_names = list(features.keys())
                feature_vector = np.array([list(features.values())])
            else:
                # 按照训练时的特征顺序排列
                feature_values = []
                for name in self.feature_names:
                    if name in features:
                        feature_values.append(features[name])
                    else:
                        logger.warning(f"特征 '{name}' 在输入中缺失，使用0填充")
                        feature_values.append(0.0)
                feature_vector = np.array([feature_values])
        else:
            feature_vector = np.array([features])
        
        prediction = self.predict(feature_vector, preprocess=preprocess)[0]
        
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.predict_proba(feature_vector, preprocess=preprocess)[0]
            confidence = np.max(proba)
        else:
            confidence = 1.0
            proba = None
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': proba.tolist() if proba is not None else None,
            'labels': self.labels.tolist() if self.labels is not None else None
        }
    
    # ==================== 模型评估 ====================
    
    def evaluate(self, X, y, preprocess=True):
        """评估模型性能
        
        Args:
            X: 特征矩阵
            y: 真实标签
            preprocess: 是否进行数据预处理
            
        Returns:
            评估结果字典
        """
        y_pred = self.predict(X, preprocess=preprocess)
        
        # 计算各种指标
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        
        # 分类报告
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def get_feature_importance(self, feature_names=None):
        """获取特征重要性（仅适用于支持的特征分类器）
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        feature_names = feature_names or self.feature_names
        
        if self.algorithm == 'rf':
            importance = self.classifier.feature_importances_
        elif self.algorithm == 'gb':
            importance = self.classifier.feature_importances_
        else:
            logger.warning(f"{self.algorithm} 分类器不支持特征重要性")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        indices = np.argsort(importance)[::-1]
        
        return {
            'feature_names': feature_names,
            'importance': importance,
            'ranking': indices,
            'sorted_importance': [(feature_names[i], importance[i]) for i in indices]
        }
    
    # ==================== 模型管理 ====================
    
    def save(self, filepath, version_manager: ModelVersionManager = None):
        """保存模型
        
        Args:
            filepath: 保存路径
            version_manager: 版本管理器实例（可选）
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        model_data = {
            'algorithm': self.algorithm,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'labels': self.labels,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'version': self.version
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        joblib.dump(model_data, filepath)
        
        if version_manager is not None:
            accuracy = self.training_history.get('test_accuracy', 0)
            version = version_manager.generate_version(self.algorithm, accuracy)
            self.version = version
            
            version_manager.register_version(version, {
                'algorithm': self.algorithm,
                'test_accuracy': accuracy,
                'train_accuracy': self.training_history.get('train_accuracy', 0),
                'precision': self.training_history.get('precision', 0),
                'recall': self.training_history.get('recall', 0),
                'f1_score': self.training_history.get('f1_score', 0),
                'training_time': self.training_history.get('training_time', 0),
                'n_classes': len(self.labels) if self.labels is not None else 0,
                'filepath': filepath
            })
            
            model_data['version'] = version
            joblib.dump(model_data, filepath)
        
        logger.info(f"模型已保存: {filepath}" + (f", 版本: {self.version}" if self.version else ""))
    
    def save_with_version(self, models_dir='models', version_manager: ModelVersionManager = None) -> str:
        """保存模型并自动生成版本化文件名
        
        Args:
            models_dir: 模型目录
            version_manager: 版本管理器实例
            
        Returns:
            保存的文件路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        if version_manager is None:
            version_manager = ModelVersionManager(models_dir)
        
        accuracy = self.training_history.get('test_accuracy', 0)
        version = version_manager.generate_version(self.algorithm, accuracy)
        self.version = version
        
        filename = f"{version}.pkl"
        filepath = os.path.join(models_dir, filename)
        
        self.save(filepath, version_manager)
        
        return filepath
    
    def load(self, filepath):
        """加载模型
        
        Args:
            filepath: 模型文件路径
        """
        model_data = joblib.load(filepath)
        
        self.algorithm = model_data['algorithm']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.labels = model_data['labels']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        self.version = model_data.get('version')
        self.is_trained = True
        
        logger.info(f"模型已加载: {filepath}")
        logger.info(f"算法: {self.algorithm}, 类别数: {len(self.labels) if self.labels is not None else 'N/A'}" + 
                   (f", 版本: {self.version}" if self.version else ""))
    
    def get_model_info(self):
        """获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'algorithm': self.algorithm,
            'is_trained': self.is_trained,
            'n_classes': len(self.labels) if self.labels is not None else None,
            'labels': self.labels.tolist() if self.labels is not None else None,
            'n_features': len(self.feature_names) if self.feature_names is not None else None,
            'training_history': self.training_history,
            'version': self.version
        }


class GestureClassifierManager:
    """手势分类器管理器
    
    管理多个分类器，支持自动训练所有模型、智能选择最优模型和集成预测
    """
    
    def __init__(self, models_dir='models'):
        """初始化管理器
        
        Args:
            models_dir: 模型保存目录
        """
        self.models_dir = models_dir
        self.classifiers = {}
        self.best_classifier = None
        self.best_algorithm = None
        self.version_manager = ModelVersionManager(models_dir)
        
        self.model_weights = {}
        self.ensemble_enabled = True
        self.min_confidence_for_ensemble = 0.6
        self.model_performance = {}
        
        os.makedirs(models_dir, exist_ok=True)
    
    def train_all(self, X, y, algorithms=None, test_size=0.2, auto_select_best=True):
        """自动训练所有分类器并选择最优模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
            algorithms: 要训练的算法列表，默认训练所有支持的算法
            test_size: 测试集比例
            auto_select_best: 是否自动选择最优模型
            
        Returns:
            所有分类器的训练结果
        """
        algorithms = algorithms or GestureClassifier.ALGORITHMS
        results = {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始自动训练所有模型，共 {len(algorithms)} 个算法")
        logger.info(f"{'='*60}")
        
        for algorithm in algorithms:
            logger.info(f"\n{'='*50}")
            logger.info(f"训练 {algorithm.upper()} 分类器...")
            logger.info(f"{'='*50}")
            
            try:
                classifier = GestureClassifier(algorithm=algorithm)
                result = classifier.train(X, y, test_size=test_size)
                
                self.classifiers[algorithm] = classifier
                results[algorithm] = result
                
                self.model_performance[algorithm] = {
                    'test_accuracy': result.get('test_accuracy', 0),
                    'f1_score': result.get('f1_score', 0),
                    'precision': result.get('precision', 0),
                    'recall': result.get('recall', 0),
                    'training_time': result.get('training_time', 0)
                }
                
            except Exception as e:
                logger.error(f"训练 {algorithm} 失败: {e}")
                results[algorithm] = {'error': str(e)}
        
        if auto_select_best:
            self._select_best_model(results)
            self._calculate_model_weights()
        
        self._log_training_summary(results)
        
        return results
    
    def _select_best_model(self, results):
        """选择最优模型
        
        Args:
            results: 训练结果字典
        """
        valid_results = {k: v for k, v in results.items() if 'test_accuracy' in v}
        
        if not valid_results:
            logger.warning("没有成功训练的模型")
            return
        
        best_score = -1
        for algorithm, result in valid_results.items():
            score = self._calculate_model_score(result)
            if score > best_score:
                best_score = score
                self.best_algorithm = algorithm
                self.best_classifier = self.classifiers[algorithm]
        
        if self.best_algorithm:
            logger.info(f"\n{'='*60}")
            logger.info(f"最优模型: {self.best_algorithm.upper()}")
            logger.info(f"综合得分: {best_score:.4f}")
            logger.info(f"测试准确率: {valid_results[self.best_algorithm]['test_accuracy']:.4f}")
            logger.info(f"F1分数: {valid_results[self.best_algorithm]['f1_score']:.4f}")
            logger.info(f"{'='*60}")
    
    def _calculate_model_score(self, result):
        """计算模型综合得分
        
        综合考虑准确率、F1分数和训练时间
        
        Args:
            result: 训练结果
            
        Returns:
            综合得分
        """
        accuracy = result.get('test_accuracy', 0)
        f1 = result.get('f1_score', 0)
        precision = result.get('precision', 0)
        recall = result.get('recall', 0)
        
        score = accuracy * 0.4 + f1 * 0.3 + precision * 0.15 + recall * 0.15
        
        return score
    
    def _calculate_model_weights(self):
        """计算集成预测时的模型权重
        
        基于各模型的测试准确率计算权重
        """
        if not self.model_performance:
            return
        
        total_score = 0
        scores = {}
        
        for algorithm, perf in self.model_performance.items():
            score = self._calculate_model_score(perf)
            scores[algorithm] = score
            total_score += score
        
        if total_score > 0:
            for algorithm, score in scores.items():
                self.model_weights[algorithm] = score / total_score
        else:
            equal_weight = 1.0 / len(self.model_performance)
            for algorithm in self.model_performance:
                self.model_weights[algorithm] = equal_weight
        
        logger.info(f"\n模型权重分配:")
        for algo, weight in sorted(self.model_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {algo}: {weight:.4f}")
    
    def _log_training_summary(self, results):
        """输出训练摘要
        
        Args:
            results: 训练结果
        """
        logger.info(f"\n{'='*60}")
        logger.info("训练摘要")
        logger.info(f"{'='*60}")
        
        summary_data = []
        for algorithm, result in results.items():
            if 'error' not in result:
                summary_data.append({
                    'algorithm': algorithm,
                    'train_acc': result.get('train_accuracy', 0),
                    'test_acc': result.get('test_accuracy', 0),
                    'f1': result.get('f1_score', 0),
                    'time': result.get('training_time', 0)
                })
        
        summary_data.sort(key=lambda x: x['test_acc'], reverse=True)
        
        logger.info(f"{'算法':<12} {'训练准确率':<12} {'测试准确率':<12} {'F1分数':<12} {'训练时间(s)':<12}")
        logger.info("-" * 60)
        for data in summary_data:
            logger.info(f"{data['algorithm']:<12} {data['train_acc']:<12.4f} {data['test_acc']:<12.4f} {data['f1']:<12.4f} {data['time']:<12.2f}")
    
    def predict_ensemble(self, X, preprocess=True, strategy='weighted_voting'):
        """集成预测：综合多个模型的预测结果
        
        Args:
            X: 特征矩阵
            preprocess: 是否进行数据预处理
            strategy: 集成策略
                - 'weighted_voting': 加权投票（默认）
                - 'best_only': 仅使用最优模型
                - 'stacking': 堆叠策略（需要置信度）
                - 'adaptive': 自适应策略（根据置信度动态选择）
            
        Returns:
            预测结果字典
        """
        if not self.classifiers:
            raise ValueError("没有可用的分类器，请先训练模型")
        
        if strategy == 'best_only' or not self.ensemble_enabled:
            return self._predict_best_only(X, preprocess)
        
        all_predictions = {}
        all_probabilities = {}
        
        for algorithm, classifier in self.classifiers.items():
            if not classifier.is_trained:
                continue
            
            try:
                pred = classifier.predict(X, preprocess=preprocess)
                proba = classifier.predict_proba(X, preprocess=preprocess)
                
                all_predictions[algorithm] = pred
                all_probabilities[algorithm] = proba
            except Exception as e:
                logger.warning(f"模型 {algorithm} 预测失败: {e}")
        
        if not all_predictions:
            raise ValueError("所有模型预测失败")
        
        if strategy == 'weighted_voting':
            return self._weighted_voting_predict(all_predictions, all_probabilities)
        elif strategy == 'adaptive':
            return self._adaptive_predict(all_predictions, all_probabilities)
        else:
            return self._weighted_voting_predict(all_predictions, all_probabilities)
    
    def _predict_best_only(self, X, preprocess):
        """仅使用最优模型预测
        
        Args:
            X: 特征矩阵
            preprocess: 是否预处理
            
        Returns:
            预测结果
        """
        if self.best_classifier is None:
            raise ValueError("最优模型未确定")
        
        predictions = self.best_classifier.predict(X, preprocess=preprocess)
        probabilities = self.best_classifier.predict_proba(X, preprocess=preprocess)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'method': 'best_only',
            'best_algorithm': self.best_algorithm
        }
    
    def _weighted_voting_predict(self, all_predictions, all_probabilities):
        """加权投票预测
        
        Args:
            all_predictions: 所有模型的预测结果
            all_probabilities: 所有模型的概率预测
            
        Returns:
            预测结果
        """
        n_samples = len(next(iter(all_predictions.values())))
        final_predictions = []
        final_probabilities = []
        
        labels = self.best_classifier.labels if self.best_classifier else None
        
        for i in range(n_samples):
            sample_probs = {}
            
            for algorithm, proba in all_probabilities.items():
                weight = self.model_weights.get(algorithm, 1.0 / len(all_probabilities))
                
                for j, prob in enumerate(proba[i]):
                    label_idx = j
                    if label_idx not in sample_probs:
                        sample_probs[label_idx] = 0
                    sample_probs[label_idx] += prob * weight
            
            best_label_idx = max(sample_probs.keys(), key=lambda k: sample_probs[k])
            final_predictions.append(best_label_idx)
            final_probabilities.append(sample_probs)
        
        if labels is not None:
            final_predictions = [labels[p] for p in final_predictions]
        
        return {
            'predictions': np.array(final_predictions),
            'probabilities': final_probabilities,
            'method': 'weighted_voting',
            'model_weights': self.model_weights,
            'participating_models': list(all_predictions.keys())
        }
    
    def _adaptive_predict(self, all_predictions, all_probabilities):
        """自适应预测：根据置信度动态选择模型或集成
        
        Args:
            all_predictions: 所有模型的预测结果
            all_probabilities: 所有模型的概率预测
            
        Returns:
            预测结果
        """
        n_samples = len(next(iter(all_predictions.values())))
        final_predictions = []
        decision_info = []
        
        labels = self.best_classifier.labels if self.best_classifier else None
        
        for i in range(n_samples):
            best_confidence = 0
            best_pred = None
            best_algo = None
            
            for algorithm, proba in all_probabilities.items():
                max_prob = max(proba[i])
                if max_prob > best_confidence:
                    best_confidence = max_prob
                    best_pred = np.argmax(proba[i])
                    best_algo = algorithm
            
            if best_confidence >= self.min_confidence_for_ensemble:
                if labels is not None:
                    final_predictions.append(labels[best_pred])
                else:
                    final_predictions.append(best_pred)
                decision_info.append({
                    'method': 'single_model',
                    'algorithm': best_algo,
                    'confidence': best_confidence
                })
            else:
                ensemble_result = self._weighted_voting_predict(
                    {k: np.array([v[i]]) for k, v in all_predictions.items()},
                    {k: np.array([v[i]]) for k, v in all_probabilities.items()}
                )
                final_predictions.append(ensemble_result['predictions'][0])
                decision_info.append({
                    'method': 'ensemble',
                    'confidence': best_confidence
                })
        
        return {
            'predictions': np.array(final_predictions),
            'method': 'adaptive',
            'decision_info': decision_info,
            'participating_models': list(all_predictions.keys())
        }
    
    def predict_single_ensemble(self, features, preprocess=True, strategy='weighted_voting'):
        """集成预测单个样本
        
        Args:
            features: 特征向量或字典
            preprocess: 是否进行数据预处理
            strategy: 集成策略
            
        Returns:
            预测结果字典
        """
        if isinstance(features, dict):
            feature_vector = np.array([list(features.values())])
        else:
            feature_vector = np.array([features])
        
        result = self.predict_ensemble(feature_vector, preprocess=preprocess, strategy=strategy)
        
        prediction = result['predictions'][0]
        
        if self.best_classifier and hasattr(self.best_classifier, 'labels') and self.best_classifier.labels is not None:
            labels = self.best_classifier.labels.tolist()
            if result['method'] == 'weighted_voting':
                prob_dict = result['probabilities'][0]
                proba = [prob_dict.get(i, 0) for i in range(len(labels))]
                confidence = max(proba)
            else:
                proba = result['probabilities'][0] if 'probabilities' in result else None
                confidence = max(proba) if proba is not None else 1.0
        else:
            labels = None
            proba = None
            confidence = 1.0
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': proba,
            'labels': labels,
            'method': result['method'],
            'participating_models': result.get('participating_models', [])
        }
    
    def compare_results(self, results):
        """比较所有分类器的结果
        
        Args:
            results: 训练结果字典
            
        Returns:
            比较结果DataFrame
        """
        import pandas as pd
        
        comparison_data = []
        
        for algorithm, result in results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Algorithm': algorithm,
                    'Train Accuracy': result.get('train_accuracy', 0),
                    'Test Accuracy': result.get('test_accuracy', 0),
                    'Precision': result.get('precision', 0),
                    'Recall': result.get('recall', 0),
                    'F1 Score': result.get('f1_score', 0),
                    'Training Time (s)': result.get('training_time', 0),
                    'Weight': self.model_weights.get(algorithm, 0)
                })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test Accuracy', ascending=False)
        
        return df
    
    def save_best(self, filepath=None, with_version=True):
        """保存最佳分类器
        
        Args:
            filepath: 保存路径
            with_version: 是否使用版本管理
        """
        if self.best_classifier is None:
            raise ValueError("没有可用的最佳分类器")
        
        if filepath is None:
            if with_version:
                filepath = self.best_classifier.save_with_version(
                    self.models_dir, 
                    self.version_manager
                )
                return filepath
            else:
                filepath = os.path.join(self.models_dir, f'best_classifier_{self.best_algorithm}.pkl')
        
        self.best_classifier.save(filepath, self.version_manager if with_version else None)
        return filepath
    
    def save_all(self, with_version=True):
        """保存所有训练好的模型
        
        Args:
            with_version: 是否使用版本管理
            
        Returns:
            保存的文件路径列表
        """
        saved_paths = []
        
        for algorithm, classifier in self.classifiers.items():
            if classifier.is_trained:
                if with_version:
                    path = classifier.save_with_version(self.models_dir, self.version_manager)
                else:
                    path = os.path.join(self.models_dir, f'classifier_{algorithm}.pkl')
                    classifier.save(path)
                saved_paths.append(path)
        
        self._save_ensemble_config()
        
        logger.info(f"已保存 {len(saved_paths)} 个模型")
        return saved_paths
    
    def _save_ensemble_config(self):
        """保存集成配置"""
        config_path = os.path.join(self.models_dir, 'ensemble_config.json')
        config_data = {
            'best_algorithm': self.best_algorithm,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'ensemble_enabled': self.ensemble_enabled,
            'min_confidence_for_ensemble': self.min_confidence_for_ensemble,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"集成配置已保存: {config_path}")
    
    def load_ensemble_config(self):
        """加载集成配置"""
        config_path = os.path.join(self.models_dir, 'ensemble_config.json')
        
        if not os.path.exists(config_path):
            logger.warning("集成配置文件不存在")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        self.best_algorithm = config_data.get('best_algorithm')
        self.model_weights = config_data.get('model_weights', {})
        self.model_performance = config_data.get('model_performance', {})
        self.ensemble_enabled = config_data.get('ensemble_enabled', True)
        self.min_confidence_for_ensemble = config_data.get('min_confidence_for_ensemble', 0.6)
        
        logger.info(f"集成配置已加载，最优模型: {self.best_algorithm}")
        return True
    
    def load_classifier(self, filepath, name=None):
        """加载分类器
        
        Args:
            filepath: 模型文件路径
            name: 分类器名称
        """
        classifier = GestureClassifier(algorithm='svm')
        classifier.load(filepath)
        
        name = name or classifier.algorithm
        self.classifiers[name] = classifier
        
        return classifier
    
    def load_all_classifiers(self):
        """加载所有保存的模型
        
        Returns:
            加载的模型数量
        """
        if not os.path.exists(self.models_dir):
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return 0
        
        loaded_count = 0
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.models_dir, filename)
                try:
                    classifier = GestureClassifier(algorithm='svm')
                    classifier.load(filepath)
                    self.classifiers[classifier.algorithm] = classifier
                    loaded_count += 1
                    logger.info(f"已加载模型: {filename} ({classifier.algorithm})")
                except Exception as e:
                    logger.warning(f"加载模型 {filename} 失败: {e}")
        
        self.load_ensemble_config()
        
        if self.best_algorithm and self.best_algorithm in self.classifiers:
            self.best_classifier = self.classifiers[self.best_algorithm]
        
        logger.info(f"共加载 {loaded_count} 个模型")
        return loaded_count
    
    def predict(self, X, use_ensemble=True, strategy='weighted_voting'):
        """使用分类器预测（自动选择最优策略）
        
        Args:
            X: 特征矩阵
            use_ensemble: 是否使用集成预测
            strategy: 集成策略
            
        Returns:
            预测结果
        """
        if use_ensemble and len(self.classifiers) > 1:
            result = self.predict_ensemble(X, strategy=strategy)
            return result['predictions']
        elif self.best_classifier:
            return self.best_classifier.predict(X)
        else:
            raise ValueError("没有可用的分类器")
    
    def set_ensemble_mode(self, enabled: bool, min_confidence: float = None):
        """设置集成模式
        
        Args:
            enabled: 是否启用集成预测
            min_confidence: 自适应模式的最小置信度阈值
        """
        self.ensemble_enabled = enabled
        if min_confidence is not None:
            self.min_confidence_for_ensemble = min_confidence
        logger.info(f"集成模式: {'启用' if enabled else '禁用'}")
    
    def get_model_info(self):
        """获取所有模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'total_models': len(self.classifiers),
            'trained_models': sum(1 for c in self.classifiers.values() if c.is_trained),
            'best_algorithm': self.best_algorithm,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'ensemble_enabled': self.ensemble_enabled,
            'available_algorithms': list(self.classifiers.keys())
        }


# 全局分类器实例
_gesture_classifier = None

def get_gesture_classifier(algorithm='svm'):
    """获取全局手势分类器实例
    
    Args:
        algorithm: 分类算法
        
    Returns:
        GestureClassifier实例
    """
    global _gesture_classifier
    if _gesture_classifier is None:
        _gesture_classifier = GestureClassifier(algorithm=algorithm)
    return _gesture_classifier
