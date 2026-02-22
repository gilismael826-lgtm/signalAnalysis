#!/usr/bin/env python3
"""
手势分类模块测试
验证手势分类器的训练、评估和预测功能
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gesture_classifier import GestureClassifier, GestureClassifierManager, get_gesture_classifier
from src.feature_extractor import FeatureExtractor


def test_svm_classifier():
    """测试SVM分类器"""
    print("=" * 60)
    print("测试SVM分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='svm')
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    # 为不同类别添加特征差异
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别: {np.unique(y)}")
    
    # 训练
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"训练集准确率: {result['train_accuracy']:.4f}")
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    print(f"F1分数: {result['f1_score']:.4f}")
    print()


def test_knn_classifier():
    """测试KNN分类器"""
    print("=" * 60)
    print("测试KNN分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='knn')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    print()


def test_rf_classifier():
    """测试随机森林分类器"""
    print("=" * 60)
    print("测试随机森林分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    
    # 获取特征重要性
    feature_names = [f'feature_{i}' for i in range(50)]
    importance = classifier.get_feature_importance(feature_names)
    if importance:
        print(f"Top 5重要特征: {[importance['sorted_importance'][i][0] for i in range(5)]}")
    print()


def test_mlp_classifier():
    """测试MLP分类器"""
    print("=" * 60)
    print("测试MLP分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='mlp')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    print()


def test_ensemble_classifier():
    """测试集成学习分类器"""
    print("=" * 60)
    print("测试集成学习分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='ensemble')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    print()


def test_gb_classifier():
    """测试梯度提升分类器"""
    print("=" * 60)
    print("测试梯度提升分类器")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='gb')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    print()


def test_cross_validation():
    """测试交叉验证训练"""
    print("=" * 60)
    print("测试交叉验证训练")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='svm')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    result = classifier.train_with_cv(X, y, n_splits=5)
    
    print(f"交叉验证分数: {result['cv_scores']}")
    print(f"平均准确率: {result['mean_accuracy']:.4f} (+/- {result['std_accuracy']*2:.4f})")
    print()


def test_grid_search():
    """测试网格搜索超参数调优"""
    print("=" * 60)
    print("测试网格搜索超参数调优")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='svm')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear']
    }
    
    result = classifier.grid_search(X, y, param_grid, cv=3)
    
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳分数: {result['best_score']:.4f}")
    print()


def test_predict_proba():
    """测试概率预测"""
    print("=" * 60)
    print("测试概率预测")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='svm')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    classifier.train(X, y, test_size=0.2)
    
    # 预测新样本
    X_new = np.random.randn(5, 50)
    X_new[0, :5] = 0
    X_new[1, :5] = 2
    X_new[2, :5] = 4
    X_new[3, :5] = 6
    X_new[4, :5] = 1
    
    predictions = classifier.predict(X_new)
    probas = classifier.predict_proba(X_new)
    
    print("预测结果:")
    for i, (pred, proba) in enumerate(zip(predictions, probas)):
        print(f"  样本{i}: {pred}, 置信度: {np.max(proba):.4f}")
    print()


def test_single_prediction():
    """测试单个样本预测"""
    print("=" * 60)
    print("测试单个样本预测")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    classifier.train(X, y, test_size=0.2)
    
    # 使用字典特征预测
    feature_dict = {f'feature_{i}': np.random.randn() for i in range(50)}
    feature_dict['feature_0'] = 2
    feature_dict['feature_1'] = 2
    
    result = classifier.predict_single(feature_dict)
    
    print(f"预测类别: {result['prediction']}")
    print(f"置信度: {result['confidence']:.4f}")
    print()


def test_save_load():
    """测试模型保存和加载"""
    print("=" * 60)
    print("测试模型保存和加载")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='svm')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    classifier.train(X, y, test_size=0.2)
    
    # 保存
    save_path = 'tests/test_classifier_model.pkl'
    classifier.save(save_path)
    
    # 加载
    classifier2 = GestureClassifier(algorithm='svm')
    classifier2.load(save_path)
    
    # 验证
    X_test = np.random.randn(10, 50)
    pred1 = classifier.predict(X_test)
    pred2 = classifier2.predict(X_test)
    
    print(f"原模型预测: {pred1[:3]}")
    print(f"加载模型预测: {pred2[:3]}")
    print(f"预测一致性: {np.array_equal(pred1, pred2)}")
    
    # 清理
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"已清理测试文件: {save_path}")
    print()


def test_model_evaluation():
    """测试模型评估"""
    print("=" * 60)
    print("测试模型评估")
    print("=" * 60)
    
    classifier = GestureClassifier(algorithm='rf')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    classifier.train(X, y, test_size=0.2)
    
    # 评估
    X_eval = np.random.randn(50, 50)
    y_eval = np.array(['gesture_0'] * 12 + ['gesture_1'] * 13 + ['gesture_2'] * 12 + ['gesture_3'] * 13)
    
    for i in range(4):
        if i == 0:
            X_eval[:12, :5] += i * 2
        elif i == 1:
            X_eval[12:25, :5] += i * 2
        elif i == 2:
            X_eval[25:37, :5] += i * 2
        else:
            X_eval[37:50, :5] += i * 2
    
    eval_result = classifier.evaluate(X_eval, y_eval)
    
    print(f"准确率: {eval_result['accuracy']:.4f}")
    print(f"精确率: {eval_result['precision']:.4f}")
    print(f"召回率: {eval_result['recall']:.4f}")
    print(f"F1分数: {eval_result['f1_score']:.4f}")
    print(f"混淆矩阵形状: {eval_result['confusion_matrix'].shape}")
    print()


def test_classifier_manager():
    """测试分类器管理器"""
    print("=" * 60)
    print("测试分类器管理器")
    print("=" * 60)
    
    manager = GestureClassifierManager(models_dir='tests/models')
    
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y = np.array(['gesture_0'] * 50 + ['gesture_1'] * 50 + ['gesture_2'] * 50 + ['gesture_3'] * 50)
    
    for i in range(4):
        X[i*50:(i+1)*50, :5] += i * 2
    
    # 训练所有分类器
    results = manager.train_all(X, y, algorithms=['svm', 'knn', 'rf'])
    
    # 比较结果
    comparison = manager.compare_results(results)
    print("\n分类器比较:")
    print(comparison.to_string(index=False))
    
    print(f"\n最佳分类器: {manager.best_algorithm}")
    
    # 保存最佳模型
    manager.save_best()
    print()


def test_with_real_features():
    """使用真实特征测试"""
    print("=" * 60)
    print("使用真实特征测试")
    print("=" * 60)
    
    # 创建特征提取器和分类器
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    classifier = GestureClassifier(algorithm='rf')
    
    # 创建模拟数据（多个手势样本）
    np.random.seed(42)
    n_samples_per_gesture = 30
    gestures = ['fist', 'open', 'pinch', 'wave']
    
    X_all = []
    y_all = []
    
    for gesture_idx, gesture_name in enumerate(gestures):
        for _ in range(n_samples_per_gesture):
            # 创建带手势特征的模拟数据
            emg_data = np.random.randn(8, 150) * 1000 + gesture_idx * 500
            imu_data = np.random.randn(6, 62) * 0.5 + gesture_idx * 0.2
            
            # 提取特征
            features = extractor.extract_all_features(emg_data, imu_data)
            X_all.append(list(features.values()))
            y_all.append(gesture_name)
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"类别: {np.unique(y)}")
    
    # 训练
    result = classifier.train(X, y, test_size=0.2)
    
    print(f"训练集准确率: {result['train_accuracy']:.4f}")
    print(f"测试集准确率: {result['test_accuracy']:.4f}")
    
    # 特征重要性
    feature_names = list(extractor.extract_all_features(
        np.random.randn(8, 150), np.random.randn(6, 62)
    ).keys())
    
    importance = classifier.get_feature_importance(feature_names)
    if importance:
        print("\nTop 10重要特征:")
        for i in range(10):
            name, score = importance['sorted_importance'][i]
            print(f"  {i+1}. {name}: {score:.4f}")
    print()


def test_global_instance():
    """测试全局实例"""
    print("=" * 60)
    print("测试全局分类器实例")
    print("=" * 60)
    
    classifier = get_gesture_classifier(algorithm='svm')
    print(f"全局实例算法: {classifier.algorithm}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("手势分类模块测试")
    print("=" * 60 + "\n")
    
    test_svm_classifier()
    test_knn_classifier()
    test_rf_classifier()
    test_mlp_classifier()
    test_ensemble_classifier()
    test_gb_classifier()
    test_cross_validation()
    test_grid_search()
    test_predict_proba()
    test_single_prediction()
    test_save_load()
    test_model_evaluation()
    test_classifier_manager()
    test_with_real_features()
    test_global_instance()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
