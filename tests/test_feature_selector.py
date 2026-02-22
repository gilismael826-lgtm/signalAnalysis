#!/usr/bin/env python3
"""
特征选择与降维模块测试
验证特征选择和降维功能的正确性
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_selector import FeatureSelector, get_feature_selector
from src.feature_extractor import FeatureExtractor


def test_pca():
    """测试PCA降维"""
    print("=" * 60)
    print("测试PCA降维")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 50)  # 100样本，50特征
    
    print(f"原始数据形状: {X.shape}")
    
    # 先标准化
    X_scaled = selector.fit_scaler(X)
    
    # 应用PCA
    X_pca = selector.apply_pca(X_scaled)
    
    print(f"PCA降维后形状: {X_pca.shape}")
    
    # 获取PCA信息
    pca_info = selector.get_pca_info()
    print(f"累计方差解释比例: {pca_info['cumulative_variance_ratio'][-1]:.4f}")
    
    # 测试转换新数据
    X_new = np.random.randn(10, 50)
    X_new_scaled = selector.transform_scaler(X_new)
    X_new_pca = selector.transform_pca(X_new_scaled)
    print(f"新数据PCA转换后形状: {X_new_pca.shape}")
    print()


def test_lda():
    """测试LDA降维"""
    print("=" * 60)
    print("测试LDA降维")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=5)
    
    # 创建模拟数据（3类）
    np.random.seed(42)
    X = np.random.randn(150, 50)
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)
    
    print(f"原始数据形状: {X.shape}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 标准化
    X_scaled = selector.fit_scaler(X)
    
    # 应用LDA
    X_lda = selector.apply_lda(X_scaled, y)
    
    print(f"LDA降维后形状: {X_lda.shape}")
    
    # 获取LDA信息
    lda_info = selector.get_lda_info()
    if lda_info['explained_variance_ratio'] is not None:
        print(f"LDA判别成分数: {lda_info['n_components']}")
    print()


def test_feature_importance():
    """测试特征重要性评估"""
    print("=" * 60)
    print("测试特征重要性评估")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 30)
    y = np.array([0] * 50 + [1] * 50)
    feature_names = [f'feature_{i}' for i in range(30)]
    
    # 随机森林特征重要性
    print("\n随机森林特征重要性:")
    rf_importance = selector.get_feature_importance_rf(X, y, feature_names)
    print(f"Top 5特征: {[feature_names[i] for i in rf_importance['ranking'][:5]]}")
    
    # 互信息特征重要性
    print("\n互信息特征重要性:")
    mi_importance = selector.get_feature_importance_mi(X, y, feature_names)
    print(f"Top 5特征: {[feature_names[i] for i in mi_importance['ranking'][:5]]}")
    
    # ANOVA F值特征重要性
    print("\nANOVA F值特征重要性:")
    anova_importance = selector.get_feature_importance_anova(X, y, feature_names)
    print(f"Top 5特征: {[feature_names[i] for i in anova_importance['ranking'][:5]]}")
    print()


def test_select_k_best():
    """测试选择K个最佳特征"""
    print("=" * 60)
    print("测试选择K个最佳特征")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 30)
    y = np.array([0] * 50 + [1] * 50)
    feature_names = [f'feature_{i}' for i in range(30)]
    
    print(f"原始特征数: {X.shape[1]}")
    
    # 选择K个最佳特征
    X_selected, selected_indices = selector.select_k_best_features(
        X, y, k=10, feature_names=feature_names
    )
    
    print(f"选择后特征数: {X_selected.shape[1]}")
    print(f"选择的特征索引: {selected_indices}")
    print()


def test_all_feature_importance():
    """测试综合特征重要性"""
    print("=" * 60)
    print("测试综合特征重要性评估")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.array([0] * 50 + [1] * 50)
    feature_names = [f'feature_{i}' for i in range(20)]
    
    all_importance = selector.get_all_feature_importance(X, y, feature_names)
    
    print("综合排名Top 10特征:")
    combined_ranking = all_importance['combined']['ranking']
    for i, idx in enumerate(combined_ranking[:10]):
        print(f"  {i+1}. {feature_names[idx]}: "
              f"RF={all_importance['random_forest']['importance'][idx]:.4f}, "
              f"MI={all_importance['mutual_information']['importance'][idx]:.4f}")
    print()


def test_fit_transform():
    """测试综合拟合并转换"""
    print("=" * 60)
    print("测试综合拟合并转换")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.array([0] * 50 + [1] * 50)
    
    print(f"原始数据形状: {X.shape}")
    
    # PCA
    X_pca = selector.fit_transform(X, method='pca', n_components=10)
    print(f"PCA降维后形状: {X_pca.shape}")
    
    # 重置选择器
    selector2 = FeatureSelector(n_components=5)
    
    # LDA
    X_lda = selector2.fit_transform(X, y=y, method='lda', n_components=5)
    print(f"LDA降维后形状: {X_lda.shape}")
    print()


def test_with_real_features():
    """使用真实特征提取结果测试"""
    print("=" * 60)
    print("使用真实特征提取结果测试")
    print("=" * 60)
    
    # 创建特征提取器和选择器
    extractor = FeatureExtractor(emg_sample_rate=250.0, imu_sample_rate=104.0)
    selector = FeatureSelector(n_components=20)
    
    # 创建模拟EMG和IMU数据
    np.random.seed(42)
    emg_data = np.random.randn(8, 150) * 1000
    imu_data = np.random.randn(6, 62) * 0.5
    
    # 提取特征
    features = extractor.extract_all_features(emg_data, imu_data)
    feature_names = list(features.keys())
    X = np.array([list(features.values())])
    
    # 创建多个样本（模拟多个手势样本）
    n_samples = 50
    X_all = np.zeros((n_samples, len(feature_names)))
    for i in range(n_samples):
        emg_data = np.random.randn(8, 150) * 1000
        imu_data = np.random.randn(6, 62) * 0.5
        features = extractor.extract_all_features(emg_data, imu_data)
        X_all[i] = list(features.values())
    
    # 创建模拟标签（3类手势）
    y = np.array([0] * 17 + [1] * 17 + [2] * 16)
    
    print(f"特征矩阵形状: {X_all.shape}")
    print(f"特征数量: {len(feature_names)}")
    
    # PCA降维
    X_pca = selector.fit_transform(X_all, method='pca', n_components=20)
    print(f"PCA降维后形状: {X_pca.shape}")
    
    # 特征重要性
    print("\n特征重要性Top 10:")
    importance = selector.get_feature_importance_rf(X_all, y, feature_names)
    for i, idx in enumerate(importance['ranking'][:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importance['importance'][idx]:.4f}")
    print()


def test_save_load():
    """测试保存和加载"""
    print("=" * 60)
    print("测试保存和加载")
    print("=" * 60)
    
    selector = FeatureSelector(n_components=10)
    
    # 创建模拟数据并拟合
    np.random.seed(42)
    X = np.random.randn(100, 50)
    X_scaled = selector.fit_scaler(X)
    selector.apply_pca(X_scaled)
    
    # 保存
    save_path = 'tests/test_selector_state.pkl'
    selector.save(save_path)
    
    # 加载
    selector2 = FeatureSelector(n_components=5)  # 不同的初始值
    selector2.load(save_path)
    
    print(f"加载后n_components: {selector2.n_components}")
    print(f"PCA是否加载: {selector2.pca is not None}")
    print(f"Scaler是否加载: {selector2.scaler is not None}")
    
    # 清理测试文件
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"已清理测试文件: {save_path}")
    print()


def test_global_instance():
    """测试全局实例"""
    print("=" * 60)
    print("测试全局特征选择器实例")
    print("=" * 60)
    
    selector = get_feature_selector(n_components=15)
    print(f"全局实例n_components: {selector.n_components}")
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("特征选择与降维模块测试")
    print("=" * 60 + "\n")
    
    test_pca()
    test_lda()
    test_feature_importance()
    test_select_k_best()
    test_all_feature_importance()
    test_fit_transform()
    test_with_real_features()
    test_save_load()
    test_global_instance()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
