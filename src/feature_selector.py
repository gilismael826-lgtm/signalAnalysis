#!/usr/bin/env python3
"""
特征选择与降维模块
负责特征降维、特征重要性评估和特征可视化
"""

import numpy as np
import os
import sys

# 确保可以找到src模块
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.config import logger

try:
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn库未安装，特征选择功能将不可用。请运行: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib库未安装，可视化功能将不可用")


class FeatureSelector:
    """特征选择与降维器
    
    提供PCA、LDA降维和特征重要性评估功能
    """
    
    def __init__(self, n_components=10):
        """初始化特征选择器
        
        Args:
            n_components: 降维后的特征数量
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn库未安装，无法使用特征选择功能")
        
        self.n_components = n_components
        self.pca = None
        self.lda = None
        self.scaler = None
        self.feature_names = None
        self.selected_feature_indices = None
        
        logger.info(f"特征选择器初始化: 目标特征数={n_components}")
    
    # ==================== 数据标准化 ====================
    
    def fit_scaler(self, X):
        """拟合标准化器
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            标准化后的特征矩阵
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        logger.info(f"标准化器拟合完成: 均值形状={self.scaler.mean_.shape}")
        return X_scaled
    
    def transform_scaler(self, X):
        """使用已拟合的标准化器转换数据
        
        Args:
            X: 特征矩阵
            
        Returns:
            标准化后的特征矩阵
        """
        if self.scaler is None:
            raise ValueError("标准化器未拟合，请先调用fit_scaler()")
        return self.scaler.transform(X)
    
    # ==================== PCA降维 ====================
    
    def apply_pca(self, X, n_components=None, feature_names=None):
        """应用PCA降维
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            n_components: 降维后的特征数量，默认使用初始化时的值
            feature_names: 特征名称列表
            
        Returns:
            降维后的特征矩阵
        """
        n_components = n_components or self.n_components
        self.feature_names = feature_names
        
        # 确保n_components不超过特征数和样本数
        n_components = min(n_components, X.shape[1], X.shape[0])
        
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        logger.info(f"PCA降维: {X.shape[1]} -> {n_components}")
        logger.info(f"累计方差解释比例: {np.cumsum(self.pca.explained_variance_ratio_)[-1]:.4f}")
        
        return X_reduced
    
    def transform_pca(self, X):
        """使用已拟合的PCA转换数据
        
        Args:
            X: 特征矩阵
            
        Returns:
            降维后的特征矩阵
        """
        if self.pca is None:
            raise ValueError("PCA未拟合，请先调用apply_pca()")
        return self.pca.transform(X)
    
    def get_pca_info(self):
        """获取PCA信息
        
        Returns:
            PCA信息字典
        """
        if self.pca is None:
            return None
        
        return {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca.explained_variance_ratio_),
            'components': self.pca.components_
        }
    
    def get_pca_feature_importance(self):
        """获取PCA各主成分对原始特征的贡献
        
        Returns:
            主成分载荷矩阵
        """
        if self.pca is None:
            return None
        
        return self.pca.components_
    
    # ==================== LDA降维 ====================
    
    def apply_lda(self, X, y, n_components=None, feature_names=None):
        """应用LDA降维（有监督）
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量
            n_components: 降维后的特征数量
            feature_names: 特征名称列表
            
        Returns:
            降维后的特征矩阵
        """
        n_components = n_components or self.n_components
        self.feature_names = feature_names
        
        # LDA的n_components不能超过类别数-1
        n_classes = len(np.unique(y))
        max_components = n_classes - 1
        n_components = min(n_components, max_components, X.shape[1])
        
        if n_components < 1:
            logger.warning(f"类别数({n_classes})不足以进行LDA降维，需要至少2个类别")
            return X
        
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_reduced = self.lda.fit_transform(X, y)
        
        logger.info(f"LDA降维: {X.shape[1]} -> {X_reduced.shape[1]}")
        
        return X_reduced
    
    def transform_lda(self, X):
        """使用已拟合的LDA转换数据
        
        Args:
            X: 特征矩阵
            
        Returns:
            降维后的特征矩阵
        """
        if self.lda is None:
            raise ValueError("LDA未拟合，请先调用apply_lda()")
        return self.lda.transform(X)
    
    def get_lda_info(self):
        """获取LDA信息
        
        Returns:
            LDA信息字典
        """
        if self.lda is None:
            return None
        
        return {
            'n_components': self.lda.n_components,
            'explained_variance_ratio': self.lda.explained_variance_ratio_ if hasattr(self.lda, 'explained_variance_ratio_') else None,
            'coef': self.lda.coef_ if hasattr(self.lda, 'coef_') else None
        }
    
    # ==================== 特征重要性评估 ====================
    
    def get_feature_importance_rf(self, X, y, feature_names=None):
        """使用随机森林评估特征重要性
        
        Args:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        feature_names = feature_names or self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_dict = {
            'feature_names': feature_names,
            'importance': rf.feature_importances_,
            'ranking': np.argsort(rf.feature_importances_)[::-1]
        }
        
        logger.info(f"随机森林特征重要性评估完成，共{len(feature_names)}个特征")
        
        return importance_dict
    
    def get_feature_importance_mi(self, X, y, feature_names=None):
        """使用互信息评估特征重要性
        
        Args:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        feature_names = feature_names or self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_dict = {
            'feature_names': feature_names,
            'importance': mi_scores,
            'ranking': np.argsort(mi_scores)[::-1]
        }
        
        logger.info(f"互信息特征重要性评估完成")
        
        return importance_dict
    
    def get_feature_importance_anova(self, X, y, feature_names=None):
        """使用ANOVA F值评估特征重要性
        
        Args:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        f_scores, p_values = f_classif(X, y)
        
        feature_names = feature_names or self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_dict = {
            'feature_names': feature_names,
            'importance': f_scores,
            'p_values': p_values,
            'ranking': np.argsort(f_scores)[::-1]
        }
        
        logger.info(f"ANOVA F值特征重要性评估完成")
        
        return importance_dict
    
    def select_k_best_features(self, X, y, k=10, score_func='f_classif', feature_names=None):
        """选择K个最佳特征
        
        Args:
            X: 特征矩阵
            y: 标签向量
            k: 选择的特征数量
            score_func: 评分函数 ('f_classif' 或 'mutual_info')
            feature_names: 特征名称列表
            
        Returns:
            选择后的特征矩阵和特征索引
        """
        if score_func == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        
        self.selected_feature_indices = selected_indices
        self.feature_names = feature_names
        
        if feature_names:
            selected_names = [feature_names[i] for i in selected_indices]
            logger.info(f"选择了{k}个最佳特征: {selected_names[:5]}...")
        else:
            logger.info(f"选择了{k}个最佳特征，索引: {selected_indices}")
        
        return X_selected, selected_indices
    
    def get_all_feature_importance(self, X, y, feature_names=None):
        """获取所有特征重要性评估结果
        
        Args:
            X: 特征矩阵
            y: 标签向量
            feature_names: 特征名称列表
            
        Returns:
            综合特征重要性字典
        """
        feature_names = feature_names or self.feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        rf_importance = self.get_feature_importance_rf(X, y, feature_names)
        mi_importance = self.get_feature_importance_mi(X, y, feature_names)
        anova_importance = self.get_feature_importance_anova(X, y, feature_names)
        
        # 归一化各重要性分数
        rf_norm = rf_importance['importance'] / np.sum(rf_importance['importance'])
        mi_norm = mi_importance['importance'] / np.sum(mi_importance['importance'])
        anova_norm = anova_importance['importance'] / np.sum(anova_importance['importance'])
        
        # 综合排名（平均排名）
        rf_rank = np.argsort(np.argsort(rf_importance['importance'])[::-1])
        mi_rank = np.argsort(np.argsort(mi_importance['importance'])[::-1])
        anova_rank = np.argsort(np.argsort(anova_importance['importance'])[::-1])
        
        avg_rank = (rf_rank + mi_rank + anova_rank) / 3
        
        return {
            'feature_names': feature_names,
            'random_forest': {
                'importance': rf_importance['importance'],
                'normalized': rf_norm,
                'ranking': rf_importance['ranking']
            },
            'mutual_information': {
                'importance': mi_importance['importance'],
                'normalized': mi_norm,
                'ranking': mi_importance['ranking']
            },
            'anova_f': {
                'importance': anova_importance['importance'],
                'p_values': anova_importance['p_values'],
                'normalized': anova_norm,
                'ranking': anova_importance['ranking']
            },
            'combined': {
                'avg_rank': avg_rank,
                'ranking': np.argsort(avg_rank)
            }
        }
    
    # ==================== 特征可视化 ====================
    
    def plot_pca_variance(self, save_path=None):
        """绘制PCA方差解释比例图
        
        Args:
            save_path: 保存路径，如果为None则显示图形
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制图形")
            return
        
        if self.pca is None:
            logger.warning("PCA未拟合，无法绘制图形")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 方差解释比例
        axes[0].bar(range(1, len(self.pca.explained_variance_ratio_) + 1),
                   self.pca.explained_variance_ratio_,
                   alpha=0.7, color='steelblue')
        axes[0].set_xlabel('主成分')
        axes[0].set_ylabel('方差解释比例')
        axes[0].set_title('各主成分方差解释比例')
        axes[0].grid(True, alpha=0.3)
        
        # 累计方差解释比例
        cumulative = np.cumsum(self.pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumulative) + 1), cumulative, 'bo-', markersize=4)
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
        axes[1].axhline(y=0.90, color='g', linestyle='--', label='90%阈值')
        axes[1].set_xlabel('主成分数量')
        axes[1].set_ylabel('累计方差解释比例')
        axes[1].set_title('累计方差解释比例')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PCA方差图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, importance_dict, top_n=20, save_path=None):
        """绘制特征重要性图
        
        Args:
            importance_dict: 特征重要性字典
            top_n: 显示前N个重要特征
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制图形")
            return
        
        feature_names = importance_dict['feature_names']
        importance = importance_dict['importance']
        ranking = importance_dict['ranking']
        
        # 获取top_n特征
        top_indices = ranking[:top_n]
        top_names = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
        
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_importance[::-1], align='center', color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel('重要性分数')
        ax.set_title(f'特征重要性排名 (Top {top_n})')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_correlation(self, X, feature_names=None, save_path=None):
        """绘制特征相关性热力图
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制图形")
            return
        
        feature_names = feature_names or self.feature_names or [f'f{i}' for i in range(X.shape[1])]
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(X.T)
        
        # 限制显示的特征数量（热力图太大不易查看）
        max_features = 50
        if len(feature_names) > max_features:
            logger.info(f"特征数量({len(feature_names)})超过{max_features}，只显示前{max_features}个特征的相关性")
            corr_matrix = corr_matrix[:max_features, :max_features]
            feature_names = feature_names[:max_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_yticklabels(feature_names, fontsize=6)
        ax.set_title('特征相关性矩阵')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"特征相关性图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pca_2d(self, X_pca, y=None, save_path=None):
        """绘制PCA 2D散点图
        
        Args:
            X_pca: PCA降维后的数据（至少2维）
            y: 标签向量
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制图形")
            return
        
        if X_pca.shape[1] < 2:
            logger.warning("PCA降维后特征数不足2，无法绘制2D散点图")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if y is not None:
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=str(label), alpha=0.6, s=30)
            ax.legend()
        else:
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
        
        ax.set_xlabel('主成分1')
        ax.set_ylabel('主成分2')
        ax.set_title('PCA 2D可视化')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"PCA 2D散点图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_lda_2d(self, X_lda, y, save_path=None):
        """绘制LDA 2D散点图
        
        Args:
            X_lda: LDA降维后的数据
            y: 标签向量
            save_path: 保存路径
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib未安装，无法绘制图形")
            return
        
        if X_lda.shape[1] < 2:
            # 如果只有1维，绘制1D图
            fig, ax = plt.subplots(figsize=(10, 4))
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_lda[mask, 0], np.zeros(np.sum(mask)), 
                          c=[colors[i]], label=str(label), alpha=0.6, s=30)
            ax.set_xlabel('判别成分1')
            ax.set_title('LDA 1D可视化')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = np.unique(y)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = y == label
                ax.scatter(X_lda[mask, 0], X_lda[mask, 1], 
                          c=[colors[i]], label=str(label), alpha=0.6, s=30)
            ax.set_xlabel('判别成分1')
            ax.set_ylabel('判别成分2')
            ax.set_title('LDA 2D可视化')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"LDA散点图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    # ==================== 综合处理 ====================
    
    def fit_transform(self, X, y=None, method='pca', n_components=None, feature_names=None):
        """拟合并转换数据
        
        Args:
            X: 特征矩阵
            y: 标签向量（LDA需要）
            method: 降维方法 ('pca' 或 'lda')
            n_components: 目标特征数
            feature_names: 特征名称列表
            
        Returns:
            降维后的特征矩阵
        """
        n_components = n_components or self.n_components
        
        # 先标准化
        X_scaled = self.fit_scaler(X)
        
        if method == 'pca':
            return self.apply_pca(X_scaled, n_components, feature_names)
        elif method == 'lda':
            if y is None:
                raise ValueError("LDA需要标签向量y")
            return self.apply_lda(X_scaled, y, n_components, feature_names)
        else:
            raise ValueError(f"未知的降维方法: {method}")
    
    def transform(self, X, method='pca'):
        """转换新数据
        
        Args:
            X: 特征矩阵
            method: 降维方法
            
        Returns:
            降维后的特征矩阵
        """
        X_scaled = self.transform_scaler(X)
        
        if method == 'pca':
            return self.transform_pca(X_scaled)
        elif method == 'lda':
            return self.transform_lda(X_scaled)
        else:
            raise ValueError(f"未知的降维方法: {method}")
    
    def save(self, filepath):
        """保存特征选择器状态
        
        Args:
            filepath: 保存路径
        """
        import joblib
        
        state = {
            'n_components': self.n_components,
            'pca': self.pca,
            'lda': self.lda,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_feature_indices': self.selected_feature_indices
        }
        
        joblib.dump(state, filepath)
        logger.info(f"特征选择器已保存: {filepath}")
    
    def load(self, filepath):
        """加载特征选择器状态
        
        Args:
            filepath: 文件路径
        """
        import joblib
        
        state = joblib.load(filepath)
        
        self.n_components = state['n_components']
        self.pca = state['pca']
        self.lda = state['lda']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.selected_feature_indices = state['selected_feature_indices']
        
        logger.info(f"特征选择器已加载: {filepath}")


# 全局特征选择器实例
feature_selector = None

def get_feature_selector(n_components=10):
    """获取全局特征选择器实例
    
    Args:
        n_components: 目标特征数
        
    Returns:
        FeatureSelector实例
    """
    global feature_selector
    if feature_selector is None:
        feature_selector = FeatureSelector(n_components=n_components)
    return feature_selector
