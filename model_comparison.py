"""多模型决策边界比较模块"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import (
    KBinsDiscretizer, PolynomialFeatures, SplineTransformer
)
from data_loader import load_iris_data

def train_models(X, y):
    """训练多个模型"""
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    
    # 训练不同模型
    models = {}
    
    # 1. 逻辑回归模型
    models['LR'] = LogisticRegression(max_iter=200)
    models['LR'].fit(X_train, y_train)
    
    # 2. 高斯过程分类模型
    models['GaussianProcess'] = GaussianProcessClassifier()
    models['GaussianProcess'].fit(X_train, y_train)
    
    # 3. 逻辑回归 + RBF 手工特征模型
    models['LR_RBF'] = make_pipeline(
        Nystroem(kernel='rbf', gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(max_iter=200)        
    )
    models['LR_RBF'].fit(X_train, y_train)
    
    # 4. 直方图梯度提升树模型
    models['GradientBoosting'] = HistGradientBoostingClassifier()
    models['GradientBoosting'].fit(X_train, y_train)
    
    # 5. 逻辑回归 + 分箱 + 交互项模型
    models['LR_Binning'] = make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method='averaged_inverted_cdf'),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(max_iter=200),
    )
    models['LR_Binning'].fit(X_train, y_train)
    
    # 6. 逻辑回归 + 样条 + 交互项模型
    models['LR_Spline'] = make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(max_iter=200),
    )
    models['LR_Spline'].fit(X_train, y_train)
    
    return models, X_train, X_test, y_train, y_test

def visualize_decision_boundaries(models, X, y):
    """可视化决策边界"""
    # 创建网格
    xx, yy = np.meshgrid(
        np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
        np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1),
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 计算概率
    probs = {}
    for name, model in models.items():
        p = model.predict_proba(grid_points)
        probs[name] = p.reshape(xx.shape[0], xx.shape[1], 3)
    
    # 预测类别
    Z = {}
    for key, model in models.items():
        Z[key] = model.predict(grid_points)
        Z[key] = Z[key].reshape(xx.shape)
    
    # 创建图形
    fig, axs = plt.subplots(6, 4, figsize=(12, 9))
    fig.subplots_adjust(left=0.22, top=0.85, bottom=0.06, hspace=0.35)
    
    # 设置颜色
    class_colors = ['red', 'green', 'blue'] 
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    cmap = plt.cm.colors.ListedColormap(class_colors)
    
    # 绘制决策边界
    for index, model_name in enumerate(models.keys()):
        axs[index][0].imshow(Z[model_name], extent=extent, origin='lower',
                            cmap=cmap, alpha=0.6)
        axs[index][0].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                              marker='o', s=50, cmap=cmap)
        if index == 0:
            axs[index][0].set_title('Decision Boundary')
    
    # 绘制概率图
    for j, model_name in enumerate(models.keys()):
        for i, class_prob in enumerate(probs[model_name].transpose(2, 0, 1)):
            ax = axs[j][i + 1]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                f'class_{i}_clormap', ['white', class_colors[i]], N=256
            )
            contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=cmap)
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', 
                       marker='o', s=50, cmap=cmap, alpha=1)
            if j == 0:
                ax.set_title(f'Class {i} Probability')
            if j == 5:
                fig.colorbar(contour, ax=ax, orientation='horizontal', pad=0.2)
    
    # 添加标签
    for i, model in enumerate(models.keys()):
        axs[i][0].set_ylabel(model, rotation=90, labelpad=8, fontsize=9)
        axs[i][0].yaxis.set_label_position('left')
    
    fig.text(0.98, 0.03, '---x:Petal Length', ha='right', va='bottom', fontsize=10)
    fig.text(0.98, 0.01, ' ---y:Petal Width', ha='right', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    return fig

def run_model_comparison():
    """运行模型比较"""
    print("\n正在训练多个分类器并进行决策边界比较...")
    
    data = load_iris_data()
    X = data['X'][:, 2:]  # 只使用后两个特征
    y = data['y']
    
    models, X_train, X_test, y_train, y_test = train_models(X, y)
    fig = visualize_decision_boundaries(models, X, y)
    
    plt.show()
    return models