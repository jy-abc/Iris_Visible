"""3D分类可视化模块--三分类"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.linear_model import LogisticRegression
from skimage import measure
from data_loader import load_iris_data

def create_3d_visualization_matplotlib():
    """创建静态3D可视化 (Matplotlib)"""
    iris = load_iris_data()
    X = iris['X'][:, [0, 2, 3]]
    y = iris['y']
    
    # 模型
    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X, y)
    
    # 网格
    lims = [[X[:, i].min() - 0.3, X[:, i].max() + 0.3] for i in range(3)]
    xx, yy, zz = np.meshgrid(
        *[np.linspace(lims[i][0], lims[i][1], 50) for i in range(3)], 
        indexing='ij'
    )
    grid_pts = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # 自定义三色渐变图
    colors = ['#FF6F61', '#3A5FCD', '#2E8B57']
    cmap3 = mcolors.LinearSegmentedColormap.from_list('iris3', colors, N=256)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点
    markers = ['o', '^', 's']
    labels = ['Setosa', 'Virginica', 'Versicolor']
    for k in range(3):
        mask = y == k
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=[k] * mask.sum(),
                   marker=markers[k],
                   s=60, edgecolor='k',
                   cmap=cmap3, vmin=0, vmax=2, label=labels[k])
    
    # 绘制决策边界
    for pair in [(0, 1), (1, 2)]:
        mask_pair = np.isin(y, pair)
        X_pair, y_pair = X[mask_pair], (y[mask_pair] == pair[1]).astype(int)
        bin_model = LogisticRegression(max_iter=1000)
        bin_model.fit(X_pair, y_pair)
        proba = bin_model.predict_proba(grid_pts)[:, 1].reshape(xx.shape)
        verts, faces, _, _ = measure.marching_cubes(
            proba, 0.5,
            spacing=[(lims[i][1] - lims[i][0]) / 49 for i in range(3)])
        for i in range(3):
            verts[:, i] += lims[i][0]
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces, color='purple',
                        alpha=0.25, linewidth=0.1)
    
    # 美化坐标轴
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 添加颜色条
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap3, norm=plt.Normalize(vmin=0, vmax=2)),
        ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_ticks([0, 0.5, 1, 1.5, 2])
    cbar.ax.set_yticklabels(['0', '0.5', '1', '1.5', '2'])
    
    # 设置标签
    ax.set_xlabel('Sepal Length\nX1', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Petal Length\nX2', fontsize=14, labelpad=15, fontweight='bold')  
    ax.set_zlabel('Petal Width\nX3', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_title('3D Classification: Setosa vs Virginica vs Versicolor', 
                 fontsize=16, pad=25, fontweight='bold')
    ax.legend()
    
    return fig

def run_3d_visualization():
    """运行3D可视化"""
    print("\n正在进行3D分类可视化--三分类...")
    fig= create_3d_visualization_matplotlib()
    plt.show()
    
    return fig