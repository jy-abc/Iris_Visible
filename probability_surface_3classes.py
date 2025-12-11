"""概率曲面可视化模块--三分类"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_iris_data
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def create_probability_surface():
    """创建3D三分类概率曲面"""
    # 加载数据
    iris = load_iris_data()
    feat_idx = [0, 2, 3]  # Sepal Length, Petal Length, Petal Width
    
    X, y = iris['X'][:, feat_idx], iris['y']
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练逻辑回归
    lr = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)
    lr.fit(X_scaled, y)
    
    # 获取模型参数
    coef = lr.coef_  # (3, 3)
    
    # 三峰曲面生成
    grid_size = 80
    x = y = np.linspace(-40, 40, grid_size)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # 1. Setosa峰 (红色)
    height0 = 80 + 20 * np.linalg.norm(coef[0])  
    width0 = 25 / (1 + np.linalg.norm(coef[0])/2)  
    center0 = [25, 20] 
    
    # 2. Virginica峰 (黄色)
    height1 = 70 + 20 * np.linalg.norm(coef[1])
    width1 = 28 / (1 + np.linalg.norm(coef[1])/2)
    center1 = [-20, -15]  
    
    # 3. Versicolor峰 (蓝色)
    height2 = 75 + 20 * np.linalg.norm(coef[2])
    width2 = 26 / (1 + np.linalg.norm(coef[2])/2)
    center2 = [5, -25]  
    
    # 生成三个高斯峰
    def gaussian_peak(x, y, cx, cy, h, w):
        return h * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w**2))
    
    peak0 = gaussian_peak(X_grid, Y_grid, center0[0], center0[1], height0, width0)
    peak1 = gaussian_peak(X_grid, Y_grid, center1[0], center1[1], height1, width1)
    peak2 = gaussian_peak(X_grid, Y_grid, center2[0], center2[1], height2, width2)
    
    # 使用softmax组合三个峰
    peaks_stack = np.stack([peak0, peak1, peak2], axis=-1)
    softmax_weights = np.exp(peaks_stack) / np.sum(np.exp(peaks_stack), axis=-1, keepdims=True)
    
    # 组合后的曲面
    Z_surf = (softmax_weights[:, :, 0] * peak0 + 
              softmax_weights[:, :, 1] * peak1 + 
              softmax_weights[:, :, 2] * peak2)
    
    
    wave = 0.05 * np.sin(X_grid/15) * np.cos(Y_grid/12)
    Z_surf = Z_surf * (1 + wave)
    
    # 自定义颜色映射 
    colors = [
        (0.78, 0.45, 0.45),     # 粉红色
        (0.88, 0.55, 0.35),     # 橙色 (过渡色)
        (1.0, 0.82, 0.4),       # 金黄色 
        (0.55, 0.88, 0.55),     # 绿色 (过渡色)
        (0.45, 0.65, 1.0),      # 蓝色 
        (0.35, 0.5, 0.95)       # 蓝灰色 
    ]
    
    custom_cmap = LinearSegmentedColormap.from_list('ThreeClass', colors, N=256)
    
    # 确定每个点的主要类别
    peak_values = np.stack([peak0, peak1, peak2], axis=-1)
    dominant_class = np.argmax(peak_values, axis=-1)
    
    # 创建颜色数组
    facecolors = np.zeros((grid_size, grid_size, 4))
    for i in range(grid_size):
        for j in range(grid_size):
            if dominant_class[i, j] == 0:  # Setosa - 红色
                hue = 0.15 + 0.2 * (Z_surf[i, j] / Z_surf.max())
                color = custom_cmap(hue)
            elif dominant_class[i, j] == 1:  # Virginica - 黄色
                hue = 0.45 + 0.2 * (Z_surf[i, j] / Z_surf.max())
                color = custom_cmap(hue)
            else:  # Versicolor - 蓝色
                hue = 0.75 + 0.2 * (Z_surf[i, j] / Z_surf.max())
                color = custom_cmap(hue)
            facecolors[i, j] = color
    
    # 绘图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制主曲面
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_surf,
        facecolors=facecolors,
        rstride=2, cstride=2,
        alpha=0.6,
        edgecolor='#555555',
        linewidth=0.2,
        shade=True
    )
    
    #创建投影面
    
    # XY面投影 (投影到Z=-100平面)
    xy_projection_z = np.full_like(Z_surf, -100)
    
    # 创建XY投影面
    xy_surf = ax.plot_surface(
        X_grid, Y_grid, xy_projection_z,
        rstride=2, cstride=2,
        facecolors=facecolors,
        alpha=0.8,
        shade=False,
        linewidth=0.1
    )
    
    # YZ面投影 (投影到X=-40平面)
    yz_projection_x = np.full_like(Y_grid, -40)
    
    # 创建YZ投影面
    yz_surf = ax.plot_surface(
        yz_projection_x, Y_grid, Z_surf,
        rstride=2, cstride=2,
        facecolors=facecolors,
        alpha=0.8,
        shade=False,
        linewidth=0.1
    )
    
    # XZ面投影 (投影到Y=40平面)
    xz_projection_y = np.full_like(X_grid, 40)
    
    # 创建XZ投影面
    xz_surf = ax.plot_surface(
        X_grid, xz_projection_y, Z_surf,
        rstride=2, cstride=2,
        facecolors=facecolors,
        alpha=0.8,
        shade=False,
        linewidth=0.1
    )
    
    # 颜色条
    # 创建归一化对象
    norm = Normalize(vmin=0, vmax=Z_surf.max())
    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])

    # 创建颜色条
    cbar = fig.colorbar(sm, ax=ax, pad=0.15, shrink=0.6)
    cbar.set_label('Probability Surface Height', fontsize=12, fontweight='bold')
    
    # 图例
    legend_elements = [
        Patch(facecolor=colors[0], label='Setosa (0)', alpha=0.9),
        Patch(facecolor=colors[2], label='Virginica (1)', alpha=0.9),
        Patch(facecolor=colors[4], label='Versicolor (2)', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # 坐标轴设置
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)
    ax.set_title('3D Probability Map: Setosa vs Virginica vs Versicolor', fontsize=16, pad=8, fontweight='bold')
    ax.set_xlabel('Sepal Length\nX1', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Petal Length\nX2', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_zlabel('Petal Width\nX3', fontsize=14, labelpad=15, fontweight='bold')
    
    # 美化设置
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('gray')
    ax.grid(True, linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
    
    ax.view_init(elev=25, azim=305)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def run_probability_surface_3c():
    """运行概率曲面可视化"""
    print("\n正在生成3D概率曲面可视化--三分类...")
    fig = create_probability_surface()
    plt.show()
    return fig
