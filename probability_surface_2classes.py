"""概率曲面可视化模块--二分类"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_iris_data

def create_probability_surface():
    """创建3D概率曲面"""
    iris = load_iris_data()
    feat_idx = [0, 2, 3]  # Sepal Length, Petal Length, Petal Width
    mask = (iris['y'] == 0) | (iris['y'] == 2)
    X, y = iris['X'][mask][:, feat_idx], iris['y'][mask]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_scaled, y)
    
    # 3D双峰曲面生成
    coef = lr.coef_[0]
    peak_offset_x = 20 * np.sign(coef[0])
    peak_offset_y = 10 * np.sign(coef[1])
    peak_width_p0 = 800 / (np.abs(coef[0]) + np.abs(coef[1]) + 1)
    peak_width_p1 = 1000 / (np.abs(coef[0]) + np.abs(coef[2]) + 1)
    
    # 生成网格数据
    x = y = np.linspace(-40, 40, 50)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # 构建双峰值曲面
    wave_effect = 0.2 * np.sin(X_grid/6) * np.cos(Y_grid/6)
    peak_p0 = -80 * np.exp(-((X_grid + peak_offset_x)**2 + (Y_grid + peak_offset_y)** 2) / peak_width_p0) * (1 + wave_effect)
    peak_p1 = 80 * np.exp(-((X_grid - peak_offset_x)**2 + (Y_grid - peak_offset_y)** 2) / peak_width_p1) * (1 + wave_effect)
    Z_surf = peak_p0 + peak_p1
    
    # 逻辑回归概率计算
    def get_lr_prob(grid_x, grid_y, fixed_feat_val):
        n = grid_x.size
        X_pred = np.empty((n, 3))
        X_pred[:, 0] = grid_x.ravel()
        X_pred[:, 1] = grid_y.ravel()
        X_pred[:, 2] = fixed_feat_val
        prob = lr.predict_proba(scaler.transform(X_pred))[:, 1]
        return prob.reshape(grid_x.shape)
    
    fixed_feat2 = X_scaled[:, 2].mean()
    lr_prob = get_lr_prob(X_grid, Y_grid, fixed_feat2)
    
    # 基础概率分布
    prob_base = (np.exp(-((X_grid - peak_offset_x)**2 + (Y_grid - peak_offset_y)** 2) / peak_width_p1) -
                 np.exp(-((X_grid + peak_offset_x)**2 + (Y_grid + peak_offset_y)** 2) / peak_width_p0))
    prob_base *= (0.6 + 0.4 * lr_prob)
    
    # 绘图
    fig = plt.figure(figsize=(14, 12), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D概率曲面
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_surf,
        rstride=2, cstride=2,
        cmap='coolwarm',
        alpha=0.6,
        edgecolor='blue',
        linewidth=0.8,
        shade=True
    )
    
    # XY面投影 (投影到Z=-100平面)
    xy_projection_z = np.full_like(Z_surf, -100)
    
    # 创建XY投影面
    xy_surf = ax.plot_surface(
        X_grid, Y_grid, xy_projection_z,
        rstride=2, cstride=2,
        facecolors=plt.cm.coolwarm((Z_surf - Z_surf.min()) / (Z_surf.max() - Z_surf.min())),
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
        facecolors=plt.cm.coolwarm((Z_surf - Z_surf.min()) / (Z_surf.max() - Z_surf.min())),
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
        facecolors=plt.cm.coolwarm((Z_surf - Z_surf.min()) / (Z_surf.max() - Z_surf.min())),
        alpha=0.8,
        shade=False,
        linewidth=0.1
    )
    
    # 添加等高线投影作为辅助
    # XY投影等高线
    ax.contour(
        X_grid, Y_grid, Z_surf,
        zdir='z', offset=-100,
        colors='k',
        alpha=0.3,
        linewidths=0.5,
        levels=10
    )
    
    # YZ投影等高线
    ax.contour(
        Z_surf, Y_grid, X_grid,
        zdir='x', offset=-40,
        colors='k',
        alpha=0.3,
        linewidths=0.5,
        levels=10
    )
    
    # XZ投影等高线
    ax.contour(
        X_grid, Z_surf, Y_grid,
        zdir='y', offset=40,
        colors='k',
        alpha=0.3,
        linewidths=0.5,
        levels=10
    )
    
    # 添加颜色条
    cbar = plt.colorbar(surf, ax=ax, pad=0.15, shrink=0.6)
    cbar.set_label('Probability Surface Height', fontsize=12, fontweight='bold')
    
    # 坐标轴设置
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)
    ax.set_title('3D Probability Map: Setosa vs Versicolor', fontsize=16, pad=8, fontweight='bold')
    ax.set_xlabel('Sepal Length\nX1', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Petal Length\nX2', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_zlabel('Petal Width\nX3', fontsize=14, labelpad=15, fontweight='bold')
    
    # 美化设置
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('gray')
    ax.grid(True, linestyle='-', color='gray', alpha=0.3, linewidth=0.5)
    
    # 视角调整
    ax.view_init(elev=30, azim=300)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def run_probability_surface():
    """运行概率曲面可视化"""
    print("\n正在生成3D概率曲面可视化--二分类...")
    fig = create_probability_surface()
    plt.show()
    return fig
