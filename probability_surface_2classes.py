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
    
    # 三个平面的2D投影计算 
    proj_xy = prob_base
    
    # YZ面投影
    Y_yz = np.linspace(-40, 40, 50)
    Z_yz = np.linspace(-100, 100, 50)
    Y_yz_grid, Z_yz_grid = np.meshgrid(Y_yz, Z_yz)
    fixed_feat0_yz = 0
    lr_prob_yz = get_lr_prob(np.full_like(Y_yz_grid, fixed_feat0_yz), Y_yz_grid, fixed_feat2)
    
    z_wave = 0.2 * np.sin(Z_yz_grid/8)
    proj_yz = (np.exp(-((fixed_feat0_yz - peak_offset_x)**2 + (Y_yz_grid - peak_offset_y)** 2) / peak_width_p1) -
               np.exp(-((fixed_feat0_yz + peak_offset_x)**2 + (Y_yz_grid + peak_offset_y)** 2) / peak_width_p0)) * (1 + z_wave)
    proj_yz *= (0.6 + 0.4 * lr_prob_yz)
    
    # XZ面投影
    X_xz = np.linspace(-40, 40, 50)
    Z_xz = np.linspace(-100, 100, 50)
    X_xz_grid, Z_xz_grid = np.meshgrid(X_xz, Z_xz)
    fixed_feat1_xz = 40
    lr_prob_xz = get_lr_prob(X_xz_grid, np.full_like(X_xz_grid, fixed_feat1_xz), fixed_feat2)
    
    z_wave_xz = 0.2 * np.cos(Z_xz_grid/8)
    proj_xz = (np.exp(-((X_xz_grid - peak_offset_x)**2 + (fixed_feat1_xz - peak_offset_y)** 2) / peak_width_p1) -
               np.exp(-((X_xz_grid + peak_offset_x)**2 + (fixed_feat1_xz + peak_offset_y)** 2) / peak_width_p0)) * (1 + z_wave_xz)
    proj_xz *= (0.6 + 0.4 * lr_prob_xz)
    
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
    
    # XY面投影
    ax.contourf(
        X_grid, Y_grid, proj_xy,
        zdir='z', offset=-100,
        cmap='coolwarm',
        alpha=0.8,
        levels=np.linspace(proj_xy.min(), proj_xy.max(), 20)
    )
    
    # YZ面投影
    ax.contourf(
        np.full_like(Y_yz_grid, -40),
        Y_yz_grid, Z_yz_grid,
        proj_yz,
        zdir='x', offset=-40,
        cmap='coolwarm',
        alpha=0.8,
        levels=np.linspace(proj_yz.min(), proj_yz.max(), 20)
    )
    
    # XZ面投影
    ax.contourf(
        X_xz_grid,
        np.full_like(X_xz_grid, 40),
        Z_xz_grid,
        proj_xz,
        zdir='y', offset=40,
        cmap='coolwarm',
        alpha=0.8,
        levels=np.linspace(proj_xz.min(), proj_xz.max(), 20)
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
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig

def run_probability_surface():
    """运行概率曲面可视化"""
    print("\n正在生成3D概率曲面可视化--二分类...")
    fig = create_probability_surface()
    plt.show()
    return fig