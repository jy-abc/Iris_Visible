import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_loader import load_iris_data

def create_probability_surface():
    '''创建3D概率曲面'''
    iris = load_iris_data()
    feat_idx = [0, 2, 3]  # Sepal Length, Petal Length, Petal Width
    X, y = iris['X'][:, feat_idx], iris['y']  # 全部三分类数据

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练逻辑回归
    lr = LogisticRegression(multi_class='ovr', random_state=42, max_iter=200)
    lr.fit(X_scaled, y)

    #3D三峰曲面生成
    coef = lr.coef_
    # 三类峰值偏移
    peak_offsets = [
        [15 * np.sign(coef[0,0]), 10 * np.sign(coef[0,1])], 
        [0, 0],                                             
        [-15 * np.sign(coef[2,0]), -10 * np.sign(coef[2,1])]
    ]
    # 峰值宽度（控制峰的扩散程度）
    peak_widths = [
        600 / (np.linalg.norm(coef[0]) + 1),  # 类别0宽度
        700 / (np.linalg.norm(coef[1]) + 1),  # 类别1宽度
        800 / (np.linalg.norm(coef[2]) + 1)   # 类别2宽度
    ]

    # 生成网格数据
    x = y = np.linspace(-40, 40, 50)
    X_grid, Y_grid = np.meshgrid(x, y)

    # 构建峰值曲面
    wave_effect = 0.2 * np.sin(X_grid/8) * np.cos(Y_grid/8) 
    # 类别0峰值（负向）
    peak0 = -60 * np.exp(-((X_grid - peak_offsets[0][0])**2 + (Y_grid - peak_offsets[0][1])**2) / peak_widths[0]) * (1 + wave_effect)
    # 类别1峰值（中间）
    peak1 = 30 * np.exp(-((X_grid - peak_offsets[1][0])**2 + (Y_grid - peak_offsets[1][1])**2) / peak_widths[1]) * (1 + 0.8*wave_effect)
    # 类别2峰值（正向）
    peak2 = 60 * np.exp(-((X_grid - peak_offsets[2][0])**2 + (Y_grid - peak_offsets[2][1])**2) / peak_widths[2]) * (1 + wave_effect)
    # 总曲面（三类叠加）
    Z_surf = peak0 + peak1 + peak2

    #三分类概率计算
    def get_lr_prob_3class(grid_x, grid_y, fixed_feat_val):
        n = grid_x.size
        X_pred = np.empty((n, 3))
        X_pred[:, 0] = grid_x.ravel()  
        X_pred[:, 1] = grid_y.ravel()  
        X_pred[:, 2] = fixed_feat_val  
        prob = lr.predict_proba(scaler.transform(X_pred))
        # 返回：类别0概率、类别1概率、类别2概率、最大概率类别
        return prob[:,0].reshape(grid_x.shape), prob[:,1].reshape(grid_x.shape), prob[:,2].reshape(grid_x.shape)

    # 固定Petal Width为数据均值
    fixed_feat2 = X_scaled[:, 2].mean()
    prob0, prob1, prob2 = get_lr_prob_3class(X_grid, Y_grid, fixed_feat2)
    # 融合三类概率作为基础投影（加权组合）
    prob_base = 0.4*prob0 + 0.3*prob1 + 0.3*prob2

    # 三个平面的2D投影计算
    # 1. XY面投影
    proj_xy = prob_base

    # 2. YZ面投影
    Y_yz = np.linspace(-40, 40, 50)
    Z_yz = np.linspace(-100, 100, 50)
    Y_yz_grid, Z_yz_grid = np.meshgrid(Y_yz, Z_yz)
    fixed_feat0_yz = -40  # 固定X=-40
    prob0_yz, prob1_yz, prob2_yz = get_lr_prob_3class(
        np.full_like(Y_yz_grid, fixed_feat0_yz), Y_yz_grid, fixed_feat2
    )
    # YZ面投影
    z_wave = 0.2 * np.sin(Z_yz_grid/10)
    proj_yz = (0.4*prob0_yz + 0.3*prob1_yz + 0.3*prob2_yz) * (1 + z_wave)

    # 3. XZ面投影
    X_xz = np.linspace(-40, 40, 50)
    Z_xz = np.linspace(-100, 100, 50)
    X_xz_grid, Z_xz_grid = np.meshgrid(X_xz, Z_xz)
    fixed_feat1_xz = 40  # 固定Y=40
    prob0_xz, prob1_xz, prob2_xz = get_lr_prob_3class(
        X_xz_grid, np.full_like(X_xz_grid, fixed_feat1_xz), fixed_feat2
    )
    # XZ面投影
    z_wave_xz = 0.2 * np.cos(Z_xz_grid/10)
    proj_xz = (0.4*prob0_xz + 0.3*prob1_xz + 0.3*prob2_xz) * (1 + z_wave_xz)

    # 绘图设置
    fig = plt.figure(figsize=(16, 14), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # 配色映射
    cmap = plt.cm.RdYlBu_r  # 红(Setosa)-黄(Virginica)-蓝(Versicolor)
    # 计算三类概率的极值，用于颜色范围固定
    vmin = Z_surf.min()
    vmax = Z_surf.max()

    # 绘制3D三分类概率曲面
    surf = ax.plot_surface(
        X_grid, Y_grid, Z_surf,
        rstride=2, cstride=2,
        cmap=cmap,
        vmin=vmin, vmax=vmax,  # 固定颜色范围
        alpha=0.7,
        edgecolor='navy',
        linewidth=0.6,
        shade=True
    )

    # 绘制XY面投影（Z=-100）
    ax.contourf(
        X_grid, Y_grid, proj_xy,
        zdir='z', offset=-100,
        cmap=cmap,
        vmin=proj_xy.min(), vmax=proj_xy.max(),
        alpha=0.8,
        levels=np.linspace(proj_xy.min(), proj_xy.max(), 25)
    )

    # 绘制YZ面投影（X=-40）
    ax.contourf(
        np.full_like(Y_yz_grid, -40),
        Y_yz_grid, Z_yz_grid,
        proj_yz,
        zdir='x', offset=-40,
        cmap=cmap,
        vmin=proj_yz.min(), vmax=proj_yz.max(),
        alpha=0.8,
        levels=np.linspace(proj_yz.min(), proj_yz.max(), 25)
    )

    # 绘制XZ面投影（Y=40）
    ax.contourf(
        X_xz_grid,
        np.full_like(X_xz_grid, 40),
        Z_xz_grid,
        proj_xz,
        zdir='y', offset=40,
        cmap=cmap,
        vmin=proj_xz.min(), vmax=proj_xz.max(),
        alpha=0.8,
        levels=np.linspace(proj_xz.min(), proj_xz.max(), 25)
    )

    #颜色条设置
    cbar = plt.colorbar(surf, ax=ax, pad=0.15, shrink=0.7)
    cbar.set_label('3-Class Probability Surface Height', fontsize=12, fontweight='bold')

    # 创建图例说明框（bbox_to_anchor控制位置，loc控制对齐方式）
    legend_text = [
        'Setosa (0) - Red',
        'Virginica (1) - Yellow',
        'Versicolor (2) - Blue'
    ]
    # 生成颜色块+文字的图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#bf3a51', edgecolor='black', label='Setosa (0)'),
        Patch(facecolor='#fff7c6', edgecolor='black', label=' Virginica(1)'),
        Patch(facecolor='#727cb9', edgecolor='black', label='Versicolor (2)')
    ]

    # 添加图例到右上角
    ax.legend(
        handles=legend_elements,
        title='Class - Color Mapping',
        title_fontsize=12,
        fontsize=11,
        loc='upper right',  # 右上角对齐
        bbox_to_anchor=(1.3, 1.0),  # 调整位置（超出坐标轴范围）
        frameon=True,
        facecolor='white',  # 背景白色
        edgecolor='gray',   # 边框灰色
        shadow=True         # 阴影效果
    )

    # 坐标轴设置
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)
    ax.set_title('3D Probability Map: Iris 3-Class Classification (Setosa/Versicolor/Virginica)', 
                fontsize=16, pad=15, fontweight='bold')
    ax.set_xlabel('Sepal Length (X1)\nScaled', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Petal Length (X2)\nScaled', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_zlabel('Petal Width (X3)\nScaled', fontsize=14, labelpad=15, fontweight='bold')

    # 透明面板+网格
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('gray')
    ax.grid(True, linestyle='-', color='gray', alpha=0.3, linewidth=0.5)

    # 视角调整
    ax.view_init(elev=25, azim=55)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def run_probability_surface_3c():
    """运行概率曲面可视化"""
    print("\n正在生成3D概率曲面可视化--三分类...")
    fig = create_probability_surface()
    plt.show()
    return fig