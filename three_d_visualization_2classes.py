'''生成3d可视化--二分类'''
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from data_loader import load_iris_data
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def create_3d_visualization_plotly():
    """创建交互式3D可视化 (Plotly)"""
    import plotly.graph_objects as go
    
    iris = load_iris_data()
    X = iris['X'][:, [0, 2, 3]]
    y = np.where(iris['y'] == 0, 0,
                 np.where(iris['y'] == 2, 1, -1))
    
    # 只使用Setosa和Versicolor来训练分类器
    train_mask = y != -1
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 构建网格
    lims = [[X[:, i].min() - 0.3, X[:, i].max() + 0.3] for i in range(3)]
    
    xx, yy, zz = np.meshgrid(
        np.linspace(lims[0][0], lims[0][1], 35),
        np.linspace(lims[1][0], lims[1][1], 35),
        np.linspace(lims[2][0], lims[2][1], 35),
        indexing='ij'
    )
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # 预测所有网格点概率
    probs = model.predict_proba(grid_points)
    p = probs[:, 1]    # Versicolor概率
    p = p.reshape(xx.shape)
    
    # 为所有数据点预测概率
    all_probs = model.predict_proba(X)[:, 1]
    
    # 创建悬停文本
    hover_texts = []
    for i in range(len(X)):
        true_class = iris['target_names'][iris['y'][i]]
        pred_class = "Versicolor-like" if all_probs[i] > 0.5 else "Setosa-like"
        hover_text = (f"Sepal Length: {X[i,0]:.2f} cm<br>"
                      f"Petal Length: {X[i,1]:.2f} cm<br>"
                      f"Petal Width: {X[i,2]:.2f} cm<br>"
                      f"真实类别: {true_class}<br>"
                      f"预测类别: {pred_class}<br>"
                      f"Versicolor概率: {all_probs[i]:.3f}")
        hover_texts.append(hover_text)
    
    # 生成等值面
    verts, faces, _, _ = measure.marching_cubes(p, 0.5, spacing=(
        (lims[0][1] - lims[0][0]) / 34,
        (lims[1][1] - lims[1][0]) / 34,
        (lims[2][1] - lims[2][0]) / 34)
    )
    
    # 平移顶点到真实坐标
    verts[:, 0] += lims[0][0]
    verts[:, 1] += lims[1][0]
    verts[:, 2] += lims[2][0]
    
    # 创建plotly图形
    fig = go.Figure()
    
    # 添加数据点
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=all_probs,
            colorscale='RdBu',
            cmin=0,
            cmax=1,
            opacity=0.8,
            colorbar=dict(
                title="Versicolor Probability",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Setosa (0.0)', '0.25', '0.5', '0.75', 'Versicolor (1.0)'],
                x=1.02,
                xpad=10,
                len=0.7
            ),
            line=dict(width=1, color='white')
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Data Points'
    ))
    
    # 添加决策边界等值面
    fig.add_trace(go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        color='rgb(225, 217, 225)',
        name='Decision Boundary',
        showlegend=True
    ))
    
    # 更新图形布局
    fig.update_layout(
        title=dict(
            text='3D Classification: Setosa vs Versicolor',
            x=0.5,
            font=dict(size=20)
        ),
        scene=dict(
            xaxis_title='Sepal Length (cm)',
            yaxis_title='Petal Length (cm)',
            zaxis_title='Petal Width (cm)',
            xaxis=dict(range=lims[0]),
            yaxis=dict(range=lims[1]),
            zaxis=dict(range=lims[2]),
            bgcolor='rgb(250, 250, 250)'
        ),
        width=1000,
        height=800,
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    fig.show()
    return fig

def create_3d_visualization_matplotlib():
    """创建静态3D可视化 (Matplotlib)"""
    iris = load_iris_data()
    X = iris['X'][:, [0, 2, 3]]
    y=np.where(iris['y']==0,0,
           np.where(iris['y']==2,1,-1))
    
    # 只使用Setosa和Versicolor来训练分类器
    train_mask = y != -1
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    # 模型
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # 网格
    lims = [[X[:, i].min() - 0.3, X[:, i].max() + 0.3] for i in range(3)]
    xx, yy, zz = np.meshgrid(
        *[np.linspace(lims[i][0], lims[i][1], 50) for i in range(3)], 
        indexing='ij'
    )
    grid_points=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    
    # 预测所有网格点概率
    probs = model.predict_proba(grid_points)
    p = probs[:, 1]    # Versicolor概率
    p = p.reshape(xx.shape)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 10),dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    #为所有点预测概率
    all_probs=model.predict_proba(X)[:,1]
    
    for i ,target_class in enumerate([0,1,2]):
        mask=iris['y']==target_class
        labels=['Setosa','Virginica','Versicolor']
        
        scatter=ax.scatter(
            X[mask,0],X[mask,1],X[mask,2],
            c=all_probs[mask],cmap='coolwarm',marker='o',
            s=80,edgecolor='white',linewidth=1.5,
            alpha=0.8,label=labels[i],vmin=0,vmax=1,
        )
    
    #生产等值面
    verts,faces,_,_=measure.marching_cubes(p,0.5,spacing=(
        (lims[0][1]-lims[0][0])/34,
        (lims[1][1]-lims[1][0])/34,
        (lims[2][1]-lims[2][0])/34)
    )
    
    #平移顶点
    verts[:,0]+=lims[0][0]
    verts[:,1]+=lims[1][0]
    verts[:,2]+=lims[2][0]
    
    #绘制决策边界
    mesh=ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],
                        triangles=faces,color='purple',alpha=0.15,
                        linewidth=0.1,edgecolor='purple',shade=True)
    
    # 设置标签
    ax.set_xlabel('Sepal Length\nX1', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_ylabel('Petal Length\nX2', fontsize=14, labelpad=15, fontweight='bold')  
    ax.set_zlabel('Petal Width\nX3', fontsize=14, labelpad=15, fontweight='bold')
    ax.set_title('3D Classification: Setosa vs Versicolor', 
                 fontsize=16, pad=25, fontweight='bold')
    
    #坐标范围
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])
        
    # 美化坐标轴
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter,ax=ax,shrink=0.5,aspect=20,pad=0.1)
    cbar.set_ticks([0,0.25,0.5,0.75,1])
    cbar.set_ticklabels(['0.0\n(Setosa-like)','0.25','0.5','0.75','1.0\n(Versicolor-like)'])
    
    
   #设置视角
    ax.view_init(elev=25,azim=45)
    
    return fig

def run_3d_visualization_2c():
    """运行3D可视化--二分类"""
    print("\n正在进行3D分类可视化--二分类...")
    fig1= create_3d_visualization_matplotlib()
    plt.show()
    
    fig2=create_3d_visualization_plotly()
    go.show()
    
    return fig1,fig2