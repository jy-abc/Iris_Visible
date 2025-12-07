"""基础统计分析模块"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def create_boxplots(df):
    """创建箱线图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    titles = ['Sepal Length by Species', 'Sepal Width by Species', 
              'Petal Length by Species', 'Petal Width by Species']
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        row = idx // 2
        col = idx % 2
        sns.boxplot(x='species', y=feature, data=df, ax=axes[row, col])
        axes[row, col].set_title(title)
        axes[row, col].set_xticklabels(['0', '1', '2'])
    
    plt.tight_layout()
    return fig

def create_interactive_scatter(df):
    """创建交互式散点图"""
    figures = []
    
    feature_pairs = [
        ('sepal_length', 'sepal_width', "Sepal Length vs Sepal Width"),
        ('sepal_length', 'petal_length', "Sepal Length vs Petal Length"),
        ('sepal_length', 'petal_width', "Sepal Length vs Petal Width"),
        ('sepal_width', 'petal_length', "Sepal Width vs Petal Length"),
        ('sepal_width', 'petal_width', "Sepal Width vs Petal Width"),
        ('petal_length', 'petal_width', "Petal Length vs Petal Width")
    ]
    
    for x_feat, y_feat, title in feature_pairs:
        fig = px.scatter(df, x=x_feat, y=y_feat, 
                        color='species', 
                        title=title)
        figures.append(fig)
    
    return figures

def run_basic_analysis():
    """运行基础分析"""
    print("正在进行基础统计分析...")
    from data_loader import load_iris_data, preprocess_data
    
    data = load_iris_data()
    df = data['df_seaborn']
    
    # 创建箱线图
    df_clean = preprocess_data(df)
    fig_box = create_boxplots(df_clean)
    plt.show()
    
    # 创建交互式散点图
    figures = create_interactive_scatter(df)
    for fig in figures:
        fig.show()