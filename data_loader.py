"""数据加载模块"""
import seaborn as sns
from sklearn.datasets import load_iris

def load_iris_data():
    """加载Iris数据集"""
    # 从seaborn加载（用于统计分析）
    df_seaborn = sns.load_dataset('iris')
    
    # 从sklearn加载（用于机器学习）
    iris_sklearn = load_iris()
    X = iris_sklearn.data
    y = iris_sklearn.target
    feature_names = iris_sklearn.feature_names
    target_names = iris_sklearn.target_names
    
    return {
        'df_seaborn': df_seaborn,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'target_names': target_names,
        'data_sklearn': iris_sklearn
    }

def preprocess_data(df):
    """数据预处理"""
    df_clean = df.dropna()
    df_clean['species'] = df_clean['species'].astype('category').cat.codes

    return df_clean
