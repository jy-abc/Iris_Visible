"""
Iris数据集可视化分析主入口脚本
运行此脚本可以调用所有可视化功能
"""

import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Iris数据集可视化分析')
    parser.add_argument('--all', action='store_true', help='运行所有可视化')
    parser.add_argument('--basic', action='store_true', help='运行基础统计分析')
    parser.add_argument('--models', action='store_true', help='运行多模型比较')
    parser.add_argument('--three_d_3c', action='store_true', help='运行3D可视化(三分类)')
    parser.add_argument('--surface_2c', action='store_true', help='运行概率曲面可视化(二分类)')
    parser.add_argument('--surface_3c', action='store_true', help='运行概率曲面可视化(三分类)')
    parser.add_argument('--three_d_2c', action='store_true', help='运行3D可视化(二分类)')
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认运行所有
    if not any([args.all, args.basic, args.models, args.three_d_3c, args.surface_2c,args.surface_3c,args.three_d_2c]):
        args.all = True
    
    print("="*60)
    print("Iris数据集可视化分析系统")
    print("="*60)
    
    try:
        # 导入各模块
        from basic_analysis import run_basic_analysis
        from model_comparison import run_model_comparison
        from three_d_visualization_3classes import run_3d_visualization
        from probability_surface_2classes import run_probability_surface
        from probability_surface_3classes import run_probability_surface_3c
        from three_d_visualization_2classes import run_3d_visualization_2c
        
        results = {}
        
        # 运行基础统计分析
        if args.all or args.basic:
            results['basic'] = run_basic_analysis()
        
        # 运行多模型比较
        if args.all or args.models:
            results['models'] = run_model_comparison()
        
        # 运行3D可视化--三分类
        if args.all or args.three_d_3c:
            results['3d_3c'] = run_3d_visualization()
        
        # 运行概率曲面可视化--二分类
        if args.all or args.surface_2c:
            results['surface_2c'] = run_probability_surface()
            
        # 运行概率曲面可视化--三分类
        if args.all or args.surface_3c:
            results['surface_3c'] = run_probability_surface_3c()
            
        # 运行3D可视化--二分类
        if args.all or args.three_d_2c:
            results['3d_2c'] = run_3d_visualization_2c()
        
        print("\n" + "="*60)
        print("所有可视化已完成！")
        print("="*60)
            
    except ImportError as e:
        print(f"导入模块时出错: {e}")
        print("请确保所有模块文件都在同一目录下")
        sys.exit(1)
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()