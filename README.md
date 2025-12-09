[README.md](https://github.com/user-attachments/files/24056052/README.md)
# Iris_Visible

## Project Overview

Iris_Visible is a comprehensive visualization toolkit for the classic Iris dataset, providing various advanced visualization methods including 3D probability maps, multi-model decision boundary comparisons, interactive classification visualizations, and more. 



## Key Features

- **Multi-Model Comparison**: Compare decision boundaries of 6 different classification models
- **3D Visualization**: Provide both binary and multi-class 3D interactive visualizations
- **Probability Surfaces**: Generate 3D probability distribution maps to visually display classification probabilities
- **Diverse Technology Stack**: Combine Matplotlib and Plotly for both static and interactive visualizations
- **Modular Design**: Each feature is an independent module, making it easy to extend and maintain



## Project Structure

```tex
iris-visualization-project/
├── main.py                      # Main entry script
├── data_loader.py              # Data loading module
├── basic_analysis.py           # Basic statistical analysis
├── model_comparison.py         # Multi-model comparison
├── three_d_visualization_2classes.py  # Binary classification 3D visualization
├── three_d_visualization_3classes.py  # Multi-class 3D visualization
├── probability_surface_2classes.py    # Binary classification probability map
├── probability_surface_3classes.py    # Multi-class probability map
└── README.md                   # Project documentation
```



## Installation and Usage

- Dependencies Installation

  ```bash
  pip install numpy
  pip install matplotlib
  pip install plotly
  pip install scikit-learn
  pip install seaborn
  pip install scikit-image
  ```

  Or install all dependencies at once using Aliyun mirror
  ```bash
  pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy matplotlib scikit-learn plotly seaborn scikit-image
  ```

  

- Running the Project

  ```bash
  cd Iris_Visible
  python main.py
  ```

  Or running specific modules

  ```bash
  #Run only basic statistical analysis:
  python main.py --basic
  
  #Run only multi-model comparison:
  python main.py --models
  
  #Run only multi-class 3D boundary visualization:
  python main.py --three_d_3c
  
  #Run only binary classification 3D boundary visualization:
  python main.py --three_d_3c
  
  #Run only multi-class 3D probability map:
  python main.py --surface_3c
  
  #Run only binary classification 3D probability map:
  python main.py --surface_2c
  ```

  View Help

  ```python
  python main.py --help
  ```

  
