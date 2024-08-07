# <img src="https://github.com/CyrilJl/AnalytiX/blob/main/_static/logo.svg" alt="Logo OptiMask" width="40" height="40"> Analytix

The `analytix` package is a simple wrapper around `xgboost`, `hyperopt`, and `shap`, aimed at making non-linear, explainable regression more accessible for beginner and intermediate users. This project is in its early stages of development and currently offers basic functionality.

### Current Features:
- Automatic One-Hot-Encoding for categorical variables
- Basic hyperparameter optimization using `hyperopt` with K-Folds cross-validation
- Simple explainability visualizations using `shap` (`beeswarm` and `dependence_plot`)
- Focus on regression tasks only

### Planned Enhancements:
1. Time-series data handling and normalization
2. A/B test analysis capabilities
3. Support for user-defined scoring functions

## Installation

To install the package, use:

```bash
pip install analytix
```

## Basic Usage

Here's a simple example of how to use the `XGBRegressorWrapper` class:

```python
from analytix import XGBRegressorWrapper
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Initialize and fit the model
xgb = XGBRegressorWrapper()
xgb.fit(X, y)

# Generate a beeswarm plot
xgb.beeswarm(X)
```

<img src="https://github.com/CyrilJl/AnalytiX/blob/main/_static/beeswarm.png" width="500">

```python
# Generate a dependence plot
xgb.scatter(X, feature='s5')
```

<img src="https://github.com/CyrilJl/AnalytiX/blob/main/_static/dependence.png" width="500">

Please note that this package is still under development, and features may change or expand in future versions.