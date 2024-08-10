# <img src="https://github.com/CyrilJl/apyxl/blob/main/_static/logo.svg" alt="Logo OptiMask" width="40" height="40"> apyxl

The ``apyxl`` package (Another PYthon package for eXplainable Learning) is a simple wrapper around [`xgboost`](https://xgboost.readthedocs.io/en/stable/python/index.html), [`hyperopt`](https://hyperopt.github.io/hyperopt/), and [`shap`](https://shap.readthedocs.io/en/latest/). It provides the user with the ability to build a performant regression or classification model and use the power of the SHAP analysis to gain a better understanding of the links the model builds between its inputs and outputs.

### Current Features:
- Automatic One-Hot-Encoding for categorical variables
- Basic hyperparameter optimization using `hyperopt` with K-Folds cross-validation
- Simple explainability visualizations using `shap` (`beeswarm`, `decision`, `force`, `scatter`)
- Focus on regression tasks only

### Planned Enhancements:
1. Time-series data handling and normalization
2. A/B test analysis capabilities
3. Support for user-defined scoring functions

## Installation
To install the package, use:
```bash
pip install apyxl
```

## Basic Usage
Here's a simple example of how to use the `XGBRegressorWrapper` class:
```python
from apyxl import XGBRegressorWrapper
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
X, y = load_diabetes(return_X_y=True, as_frame=True)

# Initialize and fit the model
xgb = XGBRegressorWrapper()
xgb.fit(X, y)

# Generate a beeswarm plot
xgb.beeswarm(X)
```
<img src="https://github.com/CyrilJl/apyxl/blob/main/_static/beeswarm.png" width="500">

```python
# Generate a dependence plot
xgb.scatter(X, feature='s5')
```
<img src="https://github.com/CyrilJl/apyxl/blob/main/_static/dependence.png" width="500">

Please note that this package is still under development, and features may change or expand in future versions.
