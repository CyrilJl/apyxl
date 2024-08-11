# <img src="https://raw.githubusercontent.com/CyrilJl/apyxl/main/_static/logo.svg" alt="Logo OptiMask" width="40" height="40"> apyxl

The `apyxl` package (Another PYthon package for eXplainable Learning) is a simple wrapper around [`xgboost`](https://xgboost.readthedocs.io/en/stable/python/index.html), [`hyperopt`](https://hyperopt.github.io/hyperopt/), and [`shap`](https://shap.readthedocs.io/en/latest/). It provides the user with the ability to build a performant regression or classification model and use the power of the SHAP analysis to gain a better understanding of the links the model builds between its inputs and outputs. With `apyxl`, processing categorical features, fitting the model using Bayesian hyperparameter search, and instantiating the associated SHAP explainer can all be accomplished in a single line of code, streamlining the entire process from data preparation to model explanation.

### Current Features:
- Automatic One-Hot-Encoding for categorical variables
- Bayesian hyperparameter optimization using `hyperopt`
- Simple explainability visualizations using `shap` (`beeswarm`, `decision`, `force`, `scatter`)
- Focus on classification and regression tasks

### Planned Enhancements:
- Time-series data handling and normalization
- A/B test analysis capabilities

## Installation
To install the package, use:
```bash
pip install apyxl
```

## Basic Usage

### Regression
```python
from apyxl import XGBRegressorWrapper
from sklearn.datasets import fetch_california_housing

X.shape, y.shape
>>> ((20640, 8), (20640,))

model = XGBRegressorWrapper().fit(X, y)
# defaults to r2 score
model.best_score
>>> 0.6671771984999055

# Plot methods can handle internally the computation of the SHAP values
model.beeswarm(X=X.sample(2_500))
```
<img src="https://raw.githubusercontent.com/CyrilJl/apyxl/main/_static/a.png" width="500">

```python
model.scatter(X=X.sample(2_500), feature='Latitude')
```
<img src="https://raw.githubusercontent.com/CyrilJl/apyxl/main/_static/b.png" width="500">

### Classification
```python
from apyxl import XGBClassifierWrapper
from sklearn.datasets import fetch_covtype

X, y = fetch_covtype(as_frame=True, return_X_y=True)
y -= 1
y.unique()
>>> array([4, 1, 0, 6, 2, 5, 3])

X.shape, y.shape
>>> ((581012, 54), (581012,))

# To speed up the process, Bayesian hyperparameter optimization can be performed on a subset of the dataset.
# The model is then fitted on the entire dataset using the optimized hyperparameters.
model = XGBClassifierWrapper().fit(X, y, n=25_000)
# defaults to Matthews correlation coefficient
model.best_score
>>> 0.5892932365687379

# Computing SHAP values can be resource-intensive, so it's advisable to calculate them once for multiple future
# uses, especially in multiclass classification scenarios where the cost is even higher compared to binary
# classification (shap values shape equals (n_samples, n_features, n_classes))
shap_values = model.compute_shap_values(X.sample(1_000))
shap_values.shape
>>> (1000, 54, 7)
# The `output` argument selects the shap values associated to the desired class
model.beeswarm(shap_values=shap_values, output=2, max_display=15)
```
<img src="https://raw.githubusercontent.com/CyrilJl/apyxl/main/_static/c.png" width="500">

```python
model.scatter(shap_values=shap_values, feature='Elevation', output=4)
```
<img src="https://raw.githubusercontent.com/CyrilJl/apyxl/main/_static/d.png" width="500">


## Note
Please note that this package is still under development, and features may change or expand in future versions.
