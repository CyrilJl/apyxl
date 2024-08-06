# Analytix

`analytix` is a developing package for simple, non-linear, explainable regression. It can be seen as a wrapper around `xgboost` for the regression model, `hyperopt` for optimizing hyperparameters, and `shap` for explainability.

Categorical variables are automatically processed through One-Hot-Encoding. Multiple K-Folds cross-validation is carried out on the training dataset by `hyperopt` to select the best hyperparameters, minimizing RMSE. The number of folds and trials can be managed by the user.

Currently:

- Only `beeswarm` and `dependence_plot` are available.
- Only regression is available.
- The possibility for the user to change the scoring function is not well-handled yet.

The package is to be extended. I would like it to be able to handle automatic time-series normalization and A/B test analysis.

## Installation

To install the package, use the following command:

```bash
pip install analytix
```

## Usage

Here is an example of how to use the `XGBRegressorWrapper` class from the `analytix` package. This example demonstrates loading the diabetes dataset, fitting the model, and generating a beeswarm plot.

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
xgb.dependence(X, feature='s5')
```

<img src="https://github.com/CyrilJl/AnalytiX/blob/main/_static/dependence.png" width="500">