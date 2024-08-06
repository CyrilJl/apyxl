# Analytix

A Python package for data analysis and model optimization.

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

## Features

- **XGBoost Wrapper**: A convenient wrapper around the XGBoost regressor.
- **Hyperparameter Optimization**: Utilizes `hyperopt` for optimizing hyperparameters.
- **Explainability**: Uses `shap` for model explainability.

## Requirements

The package requires the following dependencies:

- numpy
- pandas
- xgboost
- scikit-learn
- shap
