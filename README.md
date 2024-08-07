# <img src="https://github.com/CyrilJl/AnalytiX/blob/main/_static/logo.svg" alt="Logo OptiMask" width="40" height="40"> Analytix

The `analytix` package is an emerging tool designed to simplify non-linear, explainable regression for beginner and intermediate users. It serves as a wrapper around `xgboost` for the regression model, `hyperopt` for optimizing hyperparameters, and `shap` for model explainability. 

### Current Features:
- **Categorical Variables**: Automatically processed using One-Hot-Encoding.
- **Hyperparameter Optimization**: Conducts multiple K-Folds cross-validation on the training dataset with `hyperopt` to select the best hyperparameters, minimizing RMSE. Users can manage the number of folds and trials.
- **Explainability**: Supports `beeswarm` and `dependence_plot` visualizations.
- **Regression Models**: Currently focused solely on regression tasks.

### Upcoming Enhancements:
1. **Time-Series Normalization**: Automatic handling and normalization of time-series data to streamline model training.
2. **A/B Test Analysis**: Capabilities to analyze A/B test results, providing users with insights into different experiment groups.
3. **User-Defined Scoring Functions**: Improved handling for custom scoring functions, giving users more flexibility in model evaluation.

### Design Philosophy:
`analytix` is crafted to hide the underlying complexity, making it accessible and user-friendly. It aims to provide a seamless experience for users with varying levels of expertise, focusing on simplicity and ease of use.

By extending its functionalities, `analytix` aspires to become a comprehensive solution for non-linear regression analysis and explainability, catering to both beginners and intermediate users in data science.

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
