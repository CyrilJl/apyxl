import matplotlib.pyplot as plt
import pandas as pd
import pytest
import shap
from sklearn.datasets import load_diabetes

from apyxl import XGBRegressorWrapper
from apyxl._misc import MissingInputError, NotFittedError


@pytest.fixture
def diabetes_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X, y


def test_xgb_wrapper_init():
    xgb = XGBRegressorWrapper()
    assert xgb.best_model is None
    assert xgb.best_params is None
    assert xgb.features is None
    assert xgb.best_score is None
    assert xgb.target_name is None


def test_xgb_wrapper_fit(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    assert xgb.best_model is not None
    assert xgb.best_params is not None
    assert xgb.features is not None
    assert xgb.best_score is not None
    assert xgb.target_name is not None


def test_xgb_wrapper_predict(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    y_pred = xgb.predict(X)
    assert isinstance(y_pred, pd.Series)
    assert y_pred.name == xgb.target_name


def test_xgb_wrapper_compute_shap_values(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    shap_values = xgb.compute_shap_values(X)
    assert isinstance(shap_values, shap.Explanation)


def test_xgb_wrapper_beeswarm(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    xgb.beeswarm(X, show=False)
    plt.close()


def test_xgb_wrapper_scatter(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    xgb.scatter(X, feature='s5', show=False)
    xgb.scatter(X, feature='s5', interaction_feature='sex', show=False)


def test_xgb_wrapper_bar(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    xgb.bar(X, show=False)


def test_xgb_wrapper_decision(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    xgb.decision(X, show=False)
    plt.close()


def test_xgb_wrapper_force(diabetes_data):
    X, y = diabetes_data
    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)
    xgb.force(X.iloc[[0]], show=False)


def test_xgb_wrapper_not_fitted():
    xgb = XGBRegressorWrapper()
    with pytest.raises(NotFittedError):
        xgb.predict(pd.DataFrame({'a': [1, 2, 3]}))


def test_xgb_wrapper_missing_input():
    xgb = XGBRegressorWrapper()
    with pytest.raises(MissingInputError):
        xgb.beeswarm()
        xgb.scatter()
        xgb.bar()
        xgb.decision()
        xgb.force()
