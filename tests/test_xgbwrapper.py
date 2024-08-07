import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

from analytix import XGBRegressorWrapper


def test_main():
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    xgb = XGBRegressorWrapper()
    xgb.fit(X, y)

    xgb.beeswarm(X, show=False)
    plt.close()

    xgb.scatter(X,  feature='s5', show=False)
    plt.close()
