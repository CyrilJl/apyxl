.. _future:

Planned Enhancements
====================

- Improved sample selection when one of the subsampling options (`n` or `frac`) is activated: sampling will consider the output distribution, whether for classification or regression.
- Automatic smoothing of output trends in `apyxl.TimeSeriesNormalizer`.
- Numerical experiments will be conducted to compare the capabilities of `apyxl` with traditional econometric techniques.
- Development of an `apyxl.DiffInDiff` class.
- Model saving functionality.
- Early stopping during the fitting process.
- Automatic labeling of meaningful and non-meaningful SHAP values (e.g., are their absolute values large enough? Considering using an ensemble of xgb models to discard isolated large values).
