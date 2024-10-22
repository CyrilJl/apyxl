.. _index:

apyxl documentation
===================


The ``apyxl`` package (Another PYthon package for eXplainable Learning) is a simple wrapper around 
`xgboost <https://xgboost.readthedocs.io/en/stable/python/index.html>`_, `hyperopt 
<https://hyperopt.github.io/hyperopt/>`_, and `shap <https://shap.readthedocs.io/en/latest/>`_.
It provides the user with the ability to build a performant regression or classification model 
and use the power of the SHAP analysis to gain a better understanding of the links the model
builds between its inputs and outputs. With ``apyxl``, processing categorical features, fitting
the model using Bayesian hyperparameter search, and instantiating the associated SHAP explainer
can all be accomplished in a single line of code, streamlining the entire process from data
preparation to model explanation.

The core of this package lies in the classes ``XGBClassifierWrapper`` and ``XGBRegressorWrapper``.
However, ``apyxl`` is not limited to these, as they also feed into the ``TimeSeriesNormalizer``
class, which enables the calculation of complex time series trends in an unsupervised manner.

More broadly, ``apyxl`` shapes my thinking on the connections between explainable machine learning,
econometrics (Difference-In-Differences, Regression Discontinuity Design, Panel Analysis), time
series normalization, and A/B testing.


.. toctree::
   :maxdepth: 1
   :hidden:

   tours/index
   api_reference/index
   future
