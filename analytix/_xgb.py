# -*- coding: utf-8 -*-

# author : Cyril Joly

from hyperopt import hp
from xgboost import XGBRegressor

from ._wrapper import Wrapper


class XGBWrapper(Wrapper):
    def __init__(self, scoring, params_space, max_evals, cv, feature_perturbation, verbose):
        super().__init__(scoring=scoring, max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, verbose=verbose)
        if params_space is None:
            params_space = {
                'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                'n_estimators': hp.randint('n_estimators', 100, 1000),
                'max_depth': hp.randint('max_depth', 3, 10),
                'min_child_weight': hp.randint('min_child_weight', 1, 10),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'gamma': hp.uniform('gamma', 0, 0.5),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                'reg_alpha': hp.uniform('reg_alpha', 0, 50),
                'reg_lambda': hp.uniform('reg_lambda', 10, 100)
            }
        self.params_space = params_space

    def _get_model(self):
        raise NotImplementedError("Subclasses must implement _get_model() method.")


class XGBRegressorWrapper(XGBWrapper):
    def __init__(self, scoring='neg_mean_squared_error', params_space=None, max_evals=15, cv=5, feature_perturbation='interventional', verbose=False):
        super().__init__(scoring=scoring, params_space=params_space, max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, verbose=verbose)

    def _get_model(self):
        return XGBRegressor()
