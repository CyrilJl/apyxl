# -*- coding: utf-8 -*-

# author : Cyril Joly

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ._wrapper import Wrapper


class RandomForestWrapper(Wrapper):
    def __init__(self, scoring, params_space, max_evals, cv, feature_perturbation, verbose):
        super().__init__(scoring=scoring, max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, verbose=verbose)
        if params_space is None:
            params_space = {'n_estimators': hp.randint('n_estimators', 25, 250),
                            'max_depth': hp.randint('max_depth', 3, 15),
                            'min_samples_split': hp.uniform('min_samples_split', 0.1, 1.0),
                            'min_samples_leaf': hp.uniform('min_samples_leaf', 0.1, 0.5),
                            'max_features': hp.uniform('max_features', 0.1, 1)}
        self.params_space = params_space

    def _get_model(self):
        raise NotImplementedError("Subclasses must implement _get_model method")


class RandomForestRegressorWrapper(RandomForestWrapper):
    def __init__(self, scoring='neg_mean_squared_error', params_space=None, max_evals=15, cv=5, feature_perturbation='interventional', verbose=False):
        super().__init__(scoring=scoring, params_space=params_space, max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, verbose=verbose)

    def _get_model(self):
        return RandomForestRegressor()


class RandomForestClassifierWrapper(RandomForestWrapper):
    def __init__(self, scoring='balanced_accuracy', params_space=None, max_evals=15, cv=5, feature_perturbation='interventional', verbose=False):
        super().__init__(scoring=scoring, params_space=params_space, max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, verbose=verbose)

    def _get_model(self):
        return RandomForestClassifier()
