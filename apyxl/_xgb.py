# -*- coding: utf-8 -*-

# author : Cyril Joly

from hyperopt import hp
from xgboost import XGBClassifier, XGBRegressor

from ._wrapper import Wrapper


class XGBWrapper(Wrapper):
    def __init__(self, scoring, greater_is_better, params_space, max_evals, cv, feature_perturbation, device, verbose, random_state):
        super().__init__(scoring=scoring, greater_is_better=greater_is_better, max_evals=max_evals, cv=cv,
                         feature_perturbation=feature_perturbation, device=device, verbose=verbose, random_state=random_state)
        if params_space is None:
            params_space = {
                'learning_rate': hp.loguniform('learning_rate', -4, -0.5),
                'n_estimators': hp.randint('n_estimators', 100, 1500),
                'max_depth': hp.randint('max_depth', 2, 15),
                'min_child_weight': hp.randint('min_child_weight', 1, 10),
                'subsample': hp.uniform('subsample', 0.5, 1.0),
                'gamma': hp.uniform('gamma', 0, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
                'reg_alpha': hp.uniform('reg_alpha', 0, 100),
                'reg_lambda': hp.uniform('reg_lambda', 1, 500)
            }
        self.params_space = params_space

    def _get_model(self):
        raise NotImplementedError("Subclasses must implement _get_model() method.")


class XGBRegressorWrapper(XGBWrapper):
    def __init__(self, scoring='r2', greater_is_better=True, params_space=None, max_evals=15, cv=5, feature_perturbation='tree_path_dependent', device='cpu', verbose=False, random_state=None):
        super().__init__(scoring=scoring, greater_is_better=greater_is_better, params_space=params_space,
                         max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, device=device, verbose=verbose, random_state=random_state)

    def _get_model(self):
        return XGBRegressor(device=self.device)

    def __repr__(self):
        attrs = [f"scoring='{self.scoring}'",
                 f"greater_is_better={self.greater_is_better}",
                 f"max_evals={self.max_evals}",
                 f"cv={self.cv}",
                 f"feature_perturbation='{self.feature_perturbation}'"]
        if self.verbose:
            attrs.append("verbose=True")
        return f"XGBRegressorWrapper({', '.join(attrs)})"


class XGBClassifierWrapper(XGBWrapper):
    def __init__(self, scoring='matthews', greater_is_better=True, params_space=None, max_evals=15, cv=5, feature_perturbation='tree_path_dependent', device='cpu', verbose=False, random_state=None):
        super().__init__(scoring=scoring, greater_is_better=greater_is_better, params_space=params_space,
                         max_evals=max_evals, cv=cv, feature_perturbation=feature_perturbation, device=device, verbose=verbose, random_state=random_state)

    def _get_model(self):
        return XGBClassifier(device=self.device)

    def __repr__(self):
        attrs = [f"scoring='{self.scoring}'",
                 f"greater_is_better={self.greater_is_better}",
                 f"max_evals={self.max_evals}",
                 f"cv={self.cv}",
                 f"feature_perturbation='{self.feature_perturbation}'"]
        if self.verbose:
            attrs.append("verbose=True")
        return f"XGBClassifierWrapper({', '.join(attrs)})"
