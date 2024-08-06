# -*- coding: utf-8 -*-

# author : Cyril Joly

import numpy as np
import pandas as pd
import shap
from hyperopt import Trials, fmin, tpe
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


class Wrapper:
    def __init__(self, scoring, max_evals=15, cv=5, feature_perturbation='interventional', verbose=False):
        """
        Initialize the Wrapper class.

        Args:
            scoring (str): The scoring metric used for evaluation. A str (see model evaluation documentation) or a scorer callable
            object / function with signature scorer(estimator, X, y) which should return only a single value.
            max_evals (int, optional): Maximum number of hyperparameter optimization evaluations. Default is 15.
            cv (int, optional): Number of cross-validation folds. Default is 5.
            feature_perturbation (str, optional): The method used for feature perturbation in SHAP values calculation. Default is
            'interventional'.
            verbose (bool, optional): Whether to print verbose output. Default is False.

        Attributes:
            best_model (object): The best model selected after hyperparameter optimization.
            best_params (dict): The best hyperparameters for the selected model.
            features (list): The list of feature names.
            best_score (float): The best score achieved during hyperparameter optimization.
        """
        self.scoring = scoring
        self.max_evals = max_evals
        self.cv = cv
        self.feature_perturbation = feature_perturbation
        self.verbose = bool(verbose)

        self.best_model = None
        self.best_params = None
        self.features = None
        self.best_score = None

    def _get_model(self):
        """
        Private method to be implemented by subclasses to return the model to be optimized.
        """
        raise ValueError("Must be implemented by subclasses !")

    def create_objective(self, X, y):
        """
        Create an objective function for hyperparameter optimization.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (array-like): The target values.

        Returns:
            callable: Objective function for hyperparameter optimization.
        """
        def objective(params):
            regressor = self._get_model().set_params(**params)
            scores = cross_val_score(regressor, X, y, cv=self.cv, scoring=self.scoring)
            return -np.mean(scores)
        return objective

    def _preprocess_input(self, X):
        """
        Preprocess the input data.

        Args:
            X (pd.DataFrame): The input feature matrix.

        Returns:
            pd.DataFrame: The preprocessed feature matrix.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        object_columns = X.select_dtypes(include='object').columns.tolist()
        if object_columns:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_columns = pd.DataFrame(encoder.fit_transform(X[object_columns].values), index=X.index)
            encoded_columns.columns = encoder.get_feature_names_out().ravel()
            X = pd.concat([X.drop(columns=object_columns), encoded_columns], axis=1)

        if self.features is None:
            self.features = X.columns.tolist()
        else:
            X = X.reindex(columns=self.features)
        return X

    @staticmethod
    def _sample(X, y, frac, n):
        if (frac is None) and (n is None):
            return X, y
        index = np.arange(len(X))
        if (frac is not None) and (0 <= frac <= 1):
            num_samples = int(len(X) * frac)
            index = np.random.choice(len(X), size=num_samples, replace=False)
        elif (1 <= n <= len(X)):
            index = np.random.choice(len(X), size=n, replace=False)
        return X.iloc[index], y.iloc[index]

    def fit(self, X, y, frac=None, n=None, **params):
        """
        Fit the model with optional hyperparameters.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (array-like): The target values.
            **params: Optional hyperparameters to set for the model.

        Returns:
            self: Returns self for method chaining.
        """
        X = self._preprocess_input(X)
        X, y = self._sample(X=X, y=y, frac=frac, n=n)

        if params:
            self.best_model = self._get_model().set_params(**params)
            self.best_params = params
        else:
            trials = Trials()
            best = fmin(self.create_objective(X, y), self.params_space, algo=tpe.suggest,
                        max_evals=self.max_evals, trials=trials, verbose=self.verbose)

            best_params = {key: best[key] for key in self.params_space.keys()}
            self.best_params = best_params

            self.best_model = self._get_model().set_params(**best_params)
            self.best_score = trials.best_trial['result']['loss']
            self.trials = trials
            self.best_trial = best

        self.best_model.fit(X, y)
        self.explainer = shap.Explainer(self.best_model, feature_perturbation=self.feature_perturbation, feature_names=self.features)
        return self

    def predict(self, X):
        """
        Make predictions using the fitted model.

        Args:
            X (pd.DataFrame): The feature matrix for prediction.

        Returns:
            pd.Series: Predicted values.
        """
        if self.best_model is None:
            raise ValueError("fit() method must be called before predict()")
        X = self._preprocess_input(X)
        return pd.Series(self.best_model.predict(X), index=X.index)

    def compute_shap_values(self, X) -> shap.Explanation:
        """
        Get SHAP values for a given dataset.

        Args:
            X (pd.DataFrame): The feature matrix.

        Returns:
            shap.Explanation: SHAP values explanation.
        """
        return self.explainer(X)

    def _check_shap_values(self, shap_values):
        """
        Check if the input is a valid SHAP values explanation.

        Args:
            shap_values (shap.Explanation): The SHAP values explanation.

        Raises:
            ValueError: If shap_values is not an instance of shap.Explanation.
        """
        if not isinstance(shap_values, shap.Explanation):
            raise ValueError("shap_values must be an instance of shap.Explanation")

    def _check_fit(self):
        """
        Check if the model is fitted.

        Raises:
            ValueError: If the model is not fitted.
        """
        if self.best_model is None:
            raise ValueError("fit() method must be called before using this method")

    def _process_shap_values(self, X, shap_values):
        """
        Process SHAP values or calculate them if not provided.

        Args:
            X (pd.DataFrame): The feature matrix.
            shap_values (shap.Explanation, optional): The SHAP values explanation.

        Returns:
            Tuple[pd.DataFrame, shap.Explanation]: Processed feature matrix and SHAP values explanation.
        """
        if shap_values is None:
            self._check_fit()
            X = self._preprocess_input(X)
            shap_values = self.compute_shap_values(X)
        else:
            self._check_shap_values(shap_values)
        return X, shap_values

    def beeswarm(self, X=None, shap_values=None, max_display=None, show=True):
        """
        Create a beeswarm plot of SHAP values.

        Args:
            X (pd.DataFrame, optional): The feature matrix for which SHAP values are calculated. Default is None.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            max_display (int, optional): Maximum number of features to display in the beeswarm plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
        """
        _, shap_values = self._process_shap_values(X, shap_values)
        shap.summary_plot(shap_values, max_display=max_display, show=show)

    def dependence(self, X, shap_values=None, feature=0, show=True):
        """
        Create a dependence plot for a specific feature.

        Args:
            X (pd.DataFrame): The feature matrix.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            feature (int, optional): Index of the feature to create the dependence plot for. Default is 0.
            show (bool, optional): Whether to display the plot. Default is True.
        """
        X, shap_values = self._process_shap_values(X, shap_values)
        shap.dependence_plot(ind=feature, shap_values=shap_values.values, features=X, show=show)

    def cohorts(self, feature, X, shap_values=None):
        _, shap_values = self._process_shap_values(X, shap_values)
        cat = X[feature].astype(str).values
        shap.plots.bar(shap_values.cohorts(cat).abs.mean(0))
