# -*- coding: utf-8 -*-

# author : Cyril Joly

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hyperopt import Trials, fmin, tpe
from shap import Explanation
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

from ._misc import FeatureIndexError, FeatureNameError, MissingInputError, ModelNotImplementedError, NotFittedError, check_params


class Wrapper:
    def __init__(self, scoring, greater_is_better=True, max_evals=15, cv=5, feature_perturbation='tree_path_dependent',
                 device='cpu', verbose=False, random_state=None):
        """
        Args:
            scoring (str or callable): The scoring metric used for evaluation. A string (see model evaluation documentation)
                                       or a scorer callable object / function with signature scorer(estimator, X, y) which
                                       should return only a single value.
            greater_is_better (bool, optional): Whether a higher score indicates a better model. Default is False.
            max_evals (int, optional): Maximum number of hyperparameter optimization evaluations. Default is 15.
            cv (int, optional): Number of cross-validation folds. Default is 5. Can be any sklearn splitter.
            feature_perturbation (str, optional): The method used for feature perturbation in SHAP values calculation.
                                                  Default is 'tree_path_dependent'.
            verbose (bool, optional): Whether to print verbose output. Default is False.

        Attributes:
            best_model (object): The best model selected after hyperparameter optimization.
            best_params (dict): The best hyperparameters for the selected model.
            features (list): The list of feature names.
            best_score (float): The best score achieved during hyperparameter optimization.
            explainer (shap.TreeExplainer): The SHAP explainer object.
            target_feature (str): The name of the target feature.
        """
        self.set_scoring(scoring=scoring, greater_is_better=greater_is_better)
        self.max_evals = check_params(max_evals, types=int)
        self.cv = cv
        self.feature_perturbation = check_params(feature_perturbation, params=('interventional', 'tree_path_dependent'))
        self.device = check_params(device, params=('cpu', 'cuda', 'gpu'))
        self.verbose = bool(verbose)
        self.rng = np.random.default_rng(seed=random_state)

        self.best_model = None
        self.best_params = None
        self.features = None
        self.best_score = None
        self.target_name = None

    def set_scoring(self, scoring, greater_is_better=False):
        """
        Set the scoring metric.

        Args:
            scoring (str or callable): The scoring metric used for evaluation.
            greater_is_better (bool, optional): Whether a higher score indicates a better model. Default is False.

        Returns:
            self: Returns self for method chaining.
        """
        if scoring == 'mse':
            self.scoring = make_scorer(score_func=mean_squared_error)
            self.greater_is_better = False
        elif scoring == 'matthews':
            self.scoring = make_scorer(score_func=matthews_corrcoef)
            self.greater_is_better = True
        else:
            self.scoring = scoring
            self.greater_is_better = greater_is_better
        return self

    def _get_model(self):
        """
        Private method to be implemented by subclasses to return the model to be optimized.
        """
        raise ModelNotImplementedError(self.__class__.__name__)

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
            scores = cross_val_score(regressor, X.copy(), y.copy(), cv=self.cv, scoring=self.scoring)
            if self.greater_is_better:
                return -np.mean(scores)
            else:
                return np.mean(scores)
        return objective

    def _preprocess_input(self, X: pd.DataFrame):
        """
        Preprocess the input data.

        Args:
            X (pd.DataFrame): The input feature matrix.

        Returns:
            pd.DataFrame: The preprocessed feature matrix.
        """
        bool_columns = X.select_dtypes(include='bool').columns
        X[bool_columns] = X[bool_columns].astype(int)

        object_columns = X.select_dtypes(include='object').columns.tolist()
        if object_columns:
            encoder = OneHotEncoder(sparse_output=False)
            encoded_columns = pd.DataFrame(encoder.fit_transform(X[object_columns].values), index=X.index)
            encoded_columns.columns = encoder.get_feature_names_out(input_features=object_columns).ravel()
            X = pd.concat([X.drop(columns=object_columns), encoded_columns], axis=1)

        if self.features is None:
            self.features = X.columns.tolist()
        else:
            X = X.reindex(columns=self.features)
        return X

    @staticmethod
    def _sample(X, y, frac, n):
        """
        Sample the data based on given fraction or number of samples.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target values.
            frac (float, optional): Fraction of data to sample. Default is None.
            n (int, optional): Number of samples to sample. Default is None.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Sampled feature matrix and target values.
        """
        # TODO: sample with respect to data distribution
        if (frac is None) and (n is None):
            return X, y
        index = np.arange(len(X))
        if (frac is not None) and (0 <= frac <= 1):
            num_samples = int(len(X) * frac)
            index = np.random.choice(len(X), size=num_samples, replace=False)
        elif (1 <= n <= len(X)):
            index = np.random.choice(len(X), size=n, replace=False)
        return X.iloc[index], y.iloc[index]

    def fit(self, X: pd.DataFrame, y=None, target=None, frac=None, n=None, **params):
        """
        Fit the model with optional hyperparameters.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target values. If not provided, the target column should be specified in 'target'.
            target (str, optional): The name of the target column in X. Required if 'y' is not provided.
            frac (float, optional): Fraction of data to use for fitting. Default is None.
            n (int, optional): Number of samples to use for fitting. Default is None.
            **params: Optional hyperparameters to set for the model.

        Returns:
            self: Returns self for method chaining.

        Raises:
            AssertionError: If neither 'y' nor 'target' is provided. 
                            If 'y' is None and 'target' is not found in X's columns.
        """
        check_params(X, types=pd.DataFrame)
        check_params(y, types=(pd.Series, type(None)))
        check_params(target, types=(str, type(None)))

        assert (y is not None) or (target is not None), "'y' or 'target' must be provided."

        if y is None:
            assert target in X.columns, f"Target '{target}' must be a column in X."
            X, y = X.drop(columns=target), X[target]

        self.target_name = y.name
        X = self._preprocess_input(X)
        Xs, ys = self._sample(X=X, y=y, frac=frac, n=n)

        if params:
            self.best_model = self._get_model().set_params(**params)
            self.best_params = params
        else:
            trials = Trials()
            best = fmin(self.create_objective(Xs.copy(), ys.copy()), self.params_space, algo=tpe.suggest,
                        max_evals=self.max_evals, trials=trials, verbose=self.verbose, rstate=self.rng)

            best_params = {key: best[key] for key in self.params_space.keys()}
            self.best_params = best_params

            self.best_model = self._get_model().set_params(**best_params)
            if self.greater_is_better:
                self.best_score = -trials.best_trial['result']['loss']
            else:
                self.best_score = trials.best_trial['result']['loss']
            self.trials = trials
            self.best_trial = best

        self.best_model.fit(X, y)
        if self.feature_perturbation == 'tree_path_dependent':
            self.explainer = shap.TreeExplainer(model=self.best_model, feature_perturbation=self.feature_perturbation,
                                                feature_names=self.features)
        if self.feature_perturbation == 'interventional':
            self.explainer = shap.TreeExplainer(model=self.best_model, data=X, feature_perturbation=self.feature_perturbation,
                                                feature_names=self.features)
        return self

    def predict(self, X) -> pd.Series:
        """
        Make predictions using the fitted model.

        Args:
            X (pd.DataFrame): The feature matrix for prediction.

        Returns:
            pd.Series: Predicted values.
        """
        self._check_fit()
        X = self._preprocess_input(X)
        return pd.Series(self.best_model.predict(X), index=X.index, name=self.target_name)

    def compute_shap_values(self, X) -> shap.Explanation:
        """
        Get SHAP values for a given dataset.

        Args:
            X (pd.DataFrame): The feature matrix.

        Returns:
            shap.Explanation: SHAP values explanation.
        """
        self._check_fit()
        return self.explainer(self._preprocess_input(X))

    def _check_fit(self):
        """
        Check if the model is fitted.

        Raises:
            NotFittedError: If the model is not fitted.
        """
        if self.best_model is None:
            raise NotFittedError("fit() method must be called before using this method")

    def _process_shap_values(self, X, shap_values):
        """
        Process SHAP values or calculate them if not provided.

        Args:
            X (pd.DataFrame): The feature matrix.
            shap_values (shap.Explanation, optional): The SHAP values explanation.

        Returns:
            Tuple[pd.DataFrame, shap.Explanation]: Processed feature matrix and SHAP values explanation.
        """
        if (X is None) and (shap_values is None):
            raise MissingInputError()
        if shap_values is None:
            check_params(X, types=pd.DataFrame)
            self._check_fit()
            shap_values = self.compute_shap_values(X)
        else:
            check_params(shap_values, types=shap.Explanation)
        return X, shap_values

    @staticmethod
    def _check_feature(feature, shap_values):
        check_params(feature, types=(int, str))
        if isinstance(feature, int):
            if feature < 0 or feature >= shap_values.shape[1]:
                raise FeatureIndexError(feature, shap_values.shape[1] - 1)

        if isinstance(feature, str):
            if feature not in shap_values.feature_names:
                raise FeatureNameError(feature, shap_values.feature_names)

    def beeswarm(self, X=None, shap_values=None, max_display=None, order=Explanation.abs.mean(0), output=0, title=None, show=True, **kwargs):
        """
        Create a beeswarm plot of SHAP values.

        Args:
            X (pd.DataFrame, optional): The feature matrix for which SHAP values are calculated. Default is None.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            max_display (int, optional): Maximum number of features to display in the beeswarm plot. Default is None.
            order (callable, optional): Function to order the features. Default is shap.Explanation.abs.
            output (int, optional): The output class for which to plot SHAP values (useful for multiclass classification). Default is 0.
            title (str, optional): Title for the plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for the SHAP beeswarm plot.
        """
        _, shap_values = self._process_shap_values(X, shap_values)
        ndim = len(shap_values.shape)
        if ndim > 2:
            shap_values = shap_values[:, :, output]
        shap.plots.beeswarm(shap_values=shap_values, order=order, max_display=max_display, show=False, **kwargs)
        if ndim > 2:
            plt.xlabel(f"SHAP value (impact on model output {output})")
        if isinstance(title, str):
            plt.title(title)
        if show:
            plt.show()

    def scatter(self, X=None, shap_values=None, feature=0, interaction_feature='auto', output=0, title=None, show=True, **kwargs):
        """
        Create a dependence plot for a specific feature.

        Args:
            X (pd.DataFrame): The feature matrix.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            feature (int or str, optional): Index or name of the feature to create the dependence plot for. Default is 0.
            interaction_feature (int, str, or 'auto', optional): The feature to color the plot by. If 'auto', the method will automatically
                select a feature to color the plot based on interactions with the chosen 'feature'. If an integer or string is provided, 
                it specifies the index or name of the interaction feature. Default is 'auto'.
            output (int, optional): The output class for which to plot SHAP values (useful for multiclass classification). Default is 0.
            title (str, optional): Title for the plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for the SHAP scatter plot.

        Notes:
            - If `interaction_feature='auto'`, the plot will automatically determine the interaction feature to use for coloring.
            - If `interaction_feature` is specified (as an index or name), that feature will be used for coloring the scatter plot.
        """
        _, shap_values = self._process_shap_values(X, shap_values)
        ndim = len(shap_values.shape)
        if ndim > 2:
            shap_values = shap_values[:, :, output]
        self._check_feature(feature=feature, shap_values=shap_values)
        if interaction_feature == 'auto':
            color = shap_values
        else:
            color = shap_values[:, interaction_feature]

        shap.plots.scatter(shap_values=shap_values[:, feature], color=color, show=False, **kwargs)

        if ndim > 2:
            plt.ylabel(f"SHAP value (impact on model output {output})")
        if isinstance(title, str):
            plt.title(title)
        if show:
            plt.show()

    def bar(self, X=None, shap_values=None, max_display=10, order=Explanation.abs, output=0, title=None, show=True, **kwargs):
        """
        Create a bar plot of SHAP values.

        Args:
            X (pd.DataFrame, optional): The feature matrix. Default is None.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            max_display (int, optional): Maximum number of features to display in the bar plot. Default is 10.
            order (callable, optional): Function to order the features. Default is shap.Explanation.abs.
            output (int, optional): The output class for which to plot SHAP values (useful for multiclass classification). Default is 0.
            title (str, optional): Title for the plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for the SHAP bar plot.
        """
        _, shap_values = self._process_shap_values(X, shap_values)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, output]
        shap.plots.bar(shap_values=shap_values, order=order,  max_display=max_display, show=False, **kwargs)
        if isinstance(title, str):
            plt.title(title)
        if show:
            plt.show()

    def decision(self, X=None, shap_values=None, output=0, title=None, show=True, **kwargs):
        """
        Create a decision plot of SHAP values.

        Args:
            X (pd.DataFrame, optional): The feature matrix. Default is None.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            output (int, optional): The output class for which to plot SHAP values (useful for multiclass classification). Default is 0.
            title (str, optional): Title for the plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for the SHAP decision plot.
        """
        X, shap_values = self._process_shap_values(X, shap_values)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, output]
        shap.plots.decision(base_value=self.explainer.expected_value, shap_values=shap_values.values, features=X, show=False, **kwargs)
        if isinstance(title, str):
            plt.title(title)
        if show:
            plt.show()

    def force(self, X=None, shap_values=None, output=0, title=None, show=True, **kwargs):
        """
        Create a force plot of SHAP values.

        Args:
            X (pd.DataFrame, optional): The feature matrix. Default is None.
            shap_values (shap.Explanation, optional): Precomputed SHAP values explanation. Default is None.
            output (int, optional): The output class for which to plot SHAP values (useful for multiclass classification). Default is 0.
            title (str, optional): Title for the plot. Default is None.
            show (bool, optional): Whether to display the plot. Default is True.
            **kwargs: Additional keyword arguments for the SHAP force plot.
        """
        _, shap_values = self._process_shap_values(X, shap_values)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, output]
        shap.plots.force(base_value=shap_values, features=X, matplotlib=True, show=False, **kwargs)
        if isinstance(title, str):
            plt.title(title)
        if show:
            plt.show()
