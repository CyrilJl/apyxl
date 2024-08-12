def check_params(param, params=None, types=None):
    # Check if the parameter's type matches the accepted types
    if (types is not None) and (not isinstance(param, types)):
        if isinstance(types, type):
            accepted = f'{types}'
        else:
            accepted = f"{', '.join([str(t) for t in types])}"
        msg = f"`{param}` is not of an accepted type, it can only be of type {accepted}!"
        raise TypeError(msg)

    # Check if the parameter is among the recognized parameters
    if (params is not None) and (param not in params):
        msg = f"`{param}` is not a recognized argument, it can only be one of {', '.join(sorted(params))}!"
        raise ValueError(msg)

    # Return the parameter if it passes the checks
    return param


class MissingInputError(Exception):
    """Exception raised when both X and shap_values are missing."""


class NotFittedError(Exception):
    """Exception raised when an operation is attempted on an unfitted model."""


class ModelNotImplementedError(NotImplementedError):
    """Exception raised when a subclass doesn't implement the _get_model method."""

    def __init__(self, class_name):
        self.message = f"The _get_model() method must be implemented by {class_name}!"
        super().__init__(self.message)


class FeatureIndexError(ValueError):
    """Exception raised when the feature index is out of bounds."""

    def __init__(self, feature, max_index):
        super().__init__(f"The feature index {feature} is out of bounds. "
                         f"Valid feature indices are from 0 to {max_index}.")


class FeatureNameError(ValueError):
    """Exception raised when the feature name is not found."""

    def __init__(self, feature, feature_names):
        super().__init__(f"The feature name '{feature}' does not exist. "
                         f"Available features are: {', '.join(feature_names)}.")
