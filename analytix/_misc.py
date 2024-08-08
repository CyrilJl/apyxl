def check_params(param, params=None, types=None):
    # Check if the parameter's type matches the accepted types
    if (types is not None) and (not isinstance(param, types)):
        if isinstance(types, type):
            accepted = f'{types}'
        else:
            accepted = f"{', '.join([str(t) for t in types])}"
        # Raise a TypeError with a customized message
        msg = f"`{param}` is not of an accepted type, it can only be of type {accepted}!"
        raise TypeError(msg)

    # Check if the parameter is among the recognized parameters
    if (params is not None) and (param not in params):
        # Raise a ValueError with a customized message
        msg = f"`{param}` is not a recognized argument, it can only be one of {', '.join(sorted(params))}!"
        raise ValueError(msg)

    # Return the parameter if it passes the checks
    return param


class MissingInputError(Exception):
    """Exception raised when both X and shap_values are missing."""


class NotFittedError(Exception):
    """Exception raised when an operation is attempted on an unfitted model."""
