
import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return np.sqrt(mean_squared_error(y_true, y_pred))