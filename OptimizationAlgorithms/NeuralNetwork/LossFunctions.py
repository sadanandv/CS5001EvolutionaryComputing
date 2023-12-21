import numpy as np

class LossFunctions:

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -y_true / y_pred

    @staticmethod
    def hinge_loss(y_true, y_pred):
        return np.mean(np.maximum(1 - y_true * y_pred, 0))

    @staticmethod
    def hinge_loss_derivative(y_true, y_pred):
        return np.where(y_true * y_pred < 1, -y_true, 0)

    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def mean_absolute_error_derivative(y_true, y_pred):
        return np.where(y_pred > y_true, 1, -1)