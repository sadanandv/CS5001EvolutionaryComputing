import numpy as np

class ActivationFunctions:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return ActivationFunctions.sigmoid(x) * (1 - ActivationFunctions.sigmoid(x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

    @staticmethod
    def prelu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def prelu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def selu(x, lambda_=1.0507, alpha=1.67326):
        return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def selu_derivative(x, lambda_=1.0507, alpha=1.67326):
        return lambda_ * np.where(x > 0, 1, alpha * np.exp(x))

    @staticmethod
    def softsign(x):
        return x / (1 + np.abs(x))

    @staticmethod
    def softsign_derivative(x):
        return 1 / ((1 + np.abs(x))**2)

    @staticmethod
    def softplus(x):
        return np.log1p(np.exp(x))  # log1p is more accurate for small x

    @staticmethod
    def softplus_derivative(x):
        return ActivationFunctions.sigmoid(x)

    @staticmethod
    def hard_sigmoid(x):
        return np.clip((x + 1) / 2, 0, 1)

    @staticmethod
    def hard_sigmoid_derivative(x):
        return np.where((x > -1) & (x < 1), 0.5, 0)

    @staticmethod
    def swish(x):
        return x * ActivationFunctions.sigmoid(x)

    @staticmethod
    def swish_derivative(x):
        sig_x = ActivationFunctions.sigmoid(x)
        return sig_x + x * sig_x * (1 - sig_x)

    @staticmethod
    def mish(x):
        return x * np.tanh(ActivationFunctions.softplus(x))

    @staticmethod
    def mish_derivative(x):
        sp = ActivationFunctions.softplus(x)
        tanh_sp = np.tanh(sp)
        return tanh_sp + x * (1 - tanh_sp**2) * (1 - np.exp(-sp))
    
    @staticmethod
    def dynamic_theta(x, theta_0=1.0, beta=1.0, gamma=0.0):
        """
        Dynamic threshold function that adapts based on the input magnitude. Used for Newly Defined Function
        x: The input array.
        theta_0: Baseline threshold.
        beta: Controls the adaptation rate.
        gamma: Sets the point where adaptation begins.
        """
        return theta_0 + beta * ActivationFunctions.softplus(x - gamma)

    @staticmethod
    def af_23120023(x, theta_0=1.0, beta=1.0, gamma=0.0, alpha=0.1):
        """
        Custom activation function with a dynamic threshold.
        """
        theta = ActivationFunctions.dynamic_theta(x, theta_0, beta, gamma)
        sigmoid_component = 1 / (1 + np.exp(-theta * x))
        exponential_component = np.exp(alpha * x)
        return sigmoid_component * exponential_component

    @staticmethod
    def af_23120023_derivative(x, theta_0=1.0, beta=1.0, gamma=0.0, alpha=0.1):
        """
        Derivative of the custom activation function with a dynamic threshold.
        """
        theta = ActivationFunctions.dynamic_theta(x, theta_0, beta, gamma)
        sigmoid_component = 1 / (1 + np.exp(-theta * x))
        exponential_component = np.exp(alpha * x)
        derivative_sigmoid = sigmoid_component * (1 - sigmoid_component)
        derivative_exponential = alpha * exponential_component
        # Derivative with respect to x, applying the chain rule for theta(x)
        derivative_theta = beta * ActivationFunctions.softplus_derivative(x - gamma)
        derivative = (derivative_sigmoid * (1 + theta * derivative_theta * x) + sigmoid_component * derivative_exponential)
        return derivative