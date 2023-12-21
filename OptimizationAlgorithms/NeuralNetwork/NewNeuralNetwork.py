import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_func, activation_derivative_func):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_func = activation_func
        self.activation_derivative_func = activation_derivative_func
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]

    def forward_pass(self, X):
        activations = [X]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            a = self.activation_func(z)
            activations.append(a)
            zs.append(z)

        return activations, zs

    def backward_pass(self, y, activations, zs, loss_derivative_func):
        gradients = {'d_weights': [], 'd_biases': []}
        L = len(self.weights) - 1
        delta = loss_derivative_func(y, activations[-1]) * self.activation_derivative_func(zs[-1])

        gradients['d_weights'].append(np.dot(activations[L].T, delta))
        gradients['d_biases'].append(np.sum(delta, axis=0, keepdims=True))

        for l in reversed(range(L)):
            delta = np.dot(delta, self.weights[l].T) * self.activation_derivative_func(zs[l])
            gradients['d_weights'].insert(0, np.dot(activations[l].T, delta))
            gradients['d_biases'].insert(0, np.sum(delta, axis=0, keepdims=True))

        return gradients

    def train(self, X_train, y_train, optimizer, epochs, loss_derivative_func):
        for epoch in range(epochs):
            activations, zs = self.forward_pass(X_train)
            gradients = optimizer.compute_gradients(self, X_train, y_train, loss_derivative_func)
            optimizer.update_parameters(self, gradients)

    def predict(self, X):
        activations, _ = self.forward_pass(X)
        return activations[-1]

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def save_model(self, file_path):
        model_parameters = {'weights': self.weights, 'biases': self.biases}
        with open(file_path, 'wb') as file:
            pickle.dump(model_parameters, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            model_parameters = pickle.load(file)
        self.set_weights(model_parameters['weights'], model_parameters['biases'])
