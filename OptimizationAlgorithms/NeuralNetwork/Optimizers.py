import numpy as np
from LossFunctions import LossFunctions
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    
    @abstractmethod
    def compute_gradients(self, network, X, y, loss_derivative_func, activation_derivative_func):
        pass

    @abstractmethod
    def update_parameters(self, network, gradients):
        pass

class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self, learning_rate, lambda_val=0.01):
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val  # L2 regularization parameter
    
    def compute_gradients(self, network, X, y, loss_derivative_func, activation_derivative_func):
        activations, zs = network.forward_pass(X)
        gradients = network.backward_pass(y, activations, zs, loss_derivative_func)
        return gradients

    def update_parameters(self, network, gradients):
        for l in range(len(network.weights)):
            network.weights[l] -= self.learning_rate * (gradients['d_weights'][l] + self.lambda_val * network.weights[l])
            network.biases[l] -= self.learning_rate * gradients['d_biases'][l]

class AntColonyOptimizer(BaseOptimizer):
    def __init__(self, num_ants, num_generations, decay_rate, alpha, beta):
        self.num_ants = num_ants
        self.num_generations = num_generations
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta

    def compute_gradients(self, network, X, y, loss_derivative_func, activation_derivative_func):
        # Run the ACO optimization process
        best_ant = self.run_aco(network, X, y)
        return best_ant  # Return the best solution (ant)

    def update_parameters(self, network, best_ant):
        # Update the network's parameters based on the best solution (ant) found by ACO
        if best_ant:
            network.set_weights(best_ant['weights'], best_ant['biases'])

    def run_aco(self, network, X, y):
        # Initialize pheromone trails
        pheromone_matrix = self.initialize_pheromone_matrix(network)

        best_ant = None
        best_loss = np.inf

        for generation in range(self.num_generations):
            # Generate new solutions (ants)
            ants = self.generate_ants(pheromone_matrix, network)

            # Evaluate ants and update the best solution found
            for ant in ants:
                loss = self.evaluate_ant(ant, network, X, y)
                if loss < best_loss:
                    best_ant = ant
                    best_loss = loss

            # Update pheromones based on the performance of ants
            self.update_pheromones(pheromone_matrix, best_ant)

        return best_ant

    def initialize_pheromone_matrix(self, network):
        # Initialize the pheromone matrix with a small amount of pheromone
        pheromone_matrix = []
        for w in network.weights:
            pheromone_matrix.append(np.ones_like(w) * 0.1)
        return pheromone_matrix

    def generate_ants(self, pheromone_matrix, network):
        # Generate new solutions (ants) based on the pheromone matrix
        ants = []
        for _ in range(self.num_ants):
            ant = {
                'weights': [np.random.rand(*w.shape) * pheromone for w, pheromone in zip(network.weights, pheromone_matrix)],
                'biases': [np.random.rand(*b.shape) * pheromone for b, pheromone in zip(network.biases, pheromone_matrix)],
            }
            ants.append(ant)
        return ants

    def evaluate_ant(self, ant, network, X, y):
        # Set the network's weights to those of the ant
        network.set_weights(ant['weights'], ant['biases'])

        # Evaluate the ant (solution) and return the loss
        predictions = network.predict(X)
        loss = LossFunctions.mean_squared_error(y, predictions)  # Assuming mean squared error as the loss function
        return loss

    def update_pheromones(self, pheromone_matrix, best_ant):
        # Update pheromone levels based on the performance of the best ant
        for i, (w, b) in enumerate(zip(best_ant['weights'], best_ant['biases'])):
            pheromone_matrix[i] = pheromone_matrix[i] * (1 - self.decay_rate) + self.alpha * (1 / best_ant['loss'])
