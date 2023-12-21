'''def grey_wolf_optimization_nn(fitness_func, nn_shape, pack_size, generations):
    # Initialize wolf pack with random weights
    # for each generation:
    #     Evaluate fitness
    #     Update alpha, beta, and delta wolves
    #     Update the position of other wolves
    #     Evaluate new fitness
    # Return the best weights
'''

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize Neural Network
nn = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1, warm_start=True)
nn.fit(X_train, y_train)  # Initial fit to set up the network

# Evaluation function for the neural network
def evaluate_nn(weights):
    nn.coefs_ = [np.array(weights[:2048]).reshape(64, 32), 
                 np.array(weights[2048:2368]).reshape(32, 10)]
    nn.intercepts_ = [np.array(weights[2368:2400]), 
                      np.array(weights[2400:])]
    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)
    return accuracy_score(y_test, predictions)

# Grey Wolf Optimization Components
class Wolf:
    def __init__(self, num_weights):
        self.position = np.random.uniform(-1, 1, num_weights)
        self.score = -np.inf

def update_position(wolf, alpha, beta, delta, a):
    for i in range(len(wolf.position)):
        D_alpha = abs(C * alpha.position[i] - wolf.position[i])
        D_beta = abs(C * beta.position[i] - wolf.position[i])
        D_delta = abs(C * delta.position[i] - wolf.position[i])

        X1 = alpha.position[i] - A * D_alpha
        X2 = beta.position[i] - A * D_beta
        X3 = delta.position[i] - A * D_delta

        wolf.position[i] = (X1 + X2 + X3) / 3

# GWO Parameters
pack_size = 10
num_weights = 2410  # Total number of weights and biases
num_iterations = 50
wolves = [Wolf(num_weights) for _ in range(pack_size)]
alpha, beta, delta = None, None, None

# Main GWO Loop
for iteration in range(num_iterations):
    A = 2 - iteration * (2 / num_iterations)
    C = 2 * np.random.rand()

    for wolf in wolves:
        wolf.score = evaluate_nn(wolf.position)

    wolves.sort(key=lambda wolf: wolf.score, reverse=True)
    alpha, beta, delta = wolves[:3]

    for wolf in wolves:
        update_position(wolf, alpha, beta, delta, A)

# Final Evaluation with best weights
final_score = evaluate_nn(alpha.position)
print(f"Final Accuracy: {final_score}")
