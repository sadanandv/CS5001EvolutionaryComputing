'''    # Initialize swarm with random weights
    # for each generation:
    #     Evaluate fitness of each particle
    #     Update personal and global bests
    #     Update velocity and position of particles
    #     Evaluate new fitness
    # Return the best weights'''

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
nn = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, warm_start=True)
'''
Increase max_iter in MLPClassifier: To allow more training time for the neural network.

Tune PSO Parameters: Experiment with different numbers of particles, different coefficients for the velocity update formula, and more iterations.

Multiple Runs: Due to the stochastic nature of PSO and neural networks, running the optimization multiple times and taking the best result might be beneficial.

Fitness Function: Ensure that the fitness function accurately reflects the performance of the network. You might want to experiment with different evaluation metrics.

Particle Initialization: Experiment with different strategies for initializing the particles to cover a broader range of the search space.

Debugging: Add debugging statements or logs to ensure that the weights are being updated correctly at each step.

'''

nn.fit(X_train, y_train)  # Initial fit to set up the network

# Evaluation function for the neural network
def evaluate_nn(weights):
    # Unpack weights from the flattened array
    hidden_layer_weight_end = 2048  # 64 * 32
    hidden_layer_bias_end = hidden_layer_weight_end + 32
    output_layer_weight_end = hidden_layer_bias_end + 320  # 32 * 10

    nn.coefs_ = [
        weights[:hidden_layer_weight_end].reshape(64, 32),
        weights[hidden_layer_bias_end:output_layer_weight_end].reshape(32, 10)
    ]
    nn.intercepts_ = [
        weights[hidden_layer_weight_end:hidden_layer_bias_end],
        weights[output_layer_weight_end:]
    ]

    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)
    return accuracy_score(y_test, predictions)



# Particle Swarm Optimization Components
class Particle:
    def __init__(self, dimension):
        self.position = np.random.rand(dimension)  # Position vector
        self.velocity = np.random.rand(dimension)  # Velocity vector
        self.best_position = self.position.copy()
        self.best_score = -np.inf

def update_velocity(particle, global_best_position, w=0.5, c1=1, c2=1):
    inertia = w * particle.velocity
    cognitive = c1 * np.random.rand() * (particle.best_position - particle.position)
    social = c2 * np.random.rand() * (global_best_position - particle.position)
    particle.velocity = inertia + cognitive + social

def update_position(particle):
    particle.position += particle.velocity

# PSO Parameters
num_particles = 30
num_dimensions = 2410  # Adjusted total number of weights and biases
num_iterations = 50
particles = [Particle(num_dimensions) for _ in range(num_particles)]
global_best_position = None
global_best_score = -np.inf

# Main PSO Loop
for iteration in range(num_iterations):
    for particle in particles:
        score = evaluate_nn(particle.position)
        
        # Update personal best
        if score > particle.best_score:
            particle.best_score = score
            particle.best_position = particle.position.copy()
        
        # Update global best
        if score > global_best_score:
            global_best_score = score
            global_best_position = particle.position.copy()
    
    for particle in particles:
        update_velocity(particle, global_best_position)
        update_position(particle)

# Final Evaluation with best weights
final_score = evaluate_nn(global_best_position)
print(f"Final Accuracy: {final_score}")
