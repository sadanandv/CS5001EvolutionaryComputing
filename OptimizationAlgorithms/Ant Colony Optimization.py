'''def ant_colony_optimization(problem, num_ants, generations):
    # Initialize pheromone trails
    # for each generation:
    #     Move ants and construct solutions
    #     Update pheromones
    #     Evaluate and keep track of best solution
    # Return the best solution'''

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

# Evaluation function for the neural network
def evaluate_nn(hyperparameters):
    nn = MLPClassifier(hidden_layer_sizes=(hyperparameters['hidden_neurons'],), 
                       learning_rate_init=hyperparameters['learning_rate'], 
                       max_iter=200)  # Increased max_iter
    nn.fit(X_train, y_train)
    predictions = nn.predict(X_test)
    return accuracy_score(y_test, predictions)


# Ant Colony Optimization Components
def initialize_ants(num_ants):
    ants = []
    for _ in range(num_ants):
        # Randomly initialize hyperparameters
        hyperparameters = {
            'hidden_neurons': np.random.randint(10, 100),
            'learning_rate': np.random.uniform(0.001, 0.1)
        }
        ants.append(hyperparameters)
    return ants

def update_pheromone(pheromone_map, ants, fitnesses):
    # Update pheromone levels based on fitness
    for ant, fitness in zip(ants, fitnesses):
        key = tuple(ant.values())
        if key not in pheromone_map:
            pheromone_map[key] = 0  # Initialize if key doesn't exist
        pheromone_map[key] += fitness  # Increment pheromone level


def choose_hyperparameters(pheromone_map):
    # Choose hyperparameters based on pheromone levels
    total_pheromone = sum(pheromone_map.values())
    probabilities = {k: v / total_pheromone for k, v in pheromone_map.items()}
    hyperparameters = max(probabilities, key=probabilities.get)
    return {'hidden_neurons': hyperparameters[0], 'learning_rate': hyperparameters[1]}

# ACO Parameters
num_ants = 10
num_generations = 20
pheromone_map = {}

# Main ACO Loop
for generation in range(num_generations):
    ants = initialize_ants(num_ants)
    fitnesses = [evaluate_nn(ant) for ant in ants]
    update_pheromone(pheromone_map, ants, fitnesses)
    best_hyperparameters = choose_hyperparameters(pheromone_map)

# Final Evaluation with best hyperparameters
best_nn = MLPClassifier(hidden_layer_sizes=(best_hyperparameters['hidden_neurons'],),
                        learning_rate_init=best_hyperparameters['learning_rate'],
                        max_iter=100)
best_nn.fit(X_train, y_train)
final_predictions = best_nn.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"Final Accuracy: {final_accuracy}")
print(f"Best Hyperparameters: {best_hyperparameters}")