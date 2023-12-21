''' # Initialize population with random weights
    # for each generation:
    #     Evaluate fitness for each individual
    #     Select parents
    #     Perform crossover
    #     Perform mutation
    #     Evaluate new fitness
    #     Select the next generation
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

# Genetic Algorithm Components
def initialize_population(pop_size, num_weights):
    return [np.random.uniform(-1, 1, num_weights) for _ in range(pop_size)]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(len(parent1))
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(-1, 1)
    return individual

def select_parents(population, fitnesses, num_parents, tournament_size=3):
    selected_parents = []
    for _ in range(num_parents):
        tournament_inds = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitnesses = fitnesses[tournament_inds]
        winner_index = tournament_inds[np.argmax(tournament_fitnesses)]
        selected_parents.append(population[winner_index])
    return selected_parents


# GA Parameters
population_size = 50
num_generations = 100
mutation_rate = 0.01
num_parents = population_size // 2
num_weights = 2410  # Total number of weights and biases

# Initialize Population
population = initialize_population(population_size, num_weights)

# Main Genetic Algorithm Loop
for generation in range(num_generations):
    # Evaluate Fitness
    fitnesses = np.array([evaluate_nn(individual) for individual in population])

    # Select Parents
    parents = select_parents(population, fitnesses, num_parents)

    # Generate Next Generation
    next_generation = []
    for _ in range(len(population) // 2):
        parent_indices = np.random.choice(len(parents), 2, replace=False)
        parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

    # Update Population
    population = next_generation


# Final Evaluation
best_individual = max(population, key=lambda ind: evaluate_nn(ind))
final_score = evaluate_nn(best_individual)
print(f"Final Accuracy: {final_score}")
