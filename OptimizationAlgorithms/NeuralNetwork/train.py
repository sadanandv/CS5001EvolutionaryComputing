import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import datasets
from NewNeuralNetwork import NeuralNetwork
from ActivationFunctions import ActivationFunctions
from LossFunctions import LossFunctions
from Optimizers import GradientDescentOptimizer, AntColonyOptimizer

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def get_activation_function(name):
    activation_functions = {
        'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
        'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
        'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
        # Add other activation functions here
    }
    return activation_functions.get(name, (None, None))

def get_loss_function(name):
    loss_functions = {
        'mean_squared_error': (LossFunctions.mean_squared_error, LossFunctions.mean_squared_error_derivative),
        # Add other loss functions here
    }
    return loss_functions.get(name, (None, None))

def calculate_accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    actual = np.argmax(y_true, axis=1)
    return np.mean(predictions == actual)

def main():
    config = load_config(r"C:\Users\sadur\Desktop\Pipeline\config.json")

    # Network configuration
    input_size = config['network']['input_size']
    hidden_layers = config['network']['hidden_layers']
    output_size = config['network']['output_size']
    activation_function_name = config['network']['activation_function']
    activation_function, activation_derivative = get_activation_function(activation_function_name)

    # Initialize the neural network with both activation function and its derivative
    nn = NeuralNetwork(input_size, hidden_layers, output_size, activation_function, activation_derivative)

    # Training configuration
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    lambda_val = config['training']['lambda_val']
    loss_function_name = config['training']['loss_function']
    epochs = config['training']['epochs']
    loss_func, loss_derivative_func = get_loss_function(loss_function_name)

    # Initialize the optimizer
    if optimizer_name == 'GradientDescent':
        optimizer = GradientDescentOptimizer(learning_rate, lambda_val)
    elif optimizer_name == 'AntColonyOptimizer':
        optimizer = AntColonyOptimizer(num_ants=50, num_generations=100, decay_rate=0.1, alpha=1.0, beta=0.5)
    # Add other optimizers as necessary

    # Load and preprocess the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

    # Training loop
    for epoch in range(epochs):
        gradients = optimizer.compute_gradients(nn, X_train, y_train, loss_derivative_func, activation_derivative)
        optimizer.update_parameters(nn, gradients)

        # Validation step
        val_predictions = nn.predict(X_val)
        val_accuracy = calculate_accuracy(y_val, val_predictions)
        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    model_parameters = {'weights': nn.weights, 'biases': nn.biases}
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_parameters, f)

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
