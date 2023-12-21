import pickle
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from NewNeuralNetwork import NeuralNetwork
from ActivationFunctions import ActivationFunctions

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_parameters = pickle.load(file)
    return model_parameters

def load_and_preprocess_test_data():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    # Split the data into training and test sets
    _, X_test, _, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Make predictions using the model
    predictions = model.predict(X_test)

    # Evaluate predictions
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    return accuracy

def main():
    # Load test data
    X_test, y_test = load_and_preprocess_test_data()

    # Load the trained model
    model_parameters = load_model('trained_model.pkl')

    # Initialize a neural network with the loaded parameters
    nn = NeuralNetwork(input_size=4, hidden_sizes=[10, 8], output_size=3, activation_func=ActivationFunctions.sigmoid)
    nn.set_weights(model_parameters['weights'], model_parameters['biases'])

    # Evaluate the model
    test_accuracy = evaluate_model(nn, X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
