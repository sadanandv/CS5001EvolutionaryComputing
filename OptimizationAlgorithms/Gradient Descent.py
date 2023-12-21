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

# One-hot encode the labels
y_one_hot = np.zeros((y.size, y.max() + 1))
y_one_hot[np.arange(y.size), y] = 1

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Initialize Neural Network
nn = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1, warm_start=True)

# Training parameters
learning_rate = 0.001
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    nn.fit(X_train, y_train)

    # Directly update weights
    for i in range(len(nn.coefs_)):
        nn.coefs_[i] += np.random.randn(*nn.coefs_[i].shape) * learning_rate
    for i in range(len(nn.intercepts_)):
        nn.intercepts_[i] += np.random.randn(*nn.intercepts_[i].shape) * learning_rate

    # Evaluate the model
    predictions = nn.predict_proba(X_test)
    loss = -np.mean(np.sum(y_test * np.log(predictions + 1e-8), axis=1))  # Cross-entropy loss
    print(f"Epoch {epoch + 1}, Loss: {loss}")

# Final evaluation
final_predictions = nn.predict_proba(X_test)
final_loss = -np.mean(np.sum(y_test * np.log(final_predictions + 1e-8), axis=1))
print(f"Final Loss: {final_loss}")
