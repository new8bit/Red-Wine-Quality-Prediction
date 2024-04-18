from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from data_prepocess import load_training_data
import numpy as np
import time

# Get the data
X_train, y_train = load_training_data()

# Initialize variables
activation_functions = ['relu', 'tanh', 'logistic']
results = {}

# 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True)

# Perform 5-fold cross validation
for activation_function in activation_functions:
    start_time = time.time()

    # Define the model
    model = MLPClassifier(hidden_layer_sizes=(16, 64), activation=activation_function, solver='adam', max_iter=100)

    # Perform cross-validation
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Activation function: {activation_function}")
    print(f"Cross-validation mean accuracy: {cv_results.mean()}")
    print(f"Elapsed time: {elapsed_time} seconds")
    print()

    results[activation_function] = cv_results.mean()

print("Results:", results)