import sys
sys.path.append("..")
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_prepocess import load_training_data
import numpy as np
import time

# Get the data
X_train, y_train = load_training_data()

# Standardize features instead of just normalizing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize variables
hidden_layer_sizes = [(128, 128, 128), (256, 256, 256), (512, 512)]  # 更大的网络结构
alpha_values = [0.00001, 0.0001]  # 更小的正则化强度
learning_rate_init_values = [0.001, 0.005, 0.01]  # 更细致的学习率探索
max_iter_values = [500, 1000]  # 增加最大迭代次数
results = {}

# Define 5-fold cross validation test harness
kfold = KFold(n_splits=5, shuffle=True)

# Perform 5-fold cross validation
for hidden_layer_size in hidden_layer_sizes:
    for alpha in alpha_values:
        for learning_rate_init in learning_rate_init_values:
            for max_iter in max_iter_values:
                start_time = time.time()

                # Define the model
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation='relu', solver='adam',
                                      max_iter=max_iter, alpha=alpha, learning_rate_init=learning_rate_init)

                # Perform cross-validation
                cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

                end_time = time.time()
                elapsed_time = end_time - start_time
                config = (hidden_layer_size, alpha, learning_rate_init, max_iter)
                print(f"Config: {config}")
                print(f"Cross-validation mean accuracy: {cv_results.mean()}")
                print(f"Elapsed time: {elapsed_time} seconds")
                print()

                results[config] = cv_results.mean()

print("Results:", results)