from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def split_data(file_path='data/data_raw.csv', test_size=0.3, random_state=522117, delimiter=';'):
    """
    Split the data into training set and test set.

    Parameters:
    file_path: str, path to the data file
    test_size: float, proportion of the dataset to include in the test split
    random_state: int, the seed used by the random number generator
    delimiter: str, the delimiter used in the data file

    Returns:
    train_data: pandas.DataFrame
    test_data: pandas.DataFrame
    """
    # Manually specifying column names due to parsing issues
    column_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                    "pH", "sulphates", "alcohol", "quality"]

    # Load the data with the correct settings
    data = pd.read_csv(file_path, delimiter=delimiter, names=column_names, header=0)

    # Splitting the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data

def save_split_data(train_data, test_data, train_file='data/train.csv', test_file='data/test.csv'):
    """save the split data to csv files

    Parameters:
    train_data: pandas.DataFrame
    test_data: pandas.DataFrame
    train_file: str, path to save the training data
    test_file: str, path to save the test data
    """
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
def load_training_data(train_file='data/train.csv', target='quality'):
    """
    Load training data and split it into features and target.

    Parameters:
    train_file: str, path to the training data file
    target: str, the name of the target variable

    Returns:
    X: pandas.DataFrame, the features
    y: pandas.Series, the target variable
    """
    data = pd.read_csv(train_file)
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def load_test_data(test_file='data/test.csv', target='quality'):
    """
    Load test data and return it as features.

    Parameters:
    test_file: str, path to the test data file
    target: str, the name of the target variable

    Returns:
    X: pandas.DataFrame, the features
    y: pandas.Series, the target variable
    """
    data = pd.read_csv(test_file)
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def normalize_features(X):
    """
    Normalize all features

    Parameters:
    X: pandas.DataFrame, the features to normalize

    Returns:
    X_normalized: pandas.DataFrame, the normalized features
    """
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_normalized

def apply_pca(data, n_components):
    """
    Apply PCA to the data and keep a fixed number of features.

    Parameters:
    data: pandas.DataFrame, the data to be transformed
    n_components: int, the number of features to keep

    Returns:
    transformed_data: pandas.DataFrame, the transformed data
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)

    return pd.DataFrame(transformed_data)

def select_features_by_importance(X, y, n_features):
    """
    Select features based on their importance determined by a random forest.

    Parameters:
    X: pandas.DataFrame, the features
    y: pandas.Series, the target variable
    n_features: int, the number of features to keep

    Returns:
    X_selected: pandas.DataFrame, the selected features
    """
    # Fit a random forest to the data
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)

    # Get the importance of each feature
    importances = forest.feature_importances_

    # Sort the features by importance and select the top n_features
    indices = np.argsort(importances)[::-1]
    selected_features = X.columns[indices[:n_features]]

    # Return the selected features
    return X[selected_features]

def remove_outliers(df, feature_names):
    '''
    For each of the features from the dataset, remove the outliers.

    Parameters:
    df: DataFrame
    feature_names: feature names

    Returns:
    cleaned_df: cleaned dataframe
    '''
    clean_df = df.copy()
    for feature in feature_names:
        Q1 = clean_df[feature].quantile(0.1)
        Q3 = clean_df[feature].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clean_df = clean_df[(clean_df[feature] >= lower_bound) & (clean_df[feature] <= upper_bound)]
    return clean_df
    
if __name__ == '__main__':
    # Split the dataset
    train_data, test_data = split_data()

    # Save the split dataset
    save_split_data(train_data, test_data)
    
    # Load the training data
    X_train, y_train = load_training_data()
    print(X_train.head())
    print('-'*50)
    print(y_train.head())