import pytest
import pandas as pd
# from src.data_processing import load_iris_data, preprocess_data, split_data, scale_features

def test_load_iris_data():
    # Test that the data is loaded correctly and has the expected columns
    df = load_iris_data()
    assert 'sepal length' in df.columns
    assert 'sepal width' in df.columns
    assert 'petal length' in df.columns
    assert 'petal width' in df.columns
    assert 'target' in df.columns
    assert 'species' in df.columns

def test_preprocess_data():
    # Test that preprocessing correctly separates features and target
    df = load_iris_data()
    X, y = preprocess_data(df)
    assert X.shape[1] == 4  # There are 4 feature columns
    assert y.shape[0] == len(df)  # Target length should match the number of samples

def test_split_data():
    # Test splitting the data into train/test sets
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)

def test_scale_features():
    # Test that scaling works and doesn't alter the number of samples or features
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
