import pytest
from sklearn.ensemble import RandomForestClassifier
from src.model import train_model, evaluate_model
from src.data_processing import load_iris_data, preprocess_data, split_data, scale_features
import pickle

def test_train_model():
    # Test that the model can be trained and is of the correct type
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    model = train_model(X_train_scaled, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    # Test that the evaluation function runs without error
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    model = train_model(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test, ['setosa', 'versicolor', 'virginica'])
    assert isinstance(accuracy, float)
    assert accuracy >= 0.9  # Expecting high accuracy

def test_save_and_load_model():
    # Test that the model can be saved and loaded correctly
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    model = train_model(X_train_scaled, y_train)
    model_path = 'models/test_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    assert isinstance(loaded_model, RandomForestClassifier)
