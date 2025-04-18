import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.model import *
from src.data_processing import *
import os
import pickle

def test_train_model():
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    assert model is not None

def test_evaluate_model():
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test, ['setosa', 'versicolor', 'virginica'])
    assert accuracy > 0.5

def test_save_and_load_model():
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    model_path = 'models/model_test.pkl'
    
    # Save the model
    save_model(model, model_path)
    assert os.path.exists(model_path)

    # Load the model and check
    loaded_model = load_model(model_path)
    assert loaded_model is not None
