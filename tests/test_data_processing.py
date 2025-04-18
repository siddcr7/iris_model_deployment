import pytest
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.data_processing import load_iris_data, preprocess_data, split_data, scale_features

def test_load_iris_data():
    df = load_iris_data()
    assert isinstance(df, pd.DataFrame)
    assert 'target' in df.columns
    assert 'species' in df.columns

def test_preprocess_data():
    df = load_iris_data()
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]

def test_split_data():
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

def test_scale_features():
    df = load_iris_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert scaler is not None
