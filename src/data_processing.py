
# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os
import pickle
from src.logger import setup_logger




logger = setup_logger('data_processing')

def load_iris_data():
    """Load the Iris dataset."""
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    feature_names = [name.replace(' (cm)', '') for name in iris.feature_names]
    
    df = pd.DataFrame(data=iris.data, columns=feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Save the raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/iris_data.csv', index=False)
    logger.info(f"Raw data saved to data/raw/iris_data.csv")
    
    # Save the target names for later reference
    with open('data/raw/target_names.pkl', 'wb') as f:
        pickle.dump(iris.target_names, f)
    logger.info(f"Target names saved to data/raw/target_names.pkl")
    
    return df

def preprocess_data(df):
    """Preprocess the data."""
    logger.info("Preprocessing data...")
    # No missing values in Iris dataset, but as a general practice:
    df = df.dropna()
    logger.debug(f"Shape after dropping NA values: {df.shape}")
    
    # For Iris, we don't need to process further as features are already numeric
    # Just separating features and target
    X = df.drop(['target', 'species'], axis=1)
    y = df['target']
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    processed_df = pd.concat([X, y], axis=1)
    processed_df.to_csv('data/processed/processed_iris.csv', index=False)
    logger.info(f"Processed data saved to data/processed/processed_iris.csv")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    logger.info("Scaling features with StandardScaler")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info("Scaler model saved to models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, scaler