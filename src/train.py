# src/train.py
import os
import pandas as pd
import pickle
# Use relative imports for modules in the same package
from src.data_processing import load_iris_data, preprocess_data, split_data, scale_features
from src.model import train_model, evaluate_model, save_model
from src.logger import setup_logger

# Set up logger for this module
logger = setup_logger('train')

def run_training():
    """Run the full training pipeline."""
    logger.info("Starting training pipeline...")
    print("Starting training pipeline...")
    
    # Load Iris data
    logger.info("Loading Iris dataset...")
    print("Loading Iris dataset...")
    df = load_iris_data()
    
    # Preprocess data
    logger.info("Preprocessing data...")
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Split data
    logger.info("Splitting data into train and test sets...")
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    logger.info("Scaling features...")
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Load target names
    with open('data/raw/target_names.pkl', 'rb') as f:
        target_names = pickle.load(f)
    
    # Train model
    logger.info("Training model...")
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    logger.info("Evaluating model...")
    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test_scaled, y_test, target_names)
    
    # Save model
    save_model(model)
    
    # Save feature names for prediction
    feature_names = X.columns.tolist()
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info("Feature names saved to models/feature_names.pkl")
    
    logger.info(f"Training completed successfully! Model accuracy: {accuracy:.4f}")
    print("Training completed successfully!")
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, feature_names, target_names