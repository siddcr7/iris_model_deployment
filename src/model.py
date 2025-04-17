# src/model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import numpy as np
from src.logger import setup_logger

# Set up a logger for this module
logger = setup_logger('model')

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest classifier."""
    logger.info(f"Training RandomForestClassifier with n_estimators={n_estimators}, random_state={random_state}")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the model performance."""
    logger.info("Evaluating model performance on test set")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    logger.info(f"Classification Report:\n{report}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Print for console output
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy

def save_model(model, model_path='models/model.pkl'):
    """Save the trained model to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    print(f"Model saved to {model_path}")

def load_model(model_path='models/model.pkl'):
    """Load the trained model from disk."""
    logger.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise