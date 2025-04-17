# src/predict.py
import pickle
import pandas as pd
import numpy as np
import os
from src.logger import setup_logger

# Set up logger for this module
logger = setup_logger('predict')

def load_artifacts():
    """Load model, scaler, and feature names."""
    logger.info("Loading model artifacts")
    model_path = 'models/model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug("Model loaded successfully")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.debug("Scaler loaded successfully")
        
        with open('data/raw/target_names.pkl', 'rb') as f:
            target_names = pickle.load(f)
        logger.debug("Target names loaded successfully")
        
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        logger.debug("Feature names loaded successfully")
        
        return model, scaler, feature_names, target_names
    
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}", exc_info=True)
        raise

def make_prediction(input_data):
    """Make prediction based on user input."""
    logger.info(f"Making prediction with input data: {input_data}")
    
    try:
        model, scaler, feature_names, target_names = load_artifacts()
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)
        logger.debug(f"Input data as DataFrame:\n{input_df}")
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        logger.debug("Input data scaled successfully")
        
        # Make prediction
        prediction_idx = model.predict(input_scaled)[0]
        predicted_species = target_names[prediction_idx]
        logger.info(f"Prediction result: {predicted_species} (class {prediction_idx})")
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        logger.debug(f"Prediction probabilities: {dict(zip(target_names, probabilities))}")
        
        return prediction_idx, predicted_species, probabilities, target_names
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise