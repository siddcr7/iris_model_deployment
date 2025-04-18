# app.py (User Interface)
"""
This is the main application file that creates the Streamlit user interface.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


# Import the required modules from the src package
from src.predict import make_prediction, load_artifacts
from src.train import run_training
from src.logger import setup_logger, logging

# Set up the main application logger
logger = setup_logger('app', log_level=logging.INFO)

def main():
    logger.info("Starting Iris Flower Classification App")
    st.title("Iris Flower Species Classifier")
    st.write("Enter the measurements to predict the Iris flower species")
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Check if model exists, if not, train it
    if not os.path.exists('models/model.pkl'):
        logger.warning("Model not found. Training the model now...")
        st.warning("Model not found. Training the model now...")
        run_training()
        st.success("Model trained successfully!")
        logger.info("Model trained successfully")
    else:
        logger.info("Using existing model")
    
    try:
        # Load feature names and target names
        logger.debug("Loading model artifacts")
        _, _, feature_names, target_names = load_artifacts()
        
        # Create input fields with default values based on typical ranges for each feature
        st.subheader("Enter Flower Measurements (cm)")
        
        input_data = {}
        
        input_data['sepal length'] = st.slider(
            "Sepal Length (cm)", 
            min_value=4.0, 
            max_value=8.0, 
            value=5.8, 
            step=0.1
        )
        
        input_data['sepal width'] = st.slider(
            "Sepal Width (cm)", 
            min_value=2.0, 
            max_value=4.5, 
            value=3.0, 
            step=0.1
        )
        
        input_data['petal length'] = st.slider(
            "Petal Length (cm)", 
            min_value=1.0, 
            max_value=7.0, 
            value=4.0, 
            step=0.1
        )
        
        input_data['petal width'] = st.slider(
            "Petal Width (cm)", 
            min_value=0.1, 
            max_value=2.5, 
            value=1.3, 
            step=0.1
        )
        
        if st.button("Predict Species"):
            logger.info(f"Making prediction with input: {input_data}")
            prediction_idx, predicted_species, probabilities, target_names = make_prediction(input_data)
            logger.info(f"Prediction result: {predicted_species} (class {prediction_idx})")
            
            # Display prediction
            st.success(f"Predicted Species: **{predicted_species}**")
            
            # Display prediction probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Species': target_names,
                'Probability': probabilities
            })
            logger.debug(f"Prediction probabilities: {dict(zip(target_names, probabilities))}")
            
            # Create a horizontal bar chart for probabilities
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.barh(prob_df['Species'], prob_df['Probability'], color=['#FF9999', '#66B2FF', '#99FF99'])
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Probabilities by Species')
            
            # Add probability values on the bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(max(width + 0.01, 0.05), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{probabilities[i]:.4f}', 
                        va='center')
            
            st.pyplot(fig)
            
            # Display input data summary
            st.subheader("Input Summary")
            st.write(pd.DataFrame([input_data]))
        
        # Add dataset information
        with st.expander("About the Iris Dataset"):
            st.write("""
            The Iris dataset is a classic dataset in machine learning and statistics. It contains measurements for 150 iris flowers from three different species:
            
            - **Iris setosa**
            - **Iris versicolor**
            - **Iris virginica**
            
            For each flower, four features were measured:
            - Sepal length (cm)
            - Sepal width (cm)
            - Petal length (cm)
            - Petal width (cm)
            
            This dataset is often used for classification tasks and is included in scikit-learn.
            """)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {e}")
        st.write("Please make sure the model is trained properly.")

if __name__ == "__main__":
    main()