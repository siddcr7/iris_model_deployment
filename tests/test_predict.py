# import pytest
# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# from src.predict import make_prediction, load_artifacts

# def test_make_prediction():
#     input_data = {
#         'sepal length': 5.8,
#         'sepal width': 3.0,
#         'petal length': 4.0,
#         'petal width': 1.3
#     }
    
#     # Load model artifacts
#     model, scaler, feature_names, target_names = load_artifacts()

#     # Make a prediction
#     prediction_idx, predicted_species, probabilities, target_names = make_prediction(input_data)

#     # Check if prediction is one of the target species
#     assert predicted_species in target_names

#     # Check if probabilities are returned as expected
#     # assert isinstance(probabilities, list)
#     # assert len(probabilities) == len(target_names)
