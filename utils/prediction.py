import numpy as np

def predict_disease(disease, input_data, model, scaler):
    """
    Generic prediction function for all diseases
    """

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability

