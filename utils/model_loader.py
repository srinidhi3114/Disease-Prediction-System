import joblib
import os

def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    models = {
        "diabetes_model": joblib.load(os.path.join(BASE_DIR, "saved_models", "diabetes.pkl")),
        "diabetes_scaler": joblib.load(os.path.join(BASE_DIR, "saved_models", "diabetes_scaler.pkl")),

        "heart_model": joblib.load(os.path.join(BASE_DIR, "saved_models", "heart.pkl")),
        "heart_scaler": joblib.load(os.path.join(BASE_DIR, "saved_models", "heart_scaler.pkl")),

        "kidney_model": joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney.pkl")),
        "kidney_scaler": joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney_scaler.pkl")),
        "kidney_features": joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney_features.pkl")),
    }

    return models
