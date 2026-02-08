from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# =====================================================
# LOAD MODELS & SCALERS
# =====================================================
BASE_DIR = os.getcwd()

diabetes_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "diabetes.pkl"))
diabetes_scaler = joblib.load(os.path.join(BASE_DIR, "saved_models", "diabetes_scaler.pkl"))

heart_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "heart.pkl"))
heart_scaler = joblib.load(os.path.join(BASE_DIR, "saved_models", "heart_scaler.pkl"))

kidney_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney.pkl"))
kidney_scaler = joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney_scaler.pkl"))
kidney_features = joblib.load(os.path.join(BASE_DIR, "saved_models", "kidney_features.pkl"))

# =====================================================
# BASIC ROUTES
# =====================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/kidney")
def kidney():
    return render_template("kidney.html")

# =====================================================
# DIABETES PREDICTION + CHART VALUES
# =====================================================
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():

    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    bp = float(request.form["bloodpressure"])
    skin = float(request.form["skinthickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    dpf = float(request.form["dpf"])
    age = float(request.form["age"])

    X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    X_scaled = diabetes_scaler.transform(X)

    prediction = diabetes_model.predict(X_scaled)[0]
    probability = diabetes_model.predict_proba(X_scaled)[0][1]

    return render_template(
        "result.html",
        disease="Diabetes",
        prediction=prediction,
        probability=round(probability * 100, 2),
        chart_labels=["Glucose", "BMI"],
        normal_values=[140, 25],
        patient_values=[glucose, bmi]
    )

# =====================================================
# HEART DISEASE PREDICTION + CHART VALUES
# =====================================================
@app.route("/predict/heart", methods=["POST"])
def predict_heart():

    age = float(request.form["age"])
    sex = int(request.form["sex"])
    cp = int(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = int(request.form["fbs"])
    restecg = int(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thal = int(request.form["thal"])

    X = np.array([[age, sex, cp, trestbps, chol, fbs,
                   restecg, thalach, exang, oldpeak,
                   slope, ca, thal]])

    X_scaled = heart_scaler.transform(X)

    prediction = heart_model.predict(X_scaled)[0]
    probability = heart_model.predict_proba(X_scaled)[0][1]

    return render_template(
        "result.html",
        disease="Heart Disease",
        prediction=prediction,
        probability=round(probability * 100, 2),
        chart_labels=["Cholesterol", "Resting BP"],
        normal_values=[200, 120],
        patient_values=[chol, trestbps]
    )

# =====================================================
# KIDNEY DISEASE PREDICTION + CHART VALUES
# =====================================================
@app.route("/predict/kidney", methods=["POST"])
def predict_kidney():

    input_dict = {
        "age": float(request.form["age"]),
        "blood_pressure": float(request.form["bp"]),
        "specific_gravity": float(request.form["sg"]),
        "albumin": float(request.form["al"]),
        "sugar": float(request.form["su"]),
        "red_blood_cells": 1,
        "pus_cell": 1,
        "pus_cell_clumps": 0,
        "bacteria": 0,
        "blood_glucose_random": float(request.form["bgr"]),
        "blood_urea": float(request.form["bu"]),
        "serum_creatinine": float(request.form["sc"]),
        "sodium": float(request.form["sod"]),
        "potassium": float(request.form["pot"]),
        "hemoglobin": float(request.form["hemo"]),
        "packed_cell_volume": float(request.form["pcv"]),
        "white_blood_cell_count": float(request.form["wc"]),
        "red_blood_cell_count": float(request.form["rc"]),
        "hypertension": int(request.form["htn"]),
        "diabetes_mellitus": int(request.form["dm"]),
        "coronary_artery_disease": int(request.form["cad"]),
        "appetite": 1,
        "pedal_edema": int(request.form["pe"]),
        "anemia": int(request.form["ane"])
    }

    input_df = pd.DataFrame([input_dict])[kidney_features]
    X_scaled = kidney_scaler.transform(input_df)

    prediction = kidney_model.predict(X_scaled)[0]
    probability = kidney_model.predict_proba(X_scaled)[0][1]

    return render_template(
        "result.html",
        disease="Kidney Disease",
        prediction=prediction,
        probability=round(probability * 100, 2),
        chart_labels=["Creatinine", "Blood Urea"],
        normal_values=[1.2, 40],
        patient_values=[input_dict["serum_creatinine"], input_dict["blood_urea"]]
    )

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
