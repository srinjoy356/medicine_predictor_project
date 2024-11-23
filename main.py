from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import re
from flask_cors import CORS
import os
from prediction_model import get_predicted_value, get_general_medications, helper, diabetes_model, heart_disease_model, \
    parkinsons_model, symptoms_dict

app = Flask(__name__)

CORS(app)


@app.route('/')
def landing():
    return "Welcome to the Health Assistant API"


@app.route('/predict', methods=['POST'])
def home():
    try:

        data = request.get_json()
        symptoms = data.get('symptoms', [])

        if not symptoms:
            return jsonify({'message': 'Please provide valid symptoms.'}), 400

        user_symptoms = [s.strip().lower() for s in symptoms]

        predicted_disease = get_predicted_value(user_symptoms)

        if predicted_disease == "Unknown disease":
            return jsonify({
                               'message': 'Unable to predict disease based on the given symptoms. Please provide more information.'}), 400

        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        if isinstance(precautions, pd.Series):
            precautions = precautions.tolist()
        my_precautions = [item for sublist in precautions for item in sublist]

        if isinstance(medications, pd.Series):
            medications = medications.tolist()

        general_meds = get_general_medications(user_symptoms)
        all_medications = list(set(medications + general_meds))

        if isinstance(rec_diet, pd.Series):
            rec_diet = rec_diet.tolist()
        if isinstance(workout, pd.Series):
            workout = workout.tolist()

        print({
            'predicted_disease': predicted_disease,
            'description': dis_des,
            'precautions': my_precautions,
            'medications': all_medications,
            'diet': rec_diet,
            'workout': workout
        })

        return jsonify({
            'predicted_disease': predicted_disease,
            'description': dis_des,
            'precautions': my_precautions,
            'medications': all_medications,
            'diet': rec_diet,
            'workout': workout
        })

    except Exception as e:
        return jsonify({'message': str(e)}), 500


@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    try:
        symptoms_list = [symptoms for symptoms, val in symptoms_dict.items()]
        return jsonify({'symptoms': symptoms_list}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# multiple disease


# API Endpoint for Diabetes Prediction
@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    try:

        data = request.get_json()

        features = [float(data[key]) for key in ['Pregnancies', 'Glucose', 'BloodPressure',
                                                 'SkinThickness', 'Insulin', 'BMI',
                                                 'DiabetesPedigreeFunction', 'Age']]

        diab_prediction = diabetes_model.predict([features])

        if diab_prediction[0] == 1:
            diagnosis = 'The person is diabetic'
        else:
            diagnosis = 'The person is not diabetic'

        return jsonify({'prediction': diagnosis})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# API Endpoint for Heart Disease Prediction
@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:

        data = request.get_json()

        features = [float(data[key]) for key in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                                                 'ca', 'thal']]

        heart_prediction = heart_disease_model.predict([features])

        if heart_prediction[0] == 1:
            diagnosis = 'The person is having heart disease'
        else:
            diagnosis = 'The person does not have any heart disease'

        return jsonify({'prediction': diagnosis})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# API Endpoint for Parkinson's Prediction
@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    try:

        data = request.get_json()

        features = [float(data[key]) for key in ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs',
                                                 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB', 'APQ3',
                                                 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
                                                 'spread2', 'D2', 'PPE']]

        parkinsons_prediction = parkinsons_model.predict([features])

        if parkinsons_prediction[0] == 1:
            diagnosis = "The person has Parkinson's disease"
        else:
            diagnosis = "The person does not have Parkinson's disease"

        return jsonify({'prediction': diagnosis})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)