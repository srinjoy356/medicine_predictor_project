import React, { useState } from 'react';
import axios from 'axios';

function HealthAssistant() {
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: '',
  });

  const [result, setResult] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict/diabetes', formData);
      setResult(response.data.prediction);
    } catch (error) {
      setResult('Error in prediction: ' + error.response.data.error);
    }
  };

  return (
    <div>
      <h1>Diabetes Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          name="Pregnancies"
          placeholder="Pregnancies"
          value={formData.Pregnancies}
          onChange={handleChange}
        />
        <input
          type="text"
          name="Glucose"
          placeholder="Glucose"
          value={formData.Glucose}
          onChange={handleChange}
        />
        <input
          type="text"
          name="BloodPressure"
          placeholder="Blood Pressure"
          value={formData.BloodPressure}
          onChange={handleChange}
        />
        <input
          type="text"
          name="SkinThickness"
          placeholder="Skin Thickness"
          value={formData.SkinThickness}
          onChange={handleChange}
        />
        <input
          type="text"
          name="Insulin"
          placeholder="Insulin"
          value={formData.Insulin}
          onChange={handleChange}
        />
        <input
          type="text"
          name="BMI"
          placeholder="BMI"
          value={formData.BMI}
          onChange={handleChange}
        />
        <input
          type="text"
          name="DiabetesPedigreeFunction"
          placeholder="Diabetes Pedigree Function"
          value={formData.DiabetesPedigreeFunction}
          onChange={handleChange}
        />
        <input
          type="text"
          name="Age"
          placeholder="Age"
          value={formData.Age}
          onChange={handleChange}
        />
        <button type="submit">Predict Diabetes</button>
      </form>
      <h2>{result}</h2>
    </div>
  );
}

export default HealthAssistant;
