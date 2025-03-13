import React, { useState } from 'react';
import { FaHome, FaChartLine } from "react-icons/fa";
import './App.css';

const BENTOML_SERVICE_URL = process.env.BENTOML_SERVICE_URL;

function App() {
  const [activeView, setActiveView] = useState("home");

  // Initialize state for all form fields
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    cp: '',
    trestbps: '',
    chol: '',
    fbs: '',
    restecg: '',
    thalach: '',
    exang: '',
    oldpeak: '',
    slope: '',
    ca: '',
    thal: ''
  });

  const [prediction, setPrediction] = useState("None");
  const [error, setError] = useState(null); // New state for error messages
  const [validationError, setValidationError] = useState(""); // New state for validation

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setValidationError("");

    if (name === "sex") {
      setFormData(prev => ({
        ...prev,
        sex: value === 'male' ? 1 : 0,
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value,
      }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const inputData = {
      input_data: Object.values(formData).map(Number), // Convert values to numbers
    };

    if (Object.values(formData).some(value => value === "")) {
      setValidationError("Please fill in all fields before submitting.");
      setPrediction("None");
      return;
    }

    try {
      const response = await fetch(`${BENTOML_SERVICE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });

      const result = await response.json();
      setPrediction(result.probabilities[0]); // Store API response in state
      setError(null);
    } catch (error) {
      console.error("Error:", error);
      setError("There was an error generating the prediction. Please try again.");
      setPrediction("None");
    }
  };

  return (
    <div className="App">
      <div className="layout">
        <aside className="sidebar">
          <h2>Options</h2>
          <button onClick={() => setActiveView("home")}>
            <FaHome style={{ marginRight: "8px" }} /> Home
          </button>
          <button onClick={() => setActiveView("predict")}>
            <FaChartLine style={{ marginRight: "8px" }} /> Predict
          </button>
        </aside>
        <main className="main-content">
          {activeView === "home" && (
            <div className="home-content">
              <h1>Welcome to the Heart Disease Prediction App</h1>
              <p>
                This application provides information about heart disease and allows you to predict your risk by filling out a form.
                Click on Predict to enter your details and see your prediction.
              </p>
            </div>
          )}
          {activeView === "predict" && (
          <div className="predict-content">
            <h1>Heart Disease Prediction App</h1>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="age">Age:</label>
                <input type="number" id="age" name="age" value={formData.age} onChange={handleChange} min="18" max="70" />
              </div>
              <div className="form-group">
                <p>Sex:</p>
                <div className="radio-option">
                  <input
                    type="radio"
                    id="male"
                    name="sex"
                    value="male"
                    onChange={handleChange}
                    checked={formData.sex === 1}
                  />
                  <label htmlFor="male">Male</label>
                </div>
                <div className="radio-option">
                  <input
                    type="radio"
                    id="female"
                    name="sex"
                    value="female"
                    onChange={handleChange}
                    checked={formData.sex === 0}
                  />
                  <label htmlFor="female">Female</label>
                </div>
              </div>
              <div className="form-group">
                <label htmlFor="cp">CP:</label>
                <input type="number" id="cp" name="cp" value={formData.cp} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="trestbps">Trestbps:</label>
                <input type="number" id="trestbps" name="trestbps" value={formData.trestbps} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="chol">Chol:</label>
                <input type="number" id="chol" name="chol" value={formData.chol} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="fbs">Fbs:</label>
                <input type="number" id="fbs" name="fbs" value={formData.fbs} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="restecg">Restecg:</label>
                <input type="number" id="restecg" name="restecg" value={formData.restecg} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="thalach">Thalach:</label>
                <input type="number" id="thalach" name="thalach" value={formData.thalach} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="exang">Exang:</label>
                <input type="number" id="exang" name="exang" value={formData.exang} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="oldpeak">Oldpeak:</label>
                <input type="number" step="any" id="oldpeak" name="oldpeak" value={formData.oldpeak} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="slope">Slope:</label>
                <input type="number" id="slope" name="slope" value={formData.slope} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="ca">Ca:</label>
                <input type="number" id="ca" name="ca" value={formData.ca} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label htmlFor="thal">Thal:</label>
                <input type="number" id="thal" name="thal" value={formData.thal} onChange={handleChange} />
              </div>
              <button type="submit">Submit</button>
            </form>
              {validationError && <div className="error-message" style={{ color: 'orange', marginTop: '10px' }}>{validationError}</div>}
              {prediction !== "None" && (
                <div className={prediction === 1 ? 'positive-prediction-result' : 'negative-prediction-result'}>
                  {prediction === 1 
                    ? "The model predicts that you're likely to have heart disease." 
                    : "The model predicts that you're not likely to have heart disease."}
                </div>
              )}
              {error && <div className="error-message" style={{ color: 'red', marginTop: '20px' }}>{error}</div>}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
