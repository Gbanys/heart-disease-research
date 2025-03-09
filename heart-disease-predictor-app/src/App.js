import React, { useState } from 'react';
import './App.css';

function App() {
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

  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;

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

    try {
      const response = await fetch("http://localhost:3050/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputData),
      });

      const result = await response.json();
      setPrediction(result); // Store API response in state
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="App">
      <div className="layout">
        <aside className="sidebar">
          <h2>Sidebar</h2>
          <p>This is the sidebar content.</p>
        </aside>
        <main className="main-content">
          <h1>Heart Disease Prediction</h1>
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
          <div>{prediction.probabilities ? prediction.probabilities[0] : "No prediction"}</div>
        </main>
      </div>
    </div>
  );
}

export default App;
