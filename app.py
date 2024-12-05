from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the scaler used for feature scaling
model = joblib.load('iris_model.pkl')  # Load the pre-trained model
scaler = StandardScaler()  # Create a StandardScaler instance

# Route to render the form where users can input feature values (Web Interface)
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form to the user

# Route to handle form submission and return the prediction (Web Interface)
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input feature values from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Create a list of features and scale them
    features = [sepal_length, sepal_width, petal_length, petal_width]
    features_scaled = scaler.fit_transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_species = prediction[0]  # Extract the predicted species

    # Render the prediction on the web page
    return render_template('index.html', prediction=predicted_species)

# API Route to handle JSON input and return JSON response
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Check if the incoming request has JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Parse the JSON input
    data = request.get_json()
    try:
        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]
    except KeyError:
        return jsonify({"error": "Missing feature(s) in JSON input"}), 400

    # Scale the input features
    features_scaled = scaler.fit_transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    predicted_species = prediction[0]

    # Return the prediction as a JSON response
    return jsonify({"prediction": predicted_species})

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
