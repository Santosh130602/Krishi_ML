
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np 

app = Flask(__name__)
CORS(app)

# Load the ML model for fertilizer recommendation
fertilizer_model = joblib.load('Fertilizer_classifier.pkl')

# Define functions to map soil type and crop type to numeric representations
def map_soil_type(soil_type):
    soil_dict = {
        'Loamy': 1,
        'Sandy': 2,
        'Clayey': 3,
        'Black': 4,
        'Red': 5
    }
    return soil_dict.get(soil_type, 0)  # Return 0 if soil type not found in dictionary

def map_crop_type(crop_type):
    crop_dict = {
        'Sugarcane': 1,
        'Cotton': 2,
        'Millets': 3,
        'Paddy': 4,
        'Pulses': 5,
        'Wheat': 6,
        'Tobacco': 7,
        'Barley': 8,
        'Oil seeds': 9,
        'Ground Nuts': 10,
        'Maize': 11
    }
    return crop_dict.get(crop_type, 0)  # Return 0 if crop type not found in dictionary

# Define endpoint for predicting fertilizer recommendations
@app.route('/fertilizer', methods=['POST'])
def fertilizer_predict():
    try:
        # Get data from request
        data = request.json
        
        # Ensure that all required keys are present in the request data
        required_keys = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil_Type', 'Crop_Type']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing key: {key}'}), 400
            
        # Extract features
        Temperature = data['Temperature']
        Humidity = data['Humidity']
        Moisture = data['Moisture']
        Nitrogen = data['Nitrogen']
        Potassium = data['Potassium']
        Phosphorous = data['Phosphorous']
        Soil_Type = data['Soil_Type']
        Crop_Type = data['Crop_Type']

        # Map soil type and crop type to numeric representations
        soil_num = map_soil_type(Soil_Type)
        crop_num = map_crop_type(Crop_Type)

        # Predict fertilizer
        features = np.array([[Temperature, Humidity, Moisture, Nitrogen, Potassium, Phosphorous, soil_num, crop_num]])
        prediction = fertilizer_model.predict(features)

        # Return prediction
        return jsonify({'recommendation': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return internal server error if any exception occurs
    

    # # ****************************   CORP RECOMMANDATION


model = joblib.load('crop_recommendation_model.pkl')

# Define endpoint for predicting crop recommendations
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    
    # Ensure that all required keys are present in the request data
    required_keys = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'Missing key: {key}'}), 400
    
    # Extract features
    features = [data[key] for key in required_keys]
    
    # Predict crop
    prediction = model.predict([features])[0]
    
    # Return prediction
    return jsonify({'predicted_crop': prediction})

if __name__ == '__main__':
    app.run(debug=False, port=4000)
