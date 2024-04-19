


import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Load the dataset
data = pd.read_csv("/Users/santosh/Desktop/Krishi_Doot/ml_model/CSV_File/Fertilizer Prediction.csv 12-35-09-184.csv")

# Preprocess soil type and crop type
data['Soil_Num'] = data['Soil Type'].map(map_soil_type)
data['Crop_Num'] = data['Crop Type'].map(map_crop_type)

# Drop original soil type and crop type columns
data = data.drop(['Soil Type', 'Crop Type'], axis=1)

# Split features and target variable
X = data.drop(['Fertilizer Name'], axis=1)
Y = data['Fertilizer Name']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_train, Y_train)

# Save the trained model
joblib.dump(fertilizer_model, 'Fertilizer_classifier.pkl')
