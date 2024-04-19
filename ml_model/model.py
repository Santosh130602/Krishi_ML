

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Data Preparation

crop_data = pd.read_csv("/Users/santosh/Desktop/Krishi_Doot/ml_model/CSV_File/Crop_recommendation.csv")
# Splitting the dataset into features (X) and target labels (y)
X = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = crop_data['label']

# Step 2: Model Selection (Random Forest Classifier)
model = RandomForestClassifier()

# Step 3: Model Training
model.fit(X, y)

# Step 4: Model Evaluation (Optional)
# You can evaluate the model's performance using cross-validation or on a holdout test set if available.

# Step 5: Save the trained model
joblib.dump(model, 'crop_recommendation_model.pkl')







