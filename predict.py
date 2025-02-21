import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset and train the model (only once when the module is imported)
file_path = 'Emotional State.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Separate features and target variable
X = data[['Body_Temperature', 'Heart_Rate', 'SPO2']]
y = data['Driver_State']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Define a function to predict the stress level
def predict_driver_state(temperature, heartrate, spo2):
    input_data = np.array([[temperature, heartrate, spo2]])
    input_data_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_data_scaled)
    return(prediction[0])

