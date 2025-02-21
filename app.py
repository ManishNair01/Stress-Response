from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)

# Load the trained ML model
model = joblib.load('./stress_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_stress():
    try:
        # Parse JSON input
        data = request.json
        heart_rate = float(data['heart_rate'])
        temperature = float(data['temperature'])
        spo2 = float(data['spo2'])

        # Prepare data for prediction
        input_data = np.array([[heart_rate, temperature, spo2]])

        # Make prediction
        prediction = model.predict(input_data)
        stress_level = int(prediction[0])

        return jsonify({"stress_level": stress_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)