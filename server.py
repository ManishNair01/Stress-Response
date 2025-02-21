from flask import Flask, request, jsonify
import joblib
import numpy as np
import predict
from flask_cors import CORS

host="192.168.27.155"

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your frontend

# Load the trained ML model
model = joblib.load('../stress_model.pkl')

# Store the latest prediction and input data to serve to the frontend
latest_prediction = {}

@app.route('/predict', methods=['POST'])
def predict_stress():
    global latest_prediction

    try:
        # Parse JSON input from Arduino
        data = request.json
        heartrate = float(data['heartRate'])
        temperature = float(data['bodyTemperature'])
        spo2 = float(data['oxygenSaturation'])

        # Prepare data for prediction
        input_data = np.array([[heartrate, temperature, spo2]])
        prediction = predict.predict_driver_state(temperature,heartrate, spo2)
        # stress_level = int(prediction[0])
        print("Output: ",prediction)

        # return jsonify({"stress_level": int(prediction)})

        # Store the processed data for the frontend
        latest_prediction = {
            "stress_level": int(prediction),
            "heartRate": heartrate,
            "bodyTemperature": temperature,
            "oxygenSaturation": spo2
        }

        print(f"Prediction: {latest_prediction}")
        return jsonify({"status": "success", "data": latest_prediction}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/get_latest_prediction', methods=['GET'])
def get_latest_prediction():
    """
    Serve the latest prediction data to the frontend.
    """
    if latest_prediction:
        return jsonify(latest_prediction), 200
    else:
        return jsonify({"error": "No data available"}), 404

if __name__ == '__main__':
    app.run(host, port=5000, debug=True)








    # .\venv\Scripts\Activate
    # python server.py
    # frontend show preview