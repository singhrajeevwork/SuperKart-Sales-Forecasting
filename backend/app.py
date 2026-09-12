
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to the trained model in the same folder as app.py
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superkart_sales_prediction_model.joblib"
)

# Load the trained model pipeline
model = joblib.load(MODEL_PATH)

# Expected input features
REQUIRED_FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_Type_Category',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_char',
    'Store_Age_Years'
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "SuperKart Sales Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is empty"
            }), 400

        # Single prediction
        if isinstance(data, dict):
            input_data = pd.DataFrame([data])

            missing_features = [
                col for col in REQUIRED_FEATURES
                if col not in input_data.columns
            ]

            if missing_features:
                return jsonify({
                    "error": f"Missing features: {missing_features}"
                }), 400

            prediction = model.predict(input_data)

            return jsonify({
                "predicted_sales": float(prediction[0])
            })

        # Batch prediction
        elif isinstance(data, list):
            if len(data) == 0:
                return jsonify({
                    "error": "Input list is empty"
                }), 400

            input_data = pd.DataFrame(data)

            missing_features = [
                col for col in REQUIRED_FEATURES
                if col not in input_data.columns
            ]

            if missing_features:
                return jsonify({
                    "error": f"Missing features: {missing_features}"
                }), 400

            predictions = model.predict(input_data)

            return jsonify({
                "predicted_sales": [
                    float(prediction) for prediction in predictions
                ]
            })

        else:
            return jsonify({
                "error": "Input must be a JSON object or a list of JSON objects"
            }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
