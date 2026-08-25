from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb

app = Flask(__name__)

# Load the trained model using XGBoost's native format
model = xgb.XGBRegressor()
model.load_model('scd_hb_model.json')

# Load the exact column list the model expects
model_columns = joblib.load('scd_hb_model_columns.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json()

        # Wrap the single patient record in a DataFrame
        input_df = pd.DataFrame([payload])

        # One-hot encode this patient's categorical fields the same way
        # training data was encoded
        input_encoded = pd.get_dummies(input_df)

        # Reindex against the saved training columns:
        # - adds any missing dummy columns, filled with 0
        # - drops any columns that don't exist in the trained model
        # - puts columns in the exact order the model expects
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_final)

        return jsonify({
            'predicted_future_hb': float(prediction[0])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)