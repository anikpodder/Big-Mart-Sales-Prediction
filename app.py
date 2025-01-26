from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    # Extract input features from the form
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    # Prepare the input feature array
    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    # Load the scaler
    scaler_path = r'C:\Users\w10\Desktop\BigMart-Sales-Prediction-With-Deployment-main\models\sc.sav'
    sc = joblib.load(scaler_path)

    # Transform the input features
    X_std = sc.transform(X)

    # Load the model
    model_path = r'C:\Users\w10\Desktop\BigMart-Sales-Prediction-With-Deployment-main\models\lr.sav'
    model = joblib.load(model_path)

    # Ensure the 'positive' attribute exists
    if not hasattr(model, "positive"):
        model.positive = False  # Set the attribute based on your model's requirements

    # Make predictions
    Y_pred = model.predict(X_std)

    # Return the prediction as JSON
    return render_template("result.html", prediction=round(float(Y_pred[0]), 2))

if __name__ == "__main__":
    app.run(debug=True, port=9457)
