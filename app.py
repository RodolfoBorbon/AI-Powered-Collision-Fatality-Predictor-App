from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded_models/'

# route the application
@app.route('/', methods=['GET'])
def home():
    # Pass Google Maps API key to template
    google_maps_api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    return render_template('index.html', google_maps_api_key=google_maps_api_key)

@app.route('/result', methods=['POST'])
def result():
    # Handle model file upload
    file = request.files['model_file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Check if the directory exists and create it if necessary
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER']) 

        file.save(file_path)
        # Load the model
        model = joblib.load(file_path)

        # Extract the model name from the filename for display
        model_name = filename.split('.')[0]  
        model_name = model_name.replace('_model', '') 

        # Replace any underscores with spaces for better readability
        model_name = model_name.replace('_', ' ')

        # Capitalize each word for consistency and display
        model_name = ' '.join(word.capitalize() for word in model_name.split())

    else:
        return "No model file provided", 400
    
    # Extract values from form
    data = {
        'INVAGE': request.form['involved_age'],
        'WEEKDAY': request.form['weekday'],
        'ENVIRONMENTAL_CONDITIONS': request.form['environmental_conditions'],
        'ROAD_CONDITIONS': request.form['road_conditions'],
        'PARTIES_INVOLVED': request.form['parties_involved'],
        'DRIVING_CONDITIONS': request.form['driving_conditions'],
        'LATITUDE': float(request.form['latitude']),
        'LONGITUDE': float(request.form['longitude']),
        'DAY': int(request.form['day']),
        'MONTH': int(request.form['month']),
    }

    # Convert dictionary to DataFrame
    data_df = pd.DataFrame(data, index=[0])

    # Make prediction
    prediction = model.predict(data_df)
    print("Prediction:", prediction)

    # Convert numerical prediction to string
    prediction = 'No Fatal Injury' if prediction == 1 else 'Fatal'

    # Check the decision function or predict_proba
    probabilities = None
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(data_df)
        print("Scores Shape:", scores.shape) 

        # Check if scores is 1D or 2D
        if len(scores.shape) == 1:
            # For 1D scores, expand the dimensions for consistent handling
            scores = np.expand_dims(scores, axis=0)

        # Normalize scores to get probabilities
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    elif hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data_df)
    else:
        print("The model does not have a decision_function or predict_proba method.")

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction, probabilities=probabilities, model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True, port=5000)