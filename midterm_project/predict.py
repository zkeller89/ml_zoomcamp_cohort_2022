import pickle

from flask import Flask, request, jsonify

model_file = 'model_midterm.bin'

with open(model_file, 'rb') as f_in:
    final_model, transform_training, DMatrix = pickle.load(f_in)

app = Flask('readmission')

def get_prediction_dict(patient_json):
    X = transform_training(patient_json)
    y_pred = model.predict(X)
    # could more precisely select threshold, will use 0.5 for now
    readmission = y_pred >= 0.5

    result = {
       'readmission_probability': float(y_pred),
       'readmission' = bool(readmission)
    }

    return result

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    result = get_prediction_dict(patient)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)