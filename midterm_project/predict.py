import pickle

from pandas import DataFrame
from flask import Flask, request, jsonify
from train_helpers import CATEGORICAL, INT64,FEATURES, prep_df

model_file = 'model_midterm.bin'

with open(model_file, 'rb') as f_in:
    model, DMatrix = pickle.load(f_in)

app = Flask('readmission')

def get_prediction_dict(patient_json):
    print(patient_json)
    df = DataFrame.from_records([patient_json])
    df, _ = prep_df(df)
    ddf = DMatrix(df, feature_names=FEATURES, enable_categorical=True)
    y_pred = model.predict(ddf)
    # could more precisely select threshold, will use 0.5 for now
    readmission = y_pred >= 0.5

    result = {
       'readmission_probability': float(y_pred),
       'readmission': bool(readmission)
    }

    return result

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    result = get_prediction_dict(patient)

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)