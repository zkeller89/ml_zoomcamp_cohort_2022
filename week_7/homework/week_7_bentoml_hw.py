import bentoml

from bentoml.io import JSON, NumpyNdarray

# model 1
model_ref = bentoml.sklearn.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5')

# model 2
# model_ref = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')

model_runner = model_ref.to_runner()
model_runner = model_ref.to_runner()

svc = bentoml.Service('mlzoomcamp_hw', runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(application_data):
    prediction = model_runner.predict.run(application_data)
    return prediction
