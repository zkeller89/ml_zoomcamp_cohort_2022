import numpy as np
import requests

import tflite_runtime.interpreter as tflite
from PIL import Image

interpreter = tflite.Interpreter('xception_model_v1.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']

def predict(url):
    img = Image.open(requests.get(url, stream=True).raw)
    img = img.resize((256, 256), Image.NEAREST)

    X = np.array(img, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], [X])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    return dict(zip(classes, preds[0]))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result