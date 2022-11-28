import tflite_runtime.interpreter as tflite

import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    size = (150, 150)
    raw_img = download_image(url=url)
    resized_img = prepare_image(img=raw_img, target_size=size)
    img_arr = np.array(resized_img)
    processed_img_arr = img_arr / 255.
    processed_img_arr = np.float32(processed_img_arr)
    
    interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
    interpreter.allocate_tensors()
    
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    preds = interpreter.set_tensor(input_index, [processed_img_arr])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    float_preds = preds[0].tolist()
    
    return float_preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
    

