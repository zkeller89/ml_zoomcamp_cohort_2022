FROM public.ecr.aws/lambda/python:3.9

RUN pip install tflite-runtime
# RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl
RUN pip install requests
RUN pip install pillow

COPY xception_model_v1.tflite .
COPY cloud_function.py .

CMD [ "cloud_function.lambda_handler" ]