import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url = 'https://ppr8u3xh1c.execute-api.us-east-1.amazonaws.com/test/predict'
data = {'url': 'https://github.com/zkeller89/ml_zoomcamp_cohort_2022/blob/main/capstone_project/images/0015.jpg?raw=true'}

result = requests.post(url, json=data).json()
print(result)