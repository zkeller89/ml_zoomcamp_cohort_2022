# ML Zoomcamp Capstone

## Project Description

This project is based on the kaggle competition: [Kitchen Classification](https://www.kaggle.com/competitions/kitchenware-classification)

The objective of this competition is:

<blockquote>
 ... to classify images of different kitchenware items into 6 classes:
 <ul>
<li>cups </li>
<li>glasses</li>
<li>plates</li>
<li>spoons</li>
<li>forks</li>
<li>knives</li>
 </ul>
</blockquote>

It is meant to motivate the students of this ML Zoomcamp cohort (and other's interested taking part in the competition) to begin building predictive models in a less contrived environment.

The solution to the classification problem will be a model used to classify images in the 'test' set, and ultimately submit a CSV specifying the classes of the images, which will be judged and ranked.

## EDA
Relevant file: `train.ipynb`:

* We explored the images sizes to see the resolution of the various images
* We examined the metadata (`dtype`s, shape, and size) of a few images
* We printed a sample of the images to get an idea for what they look like visually
* We ensure the RGB values were valid
* We looked at the distribution of values within the training dataset.

*note*: the training dataset is located in `images_by_class/train/[class_label]`

## Model Training
Relevant file: `train.ipynb`:

We attempted to train two models:
* The first attempt used a very basic model which generally followed the Keras introduction instructions. It involved a few convolutional layers and a dense layer.
* It was clear that model would no perform well, so we then moved on to train another model using transfer learning model (with an Xception base).
    * We also added an addditional dense layer on top of the base.
    * The Xception base was frozen
    * After it was clear this model was more performance, the learning rate was then chosen based on validation set accuracy.

## Exporting notebook to script
Relevant file: `train.py`

The file `train.py` conatins all info needed to train the model and save the result to a `tflite` file. To do so, install the needed depdencies and run `pythin train.py`

## Reproducibility

The relevant code needed would be in the following two files:
* `train.ipynb`
* `train.py`

Without a GPU available, training may take some time or memory issues may be encountered. [Saturn Cloud](https://saturncloud.io/) was used for this project.

## Cloud/Model deployment

* This model was deployed to AWS Lambda and can be accessed using a `POST` request.
* The url is [here](https://ppr8u3xh1c.execute-api.us-east-1.amazonaws.com/test/predict). Please see `test.py` for an example call to the lambda function, which should be accessible for more general testing purposes.

## Dependency and enviroment management

To install relevant dependencies, please use [`pipenv`](https://pipenv-fork.readthedocs.io/en/latest/) in conjunction with the requirements file. Navigate the the directory of this project and run:
```
pipenv install -r requirements.txt
```

## Containerization
* There is a Dockerfile contained in this directory
* You can emulate the serving that the lambda function would typically do by building an image and running the container by doing the following:
    1. Navigate to this directory
    2. Run
    ```
    > docker build -t capstone_model .
    ```

    * You can replace `capstone_model` with whatever you want to name the image

    3. Run the image with:

    ```
    > docker run -it --rm -p 8080:8080 capstone_model:latest
    ```

    4. You can now test the model with the following code in a python file:
    ```
    import requests

    url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
    data = {'url': 'https://github.com/zkeller89/ml_zoomcamp_cohort_2022/blob/main/capstone_project/images/0015.jpg?raw=true'}

    result = requests.post(url, json=data).json()
    print(result)
    ```

    * The `url` dict value can be replaced which the user's desired image.
