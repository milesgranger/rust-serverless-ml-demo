import requests

from collections import namedtuple
from pprint import pprint

from sklearn.datasets import load_iris
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DemoResult = namedtuple('DemoResult', ['model', 'xTest', 'yTest'])


def get_model(url):
    """
    Get an untrained model
    """
    return requests.get(url).json()["model"]


def get_train_test_split():
    """
    Load xTrain, xTest, yTrain, yTest
    """
    X, y = load_iris(True)
    y = OneHotEncoder(3).fit_transform(y.reshape(-1, 1)).toarray()  # One hot encode targets
    return train_test_split(X, y, test_size=0.05)


def get_predictions(url, model, x):
    """
    Fetch predictions given a model and data
    """
    return requests.post(url, json={"model": model, "x": x.tolist()}).json()["predictions"]


def train_model_one_round(url, model, x, y):
    """
    Train the model on one epoch and return updated model
    """
    return requests.post(url, json={"model": model, "x": x.tolist(), "y": y.tolist()}).json()["model"]


def demo(url):
    """
    Run the whole pipeline example for 10 epochs.
    """
    model = get_model(url)
    xTrain, xTest, yTrain, yTest = get_train_test_split()
    for epoch in range(1, 11):
        model = train_model_one_round(url, model, xTrain, yTrain)
        predictions = get_predictions(url, model, xTest)
        score = log_loss(yTest, predictions)
        print(f"Epoch: {epoch:0>2} - Score: {score:0>4}")
