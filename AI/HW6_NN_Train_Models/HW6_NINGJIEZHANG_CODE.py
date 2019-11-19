#Ningjie Zhang, AI, NXZ190005 2021501533

import math
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import torch

def classification_model():
    predictor = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    train = pd.read_csv('titanic_train.csv')[predictor]

    for i in range(len(train.index)):
        if pd.isna(train.at[i, "Age"]):
            if train.at[i, "Pclass"] is 1:
                train.at[i, "Age"] = 37
            elif train.at[i, "Pclass"] is 2:
                train.at[i, "Age"] = 29
            else:
                train.at[i, "Age"] = 24

    train.dropna(inplace=True)
    train.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"C": 0, "Q": 1, "S": 2},
                   }, inplace=True)

    predictor2 = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    x = np.asarray(train[predictor2]).astype(np.float64)
    y = np.asarray(train["Survived"]).astype(np.float64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # --------------------------------------------
    # codes from HW5
    x_train = torch.tensor(x_train, dtype=torch.float, device="cpu")
    x_test = torch.tensor(x_test, dtype=torch.float, device="cpu")
    y_train = torch.tensor(y_train, dtype=torch.long, device="cpu")
    y_test = torch.tensor(y_test, dtype=torch.long, device="cpu")

    inputlayer = x.shape[1]
    layer1 = 5
    layer2 = 3
    outputlinear = 2

    model = torch.nn.Sequential(
        torch.nn.Linear(inputlayer, layer1),
        torch.nn.ReLU(),
        torch.nn.Linear(layer1, layer2),
        torch.nn.Tanh(),
        torch.nn.Linear(layer2, outputlinear)
    ).to("cpu")

    Opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    LossFun = torch.nn.CrossEntropyLoss()

    for i in range(5001):
        y_pred = model(x_train)
        loss = LossFun(y_pred, y_train)
        if (i % 1000) == 0:
            print("when ", i, " Loss: ", loss.item())
        Opt.zero_grad()
        loss.backward()
        Opt.step()

    y_pred = model(x_test)
    loss = LossFun(y_pred, y_test)
    print("loss testing is: ", loss.item())
    y_test = y_test.detach().cpu()
    y_pred = np.argmax(y_pred.detach().cpu(), axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, warn_for=())
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))
    print("Support: " + str(support))



def regression_model():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    #dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])
    x = np.asarray(boston)
    y = np.asarray(boston_dataset.target)


    xmean = np.mean(x, axis=0)
    xscale = np.amax(x, axis=0) - np.amin(x, axis=0)
    xscale[xscale < np.finfo(float).eps] = 1.0
    x -= xmean
    x /= xscale


    y = y.reshape(-1, 1)
    ymean = np.mean(y, axis=0)
    yscale = np.amax(y, axis=0) - np.amin(y, axis=0)
    yscale[yscale < np.finfo(float).eps] = 1.0
    y -= ymean
    y /= yscale

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    x_train = torch.tensor(x_train, dtype=torch.float, device="cpu")
    x_test = torch.tensor(x_test, dtype=torch.float, device="cpu")
    y_train = torch.tensor(y_train, dtype=torch.float, device="cpu")
    y_test = torch.tensor(y_test, dtype=torch.float, device="cpu")

    #first fun 16 units and use relu
    inputlayer = x.shape[1]
    layer1 = 16
    layer2 = 32
    outputlinear = 1

    model = torch.nn.Sequential(
        torch.nn.Linear(inputlayer,  layer1),
        torch.nn.ReLU(),
        torch.nn.Linear(layer1, layer2),
        torch.nn.Tanh(),
        torch.nn.Linear(layer2, outputlinear)
    ).to("cpu")
    Opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    LossFun = torch.nn.MSELoss(reduction="sum")

    for i in range(4000):
        y_pred = model(x_train)
        loss = LossFun(y_pred, y_train)
        if (i % 1000) == 0:
            print("when ", i, " Loss: ", loss.item())
        Opt.zero_grad()
        loss.backward()
        Opt.step()
    y_pred = model(x_test)
    loss = LossFun(y_pred, y_test)
    print("MSE is:", loss.item())



if __name__ == "__main__":
    print("regression model")
    regression_model()
    print("")
    print("classification_model")
    classification_model()

