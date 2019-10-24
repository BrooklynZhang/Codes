import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as snsi

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report

import matplotlib.pyplot as plt

def question1():
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    boston['MEDV'] = boston_dataset.target
    X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
    Y = boston['MEDV']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_train.shape)
    #print(Y_test.shape)
    y_train_predict = lin_model.predict(X_train)
    rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
    print("train mean square " + str(rmse))
    y_test_predict = lin_model.predict(X_test)
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    print("test mean square error " + str(rmse))

def question2():
    dataset = pd.read_csv("Advertising.csv")
    predictors = ['TV', 'Radio', "Newspaper"]
    X = dataset[predictors]
    y = dataset['Sales']

    lm = LinearRegression()
    model = lm.fit(X, y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

    Y_train_predict = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, Y_train_predict))
    print("train mean square error " + str(rmse))

    y_test_predict = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, y_test_predict))
    print("test mean square error  " + str(rmse))


def question3():
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
    X = train[predictor2].copy(deep=False)
    y = train["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    logmodel = LogisticRegression(solver='liblinear')
    logmodel.fit(X_train, Y_train)

    predictions = logmodel.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, predictions))
    print(classification_report(Y_train, predictions))
    print("train mean square error" + str(rmse))

    Y_test_predictions = logmodel.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_test_predictions))
    print(classification_report(Y_test, Y_test_predictions))
    print("test mean square error " + str(rmse))

def question4():
    predictor = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Class"]
    train = pd.read_csv("pima-indians-diabetes.csv", names=predictor)
    X = train.iloc[:, 0:8].copy(deep=False)
    y = train.iloc[:, 8]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
    logmodel = LogisticRegression(solver='liblinear')
    logmodel.fit(X_train, Y_train)
    Y_train_predict = logmodel.predict(X_train)
    rmse = np.sqrt(mean_squared_error(Y_train, Y_train_predict))
    print(classification_report(Y_train, Y_train_predict))
    print("train mean square error is " + str(rmse))

    Y_test_predict = logmodel.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_test_predict))
    print(classification_report(Y_test, Y_test_predict))
    print("test mean square error is " + str(rmse))

def impute_age(cols):
    Age = cols[3]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


if __name__ == "__main__":
    print("question 1 is ===============")
    question1()
    print("question 2 is ===============")
    question2()
    print("question 3 is ===============")
    question3()
    print("question 4 is ===============")
    question4()