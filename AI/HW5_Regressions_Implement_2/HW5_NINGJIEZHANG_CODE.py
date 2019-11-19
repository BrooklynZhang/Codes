'''Ningjie Zhang, nxz190005, 2021501533 CS6364  AI'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support

def normalize_data(x):
    xmean = np.mean(x, axis=0)
    xscale = np.amax(x, axis=0) - np.amin(x, axis=0)
    xscale[xscale < np.finfo(float).eps] = 1.0
    x -= xmean
    x /= xscale
    return x, xmean, xscale

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_GD_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    y, ymean, yscale = normalize_data(y)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1) #ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()
    #def fun(a):
        #return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return -2 * np.dot(x.transpose(), (y - np.dot(x, a))) / y.shape[0]
    for i in range(100000):
        a = a - grad_fun(a) * 0.01 #learning rate
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1)) * yscale
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) * yscale \
                     + b[0] * yscale \
                     + ymean
    x = x * xscale + xmean
    y = y * yscale + ymean
    y_predict = np.dot(x, a)
    rmse = np.sqrt(np.sum((y - y_predict) ** 2) / y.shape[0])
    print("rmse is " + str(rmse))
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    print("MSE on training set is " + str(rmse))

    xtemp = np.append(np.ones(shape=(x_t.shape[0], 1)), x_t, axis=1)
    y_t_predict = np.dot(xtemp, a)
    rmse = np.sqrt(mean_squared_error(y_t, y_t_predict))
    print("MSE on test seting is " + str(rmse))


def linear_SDG_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    y, ymean, yscale = normalize_data(y)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)
    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def Grad_fun(a, xx, yy):
        return -2 * np.dot(xx.transpose(), (yy - np.dot(xx, a))) / yy.shape[0]

    for i in range(100000):
        idx = np.random.randint(x.shape[0], size=1)
        xx = x[idx, :]
        yy = y[idx, :]
        a = a - Grad_fun(a, xx, yy) * 0.01  # learning rate
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1)) * yscale
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) * yscale \
           + b[0] * yscale \
           + ymean
    x = x * xscale + xmean
    y = y * yscale + ymean
    y_predict = np.dot(x, a)
    rmse = np.sqrt(np.sum((y - y_predict) ** 2) / y.shape[0])
    print("rmse is " + str(rmse))
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    print("MSE on training set is " + str(rmse))

    xtemp = np.append(np.ones(shape=(x_t.shape[0], 1)), x_t, axis=1)
    y_t_predict = np.dot(xtemp, a)
    rmse = np.sqrt(mean_squared_error(y_t, y_t_predict))
    print("MSE on test seting is " + str(rmse))

def linear_SDGM_model(x, x_t, y, y_t):
    m = 0.9
    v = 0.0
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    y, ymean, yscale = normalize_data(y)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return -2 * np.dot(x.transpose(), (y - np.dot(x, a))) / y.shape[0]

    for i in range(100000):
        v = m * v + 0.01 * grad_fun(a)
        a = a - v

    b = a
    a = a / xscale.reshape((xscale.shape[0], 1)) * yscale
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) * yscale \
           + b[0] * yscale \
           + ymean
    x = x * xscale + xmean
    y = y * yscale + ymean
    y_predict = np.dot(x, a)
    rmse = np.sqrt(np.sum((y - y_predict) ** 2) / y.shape[0])
    print("rmse is " + str(rmse))
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    print("MSE on training set is " + str(rmse))

    xtemp = np.append(np.ones(shape=(x_t.shape[0], 1)), x_t, axis=1)
    y_t_predict = np.dot(xtemp, a)
    rmse = np.sqrt(mean_squared_error(y_t, y_t_predict))
    print("MSE on test seting is " + str(rmse))

def linear_SDGN_model(x, x_t, y, y_t):
    m = 0.9
    v = 0.0
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    y, ymean, yscale = normalize_data(y)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return -2 * np.dot(x.transpose(), (y - np.dot(x, a))) / y.shape[0]

    for i in range(100000):
        v = m*v + 0.01*grad_fun(a - m*v)
        a = a - v

    b = a
    a = a / xscale.reshape((xscale.shape[0], 1)) * yscale
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) * yscale \
           + b[0] * yscale \
           + ymean
    x = x * xscale + xmean
    y = y * yscale + ymean
    y_predict = np.dot(x, a)
    rmse = np.sqrt(np.sum((y - y_predict) ** 2) / y.shape[0])
    print("rmse is " + str(rmse))
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    print("MSE on training set is " + str(rmse))

    xtemp = np.append(np.ones(shape=(x_t.shape[0], 1)), x_t, axis=1)
    y_t_predict = np.dot(xtemp, a)
    rmse = np.sqrt(mean_squared_error(y_t, y_t_predict))
    print("MSE on test seting is " + str(rmse))


def linear_ADAGRAD_model(x, x_t, y, y_t):
    v = 0.0
    e = 1e-8

    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    y, ymean, yscale = normalize_data(y)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return -2 * np.dot(x.transpose(), (y - np.dot(x, a))) / y.shape[0]

    r = np.zeros(a.shape)
    for i in range(100000):
        r += grad_fun(a)*grad_fun(a)
        a = a - (0.01 /np.sqrt(r + e)) * grad_fun(a)

    b = a
    a = a / xscale.reshape((xscale.shape[0], 1)) * yscale
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) * yscale \
           + b[0] * yscale \
           + ymean
    x = x * xscale + xmean
    y = y * yscale + ymean
    y_predict = np.dot(x, a)
    rmse = np.sqrt(np.sum((y - y_predict) ** 2) / y.shape[0])
    print("rmse is " + str(rmse))
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    print("MSE on training set is " + str(rmse))

    xtemp = np.append(np.ones(shape=(x_t.shape[0], 1)), x_t, axis=1)
    y_t_predict = np.dot(xtemp, a)
    rmse = np.sqrt(mean_squared_error(y_t, y_t_predict))
    print("MSE on test seting is " + str(rmse))



def linear_regression_model():
    boston_dataset = load_boston()  # dict_keys(['data', 'target', 'feature_names', 'DESCR'])
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    # load the data into a pandas dataframe using pd.DataFrame

    boston['MEDV'] = boston_dataset.target
    features = ['LSTAT', 'RM']
    target = boston['MEDV']

    # X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
    # Y = boston['MEDV']

    X = np.asarray(boston[features])
    Y = np.asarray(target)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    print("------Linear The gradient descent algorithm ----------")
    linear_GD_model(x_train, x_test, y_train, y_test)
    print("------Linear The stochastic gradient descent (SGD) algorithm ----------")
    linear_SDG_model(x_train, x_test, y_train, y_test)
    print("------Linear The SGD algorithm with momentum  ----------")
    linear_SDGM_model(x_train, x_test, y_train, y_test)
    print("------Linear The SGD algorithm with Nesterov momentum  ----------")
    linear_SDGN_model(x_train, x_test, y_train, y_test)
    print("------Linear The AdaGrad algorithm ----------")
    linear_ADAGRAD_model(x_train, x_test, y_train, y_test)

def logstic_SDG_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a,xx,yy):
        return np.dot(xx.transpose(), sigmoid(np.dot(xx, a)) - yy) / yy.shape[0]

    for i in range(100000):
        idx = np.random.randint(x.shape[0], size=1)
        xx = x[idx, :]
        yy = y[idx, :]
        a = a - grad_fun(a, xx, yy) * 0.01  # learning rate
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1))
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) + b[0]
    x = x * xscale + xmean
    y_predict = sigmoid(np.dot(x, a))
    y_predict = (y_predict >= 0.5).astype(np.int)
    printmsg(y, y_predict)

def logstic_SDGM_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return np.dot(x.transpose(), sigmoid(np.dot(x, a)) - y) / y.shape[0]

    m = 0.9
    v = 0.0
    for i in range(100000):
        v = m * v + 0.01 * grad_fun(a)
        a = a - v
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1))
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) + b[0]
    x = x * xscale + xmean
    y_predict = sigmoid(np.dot(x, a))
    y_predict = (y_predict >= 0.5).astype(np.int)
    printmsg(y, y_predict)

def logstic_SDGN_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return np.dot(x.transpose(), sigmoid(np.dot(x, a)) - y) / y.shape[0]

    m = 0.9
    v = 0.0
    for i in range(100000):
        v = m * v + 0.01 * grad_fun(a - m * v)
        a = a - v
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1))
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) + b[0]
    x = x * xscale + xmean
    y_predict = sigmoid(np.dot(x, a))
    y_predict = (y_predict >= 0.5).astype(np.int)
    printmsg(y, y_predict)

def logstic_ADAGRAD_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    v = 0.0
    e = 1e-8
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return np.dot(x.transpose(), sigmoid(np.dot(x, a)) - y) / y.shape[0]

    r = np.zeros(a.shape)
    for i in range(100000):
        r += grad_fun(a) * grad_fun(a)
        a = a - (0.01 / np.sqrt(r + e)) * grad_fun(a)
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1))
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) + b[0]
    x = x * xscale + xmean
    y_predict = sigmoid(np.dot(x, a))
    y_predict = (y_predict >= 0.5).astype(np.int)
    printmsg(y, y_predict)

def logstic_GD_model(x, x_t, y, y_t):
    x = x.copy()
    y = y.copy()
    if y.shape[0] != 1:
        y = y.reshape(-1, 1)
    x, xmean, xscale = normalize_data(x)
    xmean = np.insert(xmean, 0, 0.0)
    xscale = np.insert(xscale, 0, 1.0)

    ones = np.ones(shape=(x.shape[0], 1))
    x = np.append(ones, x, axis=1)  # ax
    a = np.zeros(x.shape[1]).reshape((1, x.shape[1])).transpose()

    # def fun(a):
    # return np.sum((y - np.dot(x, a)) ** 2) / y.shape[0]
    def grad_fun(a):
        return np.dot(x.transpose(),  sigmoid(np.dot(x, a)) - y) / y.shape[0]

    for i in range(100000):
        a = a - grad_fun(a) * 0.01  # learning rate
    b = a
    a = a / xscale.reshape((xscale.shape[0], 1))
    a[0] = - np.sum(b / xscale.reshape((xscale.shape[0], 1)) * xmean.reshape(
        (xmean.shape[0], 1))) + b[0]
    x = x * xscale + xmean
    y_predict = sigmoid(np.dot(x, a))
    y_predict = (y_predict >= 0.5).astype(np.int)
    printmsg(y, y_predict)

def printmsg(y,y_predict):
    precision, recall, f1, support = precision_recall_fscore_support(y, y_predict, warn_for=())
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))
    print("Support: " + str(support))


def logstic_regression_model():
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
    X = np.asarray(train[predictor2].copy(deep=False))
    y = np.asarray(train["Survived"])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("------logstic The gradient descent algorithm ----------")
    logstic_GD_model(x_train, x_test, y_train, y_test)
    print("------logstic The stochastic gradient descent (SGD) algorithm ----------")
    logstic_SDG_model(x_train, x_test, y_train, y_test)
    print("------logstic The SGD algorithm with momentum  ----------")
    logstic_SDGM_model(x_train, x_test, y_train, y_test)
    print("------logstic The SGD algorithm with Nesterov momentum  ----------")
    logstic_SDGN_model(x_train, x_test, y_train, y_test)
    print("------logstic The AdaGrad algorithm ----------")
    logstic_ADAGRAD_model(x_train, x_test, y_train, y_test)






if __name__ == "__main__":
    linear_regression_model()
    logstic_regression_model()
