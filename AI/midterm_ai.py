import random

def gradient_descent(x, y):
    theta = [1, 1]
    loss = 10
    learning_rate = 0.01
    eps = 0.0001
    max_iterations = 10000
    iter_count = 0
    err1 = [0, 0, 0, 0]
    err2 = [0, 0, 0, 0]

    while loss > eps and iter_count < max_iterations:
        loss = 0
        err1sum = 0
        err2sum = 0

        for i in range(len(x)):
            prediction = theta[0]*x[i][0] + theta[1]*x[i][1]
            err1[i] = (prediction - y[i]) * x[i][0]
            err1sum += err1[i]
            err2[i] = (prediction - y[i]) * x[i][1]
            err2sum += err2[i]
        theta[0] = theta[0] - learning_rate * err1sum/4
        theta[1] = theta[1] - learning_rate * err2sum/4

        for i in range(len(x)):
            prediction = theta[0] * x[i][0] + theta[1] * x[i][1]
            error = (1/(len(x[0]) * (len(x))) * (prediction - y[i])) ** 2
            loss = loss + error
        iter_count += 1

    print("GD: theta is", theta)

def stochastic_gradient_descent(x, y):
    theta = [1, 1]
    loss = 10
    learning_rate = 0.01
    eps = 0.0001
    max_iterations = 10000
    iter_count = 0
    err1 = [0, 0, 0, 0]
    err2 = [0, 0, 0, 0]

    while loss > eps and iter_count < max_iterations:
        loss = 0
        i = random.randint(0, 3)
        prediction = theta[0] * x[i][0] + theta[1] * x[i][1]
        theta[0] = theta[0] - learning_rate * (prediction - y[i]) * x[i][0]
        theta[1] = theta[1] - learning_rate * (prediction - y[i]) * x[i][1]

        for i in range(len(x)):
            prediction = theta[0] * x[i][0] + theta[1] * x[i][1]
            error = ((1 / 2) * (prediction - y[i])) ** 2
            loss = loss + error
        iter_count += 1

    print("SGD: theta is", theta)

def mini_batch_gradient_descent(x, y):
    theta = [1, 1]
    loss = 10
    learning_rate = 0.01
    eps = 0.0001
    max_iterations = 10000
    iter_count = 0
    err1 = [0, 0, 0, 0]
    err2 = [0, 0, 0, 0]

    while loss > eps and iter_count < max_iterations:
        loss = 0
        i = random.randint(0,3)
        prediction1 = theta[0] * x[i][0] + theta[1] * x[i][1]
        j = (i + 1) % 4
        prediction2 = theta[0] * x[j][0] + theta[1] * x[j][1]

        theta[0] = theta[0] - learning_rate * 1 / 2 * ((prediction1 - y[i]) * x[i][0] + (prediction2 - y[j]) * x[j][0])
        theta[1] = theta[1] - learning_rate * 1 / 2 * ((prediction1 - y[i]) * x[i][1] + (prediction2 - y[j]) * x[j][1])

        for i in range(len(x)):
            prediction = theta[0] * x[i][0] + theta[1] * x[i][1]
            error = ((1 / 2) * (prediction - y[i])) ** 2
            loss = loss + error
        iter_count += 1

    print("MBGD: theta is ", theta)

def adagrad_alg(x, y):
    theta = [1, 1]
    loss = 10
    learning_rate = 0.01
    eps = 0.0001
    max_iterations = 10000
    iter_count = 0
    err1 = [0, 0, 0, 0]
    err2 = [0, 0, 0, 0]



if __name__ == "__main__":
    x = [[1,4], [2,5], [5,1], [4,2]]
    y = [19,26,19,20]
    gradient_descent(x, y)
    stochastic_gradient_descent(x, y)
    mini_batch_gradient_descent(x, y)