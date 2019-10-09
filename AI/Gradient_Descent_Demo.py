from numpy import np

def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x+b)) ** 2
    return totalError

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points)):
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i ,1]
        b_gradient += -(2/N)*(y - ((m_current * x)+ b_current))
        m_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, staring_m, learning_rate, num_iteration):
    b = starting_b
    m = staring_m

    for i in range(num_iteration):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b,m]

def run():
    points = genfromtest("data/csv", delimiter=',')
    #hyperparameters
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0
    initial_m = 0

    num_iterations = 1000
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(b)
    print(m)


if __name__ == "__main__":
    run()