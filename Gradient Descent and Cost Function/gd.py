import numpy as np

def gradientDescent(x,y) :
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learningRate = 0.08
    for i in range(iterations) :
        yp = m_curr * x + b_curr
        cost = (1 / n) * sum([val**2 for val in (y-yp)])
        md = - (2 / n) * sum(x * (y - yp))  #partial derivative
        bd = - (2 / n) * sum(y - yp) #partial derivative
        m_curr = m_curr - learningRate * md
        b_curr = b_curr - learningRate * bd
        print("m {}, b {}, cost {}, itration {}".format(m_curr, b_curr, cost, i))
pass

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradientDescent(x, y)