import numpy as np
import timeit

def brownian_path():
    x0 = np.random.randn()
    x = np.zeros(100000)
    v = np.random.randn(100000)
    x[0] = x0
    a = 0.9999
    b = 0.1
    for i in range(100000-1):
        x[i+1] = a*x[i] + b*v[i]
    return x

print(timeit.timeit("brownian_path()",
                    globals=locals(), 
                    number=1000)/1000)




