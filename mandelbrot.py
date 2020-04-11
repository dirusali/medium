import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-2,2,1000)
y = np.linspace(-1,1,1000)
conjunto = []
x_axis = []
y_axis = []

def mandel(c):
    z = 0
    for i in range(0,10):
        z = z**2 + c
    return z

def set_mandel():
    for i in x:
        for z in y:
            a = complex(i+ z*((-1)**(1/2)))
            m = mandel(a)
            m = abs(m)
            if m < 2:
                ejex.append(i)
                ejey.append(z)
