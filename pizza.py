# Same as before, but with gradient descent implementation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
# w = train(X,Y, iterations=100, lr=0.001)
# print("\nw=%.10f" % w)

# print("X: \n ",X)
# print("Y: \n ",Y)

def predict(X, w, b):
    return X * w + b

def loss(X, Y, w, b=0):
    return np.average((predict(X,w,b)-Y)**2)


def train(X,Y, iterations, lr):
    w = b = 0
    for i in range (iterations):
        # current_loss = loss(X, Y, w, b)
        # print("Iterations %4d => Loss: %.6f" % (i, current_loss))
        print("Iteration %4d => Loss: %.20f" % (i, loss(X,Y,w,b)))
        w_gradient, b_gradient = gradient(X,Y,w,b)
        w -= w_gradient * lr
        b -= b_gradient * lr
    return w, b

def gradient(X,Y, w,b):
    w_gradient = 2 * np.average(X * (predict(X, w, b) - Y))
    b_gradient = 2 * np.average(predict(X,w,b)-Y)
    return (w_gradient, b_gradient)

X,Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
w,b = train(X,Y, iterations=20000, lr=0.001)
print("\nw=%.10f, b=%.10f" % (w,b))
print("Prediction: x=%d => y=%.2f" % (20, predict(20,w,b)))

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Reservations", fontsize=32)
plt.ylabel("Pizzas", fontsize=32)
plt.plot(X, Y, "bo")
plt.show()