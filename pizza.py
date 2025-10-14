# Same as before, but with gradient descent implementation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

print("X: \n ",X)
print("Y: \n ",Y)

sns.set()
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Reservations", fontsize=32)
plt.ylabel("Pizzas", fontsize=32)
plt.plot(X, Y, "bo")
plt.show()

def predict(X, w, b=0):
    return X * w + b

y_hat = predict(20, 2.1, 0)
print("num pizzas: ", y_hat)

def loss(X, Y, w, b=0):
    return np.average((predict(X,w,b)-Y)**2)

def train(X,Y, iterations, lr):
    w = 0
    for i in range (iterations):
        current_loss = loss(X, Y, w)
        print("Iterations %4d => Loss: %.6f" % (i, current_loss))
        if loss(X,Y, w+lr) < current_loss:
            w += lr
        elif loss(X,Y, w-lr) < current_loss:
            w -= lr
        else:
            return w
    raise Exception("Couldn't cohnverge within %d iterations" % iterations)

w = train(X,Y, 10000, 0.01)
print("\nw =%.3f" % w)
print("Prediction: x=%d => y=%.2f" % (20, predict(20,w)))

def train_b(X,Y, iterations, lr):
    w = b = 0
    for i in range (iterations):
        current_loss = loss(X, Y, w, b)
        print("Iterations %4d => Loss: %.6f" % (i, current_loss))
        if loss(X,Y, w+lr, b) < current_loss:
            w += lr
        elif loss(X,Y, w-lr, b) < current_loss:
            w -= lr
        elif loss(X,Y, w, b+lr) < current_loss:
            b += lr
        elif loss(X,Y,w,b-lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Couldn't converge within %d iterations" % iteration)

# def gradient(X,Y,w):
#     return 2*np.average(X*(predict(X,w,0)-Y))
