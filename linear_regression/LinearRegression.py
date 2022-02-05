# Import Libraries
import numpy  as np
import random as rd
import pandas as pd
from matplotlib import pyplot

Orim = 1.4
Oric = 0.3
# Load Datasets
df = pd.read_csv('Data.csv')
X = df['x'].to_numpy()
Y = df['y'].to_numpy()

# plot regression line for training data
pyplot.scatter(X, Y)
pyplot.plot([min(X), max(X)], [min(Y), max(Y)], color='red')  # regression line
pyplot.show()


# Function to calculate mean squared error
def LossFunc(m, c):
    return sum((Y - (m * X + c)) ** 2)


# function to find Optimal m and c
def FindMC(X, Y):
    Alpha = 0.0001
    h = .2
    m = rd.uniform(-5, 5)  # random floating number between -10 and 10
    c = rd.uniform(-5, 5)
    OldDm = 0.0
    oldDc = 0.0
    iter = 100
    for i in range(iter):
        Dm = LossFunc(m + h, c) - LossFunc(m, c)
        Dm /= h

        Dc = LossFunc(m, h + c) - LossFunc(m, c)
        Dc /= h

        if LossFunc(m, c) > LossFunc(m - Alpha * Dm, c - Alpha * Dc):
            m = m - Alpha * Dm
            c = c - Alpha * Dc
        if (i != 0):
            if ((Dm > 0 and OldDm < 0) or (Dm < 0 and OldDm > 0)) or ((Dc > 0 and OldDc < 0) or (Dc < 0 and OldDc > 0)):
                Alpha /= 2.0
        OldDm = Dm
        OldDc = Dc
    return m, c


m, c = FindMC(X, Y)


def Predict(x):
    return (m * x + c)


print("Original m =", Orim)
print("Predicted m =", m)

print("Original c ", Oric)
print("Predicted c =", c)
Y_pred = m * X + c

print("For x= 3 Original Y is ", 3.0*Orim+Oric)
print("Predicted Y is ", Predict(3.0))

print("For x= 6 Original Y is ", 6.0*Orim+Oric)
print("Predicted Y is ", Predict(6.0))



# plot predicted vaules of Y
pyplot.scatter(X, Y)
pyplot.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
pyplot.show()
