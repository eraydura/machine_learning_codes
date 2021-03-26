import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1')
data = dataset.sort_values("FSP.1")
x = data['FSP.1'].values
y = data['FSW.1'].values

i = x.argsort()
x_1 = x[i]
y_1 = y[i]

bi = np.array([x_1]).transpose()
b2 = np.power(bi, 2)
b3 = np.power(bi, 3)

knot_array= [[55,65,70],[60,75],[62]]
color=["black","green","red"]

def knots(knots,color):

    columns = []
    for knot in knots:
        array = []
        for x in np.nditer(bi):
            res = x - knot
            if (res < 0):
                res = 0
            array.append(res)
        col = np.array([array]).transpose().__pow__(3)
        columns.append(col)
    plot(columns, color)

def plot(columns,color):

    if (columns.__len__() == 1):
        X_spline = np.hstack((np.ones((x.size, 1)), bi, b2, b3, columns[0]))
    elif (columns.__len__() == 2):
        X_spline = np.hstack((np.ones((x.size, 1)), bi, b2, b3, columns[0], columns[1]))
    elif (columns.__len__() == 3):
        X_spline = np.hstack((np.ones((x.size, 1)), bi, b2, b3, columns[0], columns[1], columns[2]))

    X_t = X_spline.transpose()

    B1 = np.dot(X_t, X_spline)
    B2 = np.linalg.inv(B1).dot(X_t)
    B3 = np.dot(B2, y_1)

    regression = X_spline.dot(B3)

    plt.scatter(x_1, y_1, color="blue")
    plt.xlabel("First serve won by player 1")
    plt.ylabel("First serve percentage of player 1")
    plt.plot(x_1, regression, color=color)
    plt.legend(["3 knots", "2 knots","1 knot"])



def c_spline():
    knots(knot_array[0],color[0])
    knots(knot_array[1] ,color[1])
    knots(knot_array[2],color[2])


c_spline()
plt.show()