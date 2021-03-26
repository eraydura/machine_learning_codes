import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

def coeffients(x, y):
    n = num.size(x)
    m_x, m_y = num.mean(x), num.mean(y)

    SS_xy = num.sum(y * x) - n * m_y * m_x
    SS_xx = num.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)

def draw(x1, y1, b2, x2, y2, b1):
    plt.figure(1)
    plt.scatter(x1, y1, color="red", marker="o", s=30)
    y_pred1 = b2[0] + b2[1] * x1
    plt.plot(x1, y_pred1, color="g")
    plt.xlabel("Age")
    plt.ylabel("Experience")

    plt.figure(2)
    plt.scatter(x2, y2, color="red", marker="o", s=30)
    y_pred2 = b1[0] + b1[1] * x2
    plt.plot(x2, y_pred2, color="g")
    plt.xlabel("Age")
    plt.ylabel("Experience")

    plt.show()

def rsquare(x, y, b):
    y_pred = b[0] + b[1] * x
    rs = num.sum((y - y_pred) ** 2)
    tss = num.sum((y - num.mean(y)) **2)
    print(1 - rs / tss)

x1 = num.array([])
y1 = num.array([])

x2 = num.array([])
y2 = num.array([])

dataset = pd.read_csv('team_1.csv', encoding='Latin')
dataset2 = pd.read_csv('team_2.csv', encoding='Latin')
x1 = dataset['Age'].values.reshape(-1,1)
y1 = dataset['Experience'].values.reshape(-1,1)
x2 = dataset2['Age'].values.reshape(-1,1)
y2 = dataset2['Experience'].values.reshape(-1,1)

b1 = coeffients(x1, y1)
b2 = coeffients(x2, y2)

draw(x1, y1, b2, x2, y2, b1)

rsquare(x1, y1, b2)
rsquare(x2, y2, b1)
