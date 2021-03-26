import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

ageList = num.array([])
expList = num.array([])
powList = num.array([])
salaryList = num.array([])

dataset = pd.read_csv('team.csv', encoding='Latin')

age = num.append(ageList, dataset['Age'].values.reshape(-1,1))
exp = num.append(expList, dataset['Experience'].values.reshape(-1,1))
pow = num.append(powList, dataset['Power'].values.reshape(-1,1))
salary = num.append(salaryList,dataset['Salary'].values.reshape(-1,1))

X = num.vstack((num.ones((1, len(age))), age, exp, pow)).T
Y = salary.T

def values(X, Y):
       a= num.linalg.inv(num.dot(X.T, X))
       a = num.dot(a, X.T)
       a = num.dot(a, Y)
       return a

print(values(X,Y))


Y_temp = num.dot(X, values(X,Y))
m_error=abs(Y_temp-Y)
plt.scatter(Y_temp, m_error)
plt.show()

def rsquare(x, y, b):
    y_pred = b[0] + b[1] * x
    rs = num.sum((y - y_pred) ** 2)
    tss = num.sum((y - num.mean(y)) **2)
    print(1 - rs / tss)

def coeffients(x, y):
    n = num.size(x)
    m_x, m_y = num.mean(x), num.mean(y)

    SS_xy = num.sum(y * x) - n * m_y * m_x
    SS_xx = num.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)

print("Showing original r square value" )
rsquare(Y, Y_temp, coeffients(Y, Y_temp))
print("Showing random r square value" )
random=num.random.randint(-500, 500, len(Y))
rsquare(Y, random, coeffients(Y, random))