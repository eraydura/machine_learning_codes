import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

ageList = num.array([])
expList = num.array([])
powList = num.array([])
salaryList = num.array([])

dataset = pd.read_csv('teams_comb.csv', encoding='Latin')

age = num.append(ageList, dataset['Age'].values.reshape(-1,1))
exp = num.append(expList, dataset['Experience'].values.reshape(-1,1))
pow = num.append(powList, dataset['Power'].values.reshape(-1,1))
salary = num.append(salaryList,dataset['Salary'].values.reshape(-1,1))

def original():
      X = num.vstack((num.ones((1, len(age))))).T
      return X

def best():
    X = num.vstack((num.ones((1, len(age))), pow)).T
    return X

Y = salary.T

def values(X, Y):
       a= num.linalg.inv(num.dot(X.T, X))
       a = num.dot(a, X.T)
       a = num.dot(a, Y)
       return a

def rsquare(x, y, b,d):
    y_pred = b[0] + b[1] * x
    rs = num.sum((y - y_pred) ** 2)
    tss = num.sum((y - num.mean(y)) **2)
    tss=tss/(y_pred.size-1)
    rs=rs/(y_pred.size-d-1)
    if rs == 0 or tss == 0:
        result = 0
    else:
        result= 1 - rs / tss
    print(result)

def coeffients(x, y):
    n = num.size(x)
    m_x, m_y = num.mean(x), num.mean(y)

    SS_xy = num.sum(y * x) - n * m_y * m_x
    SS_xx = num.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)



Y_temp_ori = num.dot(original(), values(original(), Y))
Y_temp_best = num.dot(best(), values(best(), Y))
print("Showing original r square value" )
rsquare(Y, Y_temp_ori, coeffients(Y, Y_temp_ori),0)
print("Showing best r square value" )
rsquare(Y, Y_temp_best, coeffients(Y, Y_temp_best),1)