import pandas as pandas
import matplotlib.pyplot as pilot
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

datas = pandas.read_csv('Grand-slams-men-2013 (4).csv', encoding='latin1')

x_datas = datas[['FSP.1','ACE.1','DBF.1','WNR.1','UFE.1','BPC.1','NPA.1']]
y_datas = datas[["ST1.1", "ST2.1", "ST3.1", "ST4.1", "ST5.1"]].sum(axis=1)

x_Train,x_Test,y_Train,y_Test = x_datas[:200],x_datas[200:],y_datas[:200],y_datas[200:]

array_1,array_2,array_3=[],[],[]

for i in range (1,151):
    randomforest_1 = RandomForestRegressor(max_depth=7,n_estimators = i,max_features="auto")
    randomforest_1.fit(x_Train, y_Train)
    y_pred_1 = randomforest_1.predict(x_Test)
    r2_score_1=metrics.r2_score(y_Test, y_pred_1)
    array_1.append(r2_score_1)

    randomforest_2 = RandomForestRegressor(max_depth=7,n_estimators = i,max_features="sqrt")
    randomforest_2.fit(x_Train, y_Train)
    y_pred_2 = randomforest_2.predict(x_Test)
    r2_score_2=metrics.r2_score(y_Test, y_pred_2)
    array_2.append(r2_score_2)

    randomforest_3 = RandomForestRegressor(max_depth=7,n_estimators = i,max_features=4)
    randomforest_3.fit(x_Train, y_Train)
    y_pred_3 = randomforest_3.predict(x_Test)
    r2_score_3=metrics.r2_score(y_Test, y_pred_3)
    array_3.append(r2_score_3)

randomforest_4 = RandomForestRegressor(max_depth=7,n_estimators=150,max_features=4)
randomforest_4.fit(x_Train, y_Train)
pred = randomforest_4.predict(x_Test)

randomforest_5 = RandomForestRegressor(max_depth=1,n_estimators=150,max_features=4)
randomforest_5.fit(x_Train, y_Train)
pred_2 = randomforest_5.predict(x_Test)


pilot.figure()
pilot.plot(np.arange(1,151),array_1,c='g')
pilot.plot(np.arange(1,151),array_2,c='b')
pilot.plot(np.arange(1,151),array_3,c='r')
pilot.figure()
pilot.scatter(pred,y_Test - pred)
pilot.scatter(pred_2,y_Test - pred_2)
pilot.hlines(y = 0, xmin = 0, xmax = 35, linewidth = 2)
pilot.show()

