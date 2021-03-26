import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

# Reading Data
dataset = pd.read_csv('team_2.csv', encoding='utf-8')
dataset2 = pd.read_csv('team_1.csv', encoding='utf-8')

# Collecting X and Y
age_list_1 = dataset['Age'].values.reshape(-1,1)
exp_list_1 = dataset['Experience'].values.reshape(-1,1)
age_list_2 = dataset2['Age'].values.reshape(-1,1)
exp_list_2 = dataset2['Experience'].values.reshape(-1,1)


# Mean X and Y
mean_x = np.mean(age_list_1)
mean_y = np.mean(exp_list_1)
mean_x2 = np.mean(age_list_2)
mean_y2 = np.mean(exp_list_2)

# Total number of values
n = len(age_list_1)
n2 = len(age_list_2)
# Using the formula to calculate m and c
numer = 0
denom = 0
numer2 = 0
denom2 = 0

for i in range(n):
    numer += (age_list_1[i] - mean_x) * (exp_list_1[i] - mean_y)
denom += (age_list_1[i] - mean_x) ** 2
m = numer / denom
c = mean_y - (m * mean_x)

for a in range(n2):
    numer2 += (age_list_2[i] - mean_x2) * (exp_list_2[i] - mean_y2)
denom2 += (age_list_2[i] - mean_x2) ** 2
m2 = numer2 / denom2
c2 = mean_y2 - (m * mean_x2)

# Print coefficients
print(m, c)
print(m2, c2)

# Plotting Values and Regression Line
max_x = np.max(age_list_2) + 100
min_x = np.min(X) - 100
max_x = np.max(age_list_2) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#52b920', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef4423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

