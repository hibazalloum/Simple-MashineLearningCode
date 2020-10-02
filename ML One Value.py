# Regression for One value
# import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data to the project ..
data = pd.read_csv('data.txt', header=None, names=['Population', 'Income'])

# read details data
print(data.head(21))
print('__________________________________')
print(data.describe())
print('__________________________________')

# presentation data
data.plot(kind='scatter', x='Population', y='Income', figsize=(4, 4))
plt.show()

# Gradient Descent
# Tips to calculate COST ERROR

# Tip 1 : adding column before data

# add Ones before columns
data.insert(0, 'AddOnes', 1)
print(data.head(21))
print('____________________________________')

# Tip 2 : Separate input data from output data
col = data.shape[1]
# Take all the rows & columns from zero to -1
X = data.iloc[:, 0:col - 1]  # columns index 0(addOnes),1(population)
y = data.iloc[:, col - 1:col]  # columns index 2(income)
print(X.head(21))
print('____________________________________')
print(y.head(21))
print('____________________________________')

# Tip 3: convert data to Matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))


# Tip 4: calculate Cost Error
def calCostError(X, y, theta):
    a = np.power(((X * theta.T) - y), 2)
    # print(a)
    # print(len(X))
    return np.sum(a) / (2 * len(X))


print(calCostError(X, y, theta))


# Gradient descents
def GradDesc(X, y, theta, alpha, iteration):
    temporary = np.matrix(np.zeros(theta.shape))  # make a temporyary matriex zeros
    para = int(theta.ravel().shape[1])
    cost = np.zeros(iteration)

    for i in range(iteration):
        error = (X * theta.T) - y

        for j in range(para):
            term = np.multiply(error, X[:, j])
            temporary[0, j] = theta[0, j] - (alpha / len(X)) * np.sum(term)

        theta = temporary
        cost[i] = calCostError(X, y, theta)

    return theta, cost


# make initial values for iteration
alpha = 0.014
iterations = 800
a, cost = GradDesc(X, y, theta, alpha, iterations)

# best fit line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
print(x)
print(a)

bestFit = a[0, 0] + (a[0, 1] * x)
print(bestFit)

# Finally Drawss ...
# Draw the line
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(x, bestFit, label='prediction')
ax.scatter(data.Population, data.Income,label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Income')
ax.set_title('Relationship Between Income & Population')
plt.show()

# Draw the Error
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(iterations), cost)
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Percentage of Error')
plt.show()
