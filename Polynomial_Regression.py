# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:17:01 2022

@author: 47483
"""

# Polynomial Regression

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Import the dataset
dataset = pd.read_csv(r'C:\Study\Data Science\Positions_vs_Salaries.csv')

#Independent variable X (Job position level) 
X = dataset.iloc[:, 1:-1].values

#Dependent variable y (Salary)
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the dataset
lin_regression = LinearRegression()
lin_regression.fit(X, y)

# Training the Polynomial Regression model on the dataset
"""A polynomial regression is a form of regression analysis 
in which the relationship between the independent variable X (Job position level) 
and the dependent variable y (Salary) is modelled
 as an nth degree polynomial in X (Job position level).
"""
poly_regression = PolynomialFeatures(degree = 4)
X_nth_degree_poly = poly_regression.fit_transform(X)
lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_nth_degree_poly, y)

# Visualising Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression.predict(X), color = 'green')
plt.title('Linear Regression')
plt.xlabel('Job Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising  Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_regression_2.predict(X_nth_degree_poly), color = 'green')
plt.title('Polynomial Regression')
plt.xlabel('Job Position Level')
plt.ylabel('Salary')
plt.show()

