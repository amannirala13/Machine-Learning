# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:27:46 2019

@author: Aman Nirala

Github: http://www.github.com/amannirala13
LinkedIn: http://www.linkedin/in/amannirala13
Facebook: http://www.facebook.com/amannirala13
Instagram: http://www.instagram.com/amannirala13
Twitter: http://www.twitter.com/amannirala13

"""

# >>> Importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#__________________________________________DATA PREPROCESSING__________________________________________

#You can use the data preprocessing template too for data preprocessing

# >>> Importing Dataset

dataset = pd.read_csv("Position_Salaries.csv")

#preparing attributes and dependent values
X = dataset.iloc[:, 1:2].values   #attributes
Y = dataset.iloc[:, 2:3].values   #dependent values


#__________________________________________POLYNOMIAL REGRESSION__________________________________________

# >>> Fitting first linear regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# >>> Fitting Polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

# >>> Fitting second linear regression to the polynomial dataset
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)


#__________________________________________VISUALIZATION__________________________________________

# >>> Linear regression plot
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X))
plt.title("Linear Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# >>> Polynomial plot (High resolution)
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)))
plt.title("Polynomial Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# >>> Polynomial plot (High resolution)
X_high_res = np.arange(min(X), max(X), 0.1)
X_high_res = X_high_res.reshape(len(X_high_res), 1)
plt.scatter(X, Y, color = "red")
plt.plot(X_high_res, lin_reg2.predict(poly_reg.fit_transform(X_high_res)))
plt.title("Polynomial Regression High Resolution")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#__________________________________________PREDICTION__________________________________________

print("\n\n____________________~ PREDICTION ~____________________\n\n")

# >>> Linear regression
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "grey")
plt.scatter(pd.DataFrame(data = {6.5}), lin_reg.predict(pd.DataFrame(data = {6.5})), color = "black")
plt.title("Linear Regression Prediction")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

# >>> Polynomial regression
plt.scatter(X, Y, color = "red")
plt.plot(X_high_res, lin_reg2.predict(poly_reg.fit_transform(X_high_res)), color = "grey")
plt.scatter(pd.DataFrame(data = {6.5}), lin_reg2.predict(poly_reg.fit_transform(pd.DataFrame(data = {6.5}))), color = "black")
plt.title("Polynomial Regression (High Resolution) Prediction")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()