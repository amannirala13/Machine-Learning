# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:01:16 2019

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#__________________________________________DATA PREPROCESSING__________________________________________

#You can use the data preprocessing template too for data preprocessing

# >>> Importing Dataset

dataset = pd.read_csv("Salary_data.csv")

#preparing attributes and dependent values
X = dataset.iloc[:, :-1].values   #attributes
Y = dataset.iloc[:, 1].values   #dependent values 

# >>> Spliting data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)    #Splitting the value into training and testing dataset

#No need of feature scaling in Linear Regression


#__________________________________________LINEAR REGRESSION__________________________________________

# >>> Fitiing the model

regressor = LinearRegression()       #creating an instance of LinearRegressio class
regressor.fit(X_train, Y_train)        #Fitting the Linear Regression model to the training data

# >>> Creating the predicted data set of Y

Y_pred = regressor.predict(X_test)         #Getting the predicted value of Y from the test data


#__________________________________________VISUALIZATION__________________________________________

# >>> Plotting regression line with training dataset

plt.scatter(X_train, Y_train, color = "red")                          #plotting the training data points
plt.plot(X_train, regressor.predict(X_train), color = "blue")             #plotting regression line
plt.title("Experience vs Salary (Training set)")                 # setting title of the table
plt.xlabel("Experience")                            #setting label of X - axis
plt.ylabel("Salary")                            #setting label of Y - axis
plt.show()                  #showing the graph window

# >>> Plotting regression line with testing dataset

plt.scatter(X_test, Y_test, color = "red")                          #plotting the testing data points
plt.plot(X_train, regressor.predict(X_train), color = "blue")             #plotting regression line
plt.title("Experience vs Salary (Testing set)")                 # setting title of the table
plt.xlabel("Experience")                            #setting label of X - axis
plt.ylabel("Salary")                            #setting label of Y - axis
plt.show()                  #showing the graph window