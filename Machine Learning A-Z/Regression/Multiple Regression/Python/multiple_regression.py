# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:18:39 2019

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
import statsmodels.regression.linear_model as sml
import matplotlib.pyplot as plt


#__________________________________________DATA PREPROCESSING__________________________________________

#You can use the data preprocessing template too for data preprocessing

# >>> Importing Dataset
dataset = pd.read_csv("50_Startups.csv")

# >>> Seperating the independent and dependent variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# >>> One Hot Encoding categorical data
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X = OneHotEncoder(categorical_features = [3]).fit_transform(X).toarray()

# >>> Removing Dummy Variable
X = X[:, 1:]

# >>> Splitting training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


#__________________________________________LINEAR REGRESSION__________________________________________

# >>> Fitiing the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# >>> Predicting the value
Y_pred = regressor.predict(X_test)


#__________________________________________MULTIPLE REGRESSION (BACKWAED ELEMINATION)__________________________________________


# >>> Creating an optimal model by added the cofficient for the constant
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# >>> Performing Backward Elimination
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sml.OLS(endog=Y, exog=X_opt).fit()
#print(regressor_OLS.summary())

#Eleminating the attr with highest p-value and fitting the model again
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sml.OLS(endog=Y, exog=X_opt).fit()
#print(regressor_OLS.summary())

#Eleminating the attr with highest p-value and fitting the model again
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sml.OLS(endog=Y, exog=X_opt).fit()
#print(regressor_OLS.summary())

#Eleminating the attr with highest p-value and fitting the model again
X_opt = X[:, [0,3,5]]
regressor_OLS = sml.OLS(endog=Y, exog=X_opt).fit()
#print(regressor_OLS.summary())

#Eleminating the attr with highest p-value and fitting the model again
X_opt = X[:, [0,3]]
regressor_OLS = sml.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())


#__________________________________________VISUALIZATION__________________________________________

plt.scatter(X_train[:, 3], Y_train, color = 'black')
plt.scatter(X_train[:, 3],regressor.predict(X_train), color = "red")
plt.title("TRAINING DATASET")
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.show()

plt.scatter(X_test[:, 3], Y_test, color = 'black')
plt.scatter(X_test[:, 3], Y_pred, color = 'red')
plt.title("TESTING DATASET")
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.show()