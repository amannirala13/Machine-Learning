# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:17:32 2019

@author: Aman Nirala

Github: http://www.github.com/amannirala13
LinkedIn: http://www.linkedin/in/amannirala13
Facebook: http://www.facebook.com/amannirala13
Instagram: http://www.instagram.com/amannirala13
Twitter: http://www.twitter.com/amannirala13

"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
""" from sklearn.preprocessing import Imputer """                                                #If using sklearn version -0.20
from sklearn.impute import SimpleImputer as imp                      #If using Sklearn version 0.20+
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# >>> Importing dataset


dataset = pd.read_csv('data.csv')                    #reading data from the csv file

X = dataset.iloc[:, :-1].values                  #storing the attributes
Y = dataset.iloc[:, 3].values               #storing the value to be predicted or dependent variable


# >>> Handeling missing data


X[:, 1:3] = imp(missing_values = np.nan , strategy = 'mean').fit_transform(X[:, 1:3])          #If using Sklearn version 0.20+

#If using sklearn version -0.20
"""
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)         # Initialising the Impute object (it calls __inti__ function of imputter)
imputer = imputer.fit(X[:, 1:3])                          # Fitting the imputer value to Nan from column index 1 to 2 
X[:, 1:3] = imputer.transform(X[:, 1:3])                        #Transforming the NaN positions in original dataset
"""


# >>> Encoding caterogical data

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features = [0])
X = one_hot_encoder.fit_transform(X).toarray()

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)