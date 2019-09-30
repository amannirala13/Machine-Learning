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


# >>> importing libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
""" from sklearn.preprocessing import Imputer """                                                #If using sklearn version -0.20
from sklearn.impute import SimpleImputer as imp                      #If using Sklearn version 0.20+
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# >>> Importing dataset

dataset_path = input("Enter the path of you CSV file")       #input the path of your csv file

dataset = pd.read_csv(dataset_path)                    #reading data from the csv file

X = dataset.iloc[:, :-1].values                  #storing the attributes (Change the iloc pattern according to the need and your dataset)
Y = dataset.iloc[:, 3].values               #storing the value to be predicted or dependent variable (Change the iloc pattern according to the need and your dataset)


# >>> Handeling missing data

X[:, 1:3] = imp(missing_values = np.nan , strategy = 'mean', verbose = 0).fit_transform(X[:, 1:3])          #If using Sklearn version 0.20+ (Change the iloc pattern according to the need and your dataset)

#If using sklearn version -0.20 (Change the iloc pattern accourding to the need and your dataset)
"""
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)         # Initialising the Impute object (it calls __inti__ function of imputter)
imputer = imputer.fit(X[:, 1:3])                          # Fitting the imputer value to Nan from column index 1 to 2
X[:, 1:3] = imputer.transform(X[:, 1:3])                        #Transforming the NaN positions in original dataset
"""


# >>> Encoding caterogical data

label_encoder_X = LabelEncoder()                               #creates an instance of LabelEncoder
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])                      #Fits and transforms the LabelEncoder to the first column of X (Change the column accourding to the need and your dataset)
one_hot_encoder = OneHotEncoder(categorical_features = [0])                 #Creates an instance of OneHotEncoder to one hot encode the first column of X  (Change the column number accourding to the need and your dataset)
X = one_hot_encoder.fit_transform(X).toarray()                    #Fits and transforms X using OneHotEncoder to one hot encode the column(s) of X

label_encoder_Y = LabelEncoder()                              #creates another instance of LabelEncoder
Y = label_encoder_Y.fit_transform(Y)                             #Fit and transform the LabelEncoder to the the columns of Y (Change the pattern accourding to the need and your dataset)


# >>> Splitting the data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2 , random_state = 0)   #Splits the dataset into training and testing (Change the test_size according to the need and your dataset)


# >>> Feature Scaling
# (Use this snippet only if you need to scale your dataset)
"""
stdSc = StandardScaler()                              #Created an instance of StandardScaler
X_train = stdSc.fit_transform(X_train)                         #Scaling by fitting the X_train (training) dataset
X_test = stdSc.transform(X_test)                         #Scaling X_test (testing) dataset
"""
