# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:12:01 2019

@author: Aman Nirala

Github: http://www.github.com/amannirala13
LinkedIn: http://www.linkedin/in/amannirala13
Facebook: http://www.facebook.com/amannirala13
Instagram: http://www.instagram.com/amannirala13
Twitter: http://www.twitter.com/amannirala13

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("CAW.csv")

Y = dataset.iloc[11:12,1:].values
Y_s = np.reshape(Y,(12,1))
X = pd.DataFrame({"2001":[2001], "2002":[2002], "2003":[2003], "2004":[2004], "2005":[2005], "2006":[2006], "2007":[2007], "2008":[2008], "2009":[2009], "2010":[2010], "2011":[2011], "2012":[2012]}).values;
X_s = np.reshape(X,(12,1))

regressor = LinearRegression()
regressor.fit(X_s, Y_s)

X_p = pd.DataFrame({"2014":[2014]})
Y_p = regressor.predict(X_p)

Y_a= np.array([337922])

plt.plot(X_s, Y_s, color = "blue", label="Total no of cases recorded")
plt.plot(X_s, regressor.predict(X_s), color = "grey", alpha = 0.5, label = "Line of regression")
plt.scatter(X_p, Y_p, color = "black", label = "Predicted crime rate in 2014")
plt.scatter(X_p, Y_a, color = "red", label = "Actual crime rate in 2014")
plt.grid(True)
plt.title("Crime rate against Women in India")
plt.xlabel("Years (2001 - 2014)")
plt.ylabel("Criminal cases registered")
plt.legend(loc="best")
plt.show()

print("Predicted crime rate in ",int(X_p.values[0]), " : ", int(Y_p[0]))
print("Actual crime rate in = ",int(X_p.values[0])," : ", int(Y_a))
print("Difference (Actual - Predicted) = ",int(Y_a - Y_p[0]))