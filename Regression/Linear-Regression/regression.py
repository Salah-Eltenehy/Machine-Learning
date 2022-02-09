#import libraries
import pandas as pd
from matplotlib import pyplot as plt
#read data
data = pd.read_csv('F:/AI/regression/P14-Part2-Regression/Section 6 - Simple Linear Regression/Python/Salary_Data.csv')
#Vislaisation
plt.scatter(data.iloc[:, 0].values, data.iloc[:, 1].values, color= 'blue')
plt.title('Original Data')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()
#Independent  &  dependnt variables
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1:2].values
#Scaling the data
from sklearn.preprocessing import StandardScaler
"""sc_X = StandardScaler()
sc_X.fit(X)
X = sc_X.transform(X)
"""
sc_y = StandardScaler()
sc_y.fit(y)
y = sc_y.transform(y)
#Split the data into training set and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, 
                                                    random_state =0
                                                    )
#Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict the data
y_pred = regressor.predict(X_test)
#Check accuarcy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#test set
r2_test = r2_score(y_test, y_pred)
meanSquareTest = mean_squared_error(y_test, y_pred)
meanAbsoulteTest = mean_absolute_error(y_test, y_pred)
#training set
r2_train = r2_score(y_train, regressor.predict(X_train))
meanSquareTrain = mean_squared_error(y_train, regressor.predict(X_train))
meanAbsoulteTrain = mean_absolute_error(y_train, regressor.predict(X_train))
#Vislaisation test set
plt.scatter(X_test, y_test, color = 'b')
plt.plot(X_test, y_pred, color= 'r')
plt.title('Test set')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()
#Vislaisation training set
plt.scatter(X_train, y_train, color = 'b')
plt.plot(X_train, regressor.predict(X_train), color= 'r')
plt.title('Training set')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show()