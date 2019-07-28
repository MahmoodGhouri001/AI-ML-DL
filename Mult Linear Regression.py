import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

%matplotlib inline

dataset = pd.read_csv('./petrol consumption.csv')  
dataset.head() 
dataset.describe()

X = dataset[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_license' ]]
y = dataset['Petrol_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

regressor = LinearRegression(normalize='True')  
regressor.fit(X_train, y_train) 

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
print(coeff_df)

y_pred = regressor.predict(X_train) 

#df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})  
#print(df)

print(r2_score(y_train, y_pred))




y_pred = regressor.predict(X_test) 

#df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
#print(df)

print(r2_score(y_test, y_pred))

