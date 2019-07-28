import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set()
%matplotlib inline

df = pd.read_csv("./SalaryData.csv")

df.plot.scatter(x='YearsExperience', y='Salary')
df.describe()

X = df[['YearsExperience']]
y = df[['Salary']]

plt.scatter(X, y, color='blue')

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)

salary_pred = lin_reg.predict(X)
plt.scatter(X, y, color='blue')
plt.plot(X, salary_pred, color='red', linewidth=2)

lin_reg.score(X, y)
r2_score(y, salary_pred)



######### Train and Test ###############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

print("Coefficients: ", lin_reg.coef_)
print("Intercept: ", lin_reg.intercept_)

salary_pred = lin_reg.predict(X_test)
lin_reg.score(X_test, y_test)

r2_score(y_test, salary_pred)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, salary_pred, color='red', linewidth=2)


lin_reg.predict()
