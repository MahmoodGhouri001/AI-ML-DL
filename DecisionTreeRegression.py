
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

x = 2 - 3 * np.random.normal(0, 1, 100)
y = 2 * (x ** 1.5) + np.random.normal(-100, 100, 100)
df = pd.DataFrame({'X':x, 'y': y})
df = df.fillna(0)
plt.scatter(x, y)
plt.show()


df = df.sort_values(by=['X'])
x_train, y_train = df[['X']], df[['y']]

# create a linear regression model and fit the data
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
# printing metrics of the linear model
print('The RMSE of the linear regression model is {}'.format(mean_squared_error(y_train, y_pred)))
print('The R2 score of the linear regression model is {}'.format(r2_score(y_train, y_pred)))

plt.scatter(x_train, y_train, s=10)
plt.plot(x_train, y_pred, color='r')
plt.show()

# transform the features to higher degree
for degree in range(2,10):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly_train = polynomial_features.fit_transform(x_train)

    # train a polynomial regression model with higher degree features
    polynomial_model = LinearRegression()
    polynomial_model.fit(x_poly_train, y_train)
    y_pred = polynomial_model.predict(x_poly_train)

    print('The RMSE of the polynomial regression of degree {} is {}'.format(degree,mean_squared_error(y_train, y_pred)))
    print('The R2 score of the polynomial regression of degree {} is  {}'.format(degree, r2_score(y_train, y_pred)))

    plt.scatter(x_train, y_train)
    plt.plot(x_train, y_pred, color='r')
    plt.show()


decision_tree_model = DecisionTreeRegressor(max_depth = 10, min_samples_split = 3)
decision_tree_model.fit(x_train, y_train)

y_pred = decision_tree_model.predict(x_train)

print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_train, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_train, y_pred)))

