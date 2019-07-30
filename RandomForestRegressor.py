
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('./PRSA_data.csv')
X_df, y_df = df[['year','month','day','hour',
                        'dew_point','temp','pressure','wind_speed',
                        'snow_hours','rain_hours']], df[['pm2.5']]

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)  

decision_tree_model = DecisionTreeRegressor(max_depth = 100)
decision_tree_model.fit(X_train, y_train)

y_pred = decision_tree_model.predict(X_train)
print('---------- Train Accuracy----------')
print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_train, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_train, y_pred)))


y_pred = decision_tree_model.predict(X_test)
print('---------- Test Accuracy----------')
print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_test, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_test, y_pred)))


rf_model = RandomForestRegressor(n_estimators=15, max_depth=25)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_train)
print('---------- RF Train Accuracy----------')
print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_train, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_train, y_pred)))


y_pred = rf_model.predict(X_test)
print('---------- RF Test Accuracy----------')
print('The RMSE of the Random Forest regression {}'.format(mean_squared_error(y_test, y_pred)))
print('The R2 score of the Random Forest regression {}'.format(r2_score(y_test, y_pred)))
