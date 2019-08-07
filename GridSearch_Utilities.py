import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('./PRSA_data.csv')
X_df, y_df = df[['year','month','day','hour',
                        'dew_point','temp','pressure','wind_speed',
                        'snow_hours','rain_hours']], df[['pm2.5']]

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)  

parameters = {'num_estimators':range(5,50),
        'max_depth':range(2,3), 'min_samples_split':[3,4]
        }
                            

dt_model = DecisionTreeRegressor()

clf = GridSearchCV(dt_model, parameters, cv=5)
clf.fit(X_train, y_train)

final_dt_model = clf.best_estimator_
print(final_dt_model)

y_pred = final_dt_model.predict(X_train)
print('**** Train Accuracy*****')
print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_train, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_train, y_pred)))


y_pred = final_dt_model.predict(X_test)
print('**** Test Accuracy*****')
print('The RMSE of the Decision Tree regression {}'.format(mean_squared_error(y_test, y_pred)))
print('The R2 score of the DecisionTree regression {}'.format(r2_score(y_test, y_pred)))

###############################################################################3
# Label Encoder
####################
boston_df = pd.read_csv('./Boston.csv')

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(boston_df[['TOWN']])

print(le.classes_)

print(le.transform(["Winchester"]))

boston_df['TOWN1'] = boston_df[['TOWN']].apply(lambda x: le.transform(x))

###############################################################################3
# One Hot Encoder
####################

boston_df = pd.get_dummies(boston_df,columns=['TOWN'])

###############################################################################3
# Standard Scaler
####################

import pandas as pd    
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
boston_df_scaled = pd.DataFrame(ss.fit_transform(boston_df),columns = boston_df.columns)


###############################################################################3
# Correlation Coefficient
####################
boston_df.corr(method='pearson').to_csv('pearson.csv')
boston_df_scaled.corr(method='pearson').to_csv('scaled_pearson.csv')
