import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./titanic-dataset-full.csv")
print(df.columns)

X = pd.DataFrame()
X['gender'] = df['gender']
X['age'] = df['age']
X['pclass'] = df['pclass']
X['sibsp'] = df['sibsp']
X['parch'] = df['parch']
X['survived'] = df['survived']

X["age"].fillna(X.age.mean(), inplace=True)
X = X.dropna(axis=0)


#survived will be my dependent variable, y.   I'll assign it to y and remove it from X
y = X['survived']
X = X.drop(['survived'], axis=1)

X['gender'] = pd.get_dummies(X.gender)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for neighbors in range(1, 10):
    knn_model = KNeighborsClassifier(n_neighbors=neighbors, p = 1)
    knn_model.fit(X_train,y_train)
    print ("Knn Train accuracy for ", neighbors, ": ", accuracy_score(y_train,knn_model.predict(X_train)))
    print ("Knn Test accuracy for ", neighbors, ": ", accuracy_score(y_test,knn_model.predict(X_test)))
    
    
knn_model.kneighbors([X_test.iloc[0]])

