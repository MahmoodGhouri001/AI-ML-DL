
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('pima_indian.csv')

y_df = df[['Class']]
X_df = df.drop(['Class'],axis=1).copy()

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

model.score(X_train, y_train)
model.score(X_test,y_test)
