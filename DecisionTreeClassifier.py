import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("./bank-additional_bank-additional-full.csv")

#for i in range(0,df.shape[1]):
#    if df.dtypes[i]=='object':
#        df[df.columns[i]] = le.fit_transform(df[df.columns[i]])

y_df = df[['y']]
X_df = df.drop(['y'],axis=1).copy()


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for i in range(0,X_df.shape[1]):
    if X_df.dtypes[i]=='object':
        X_df[X_df.columns[i]] = le.fit_transform(X_df[X_df.columns[i]])

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

        
from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []

for depth in range(1, 20):
    dt_clf = DecisionTreeClassifier(max_depth = depth)
    dt_clf.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train,dt_clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test,dt_clf.predict(X_test)))
    print ("*******************")
    print ("Depth " , depth)
    print ("DT accuracy train ", accuracy_score(y_train,dt_clf.predict(X_train)))
    print ("DT accuracy test ", accuracy_score(y_test,dt_clf.predict(X_test)))

plot_df = pd.DataFrame()
plot_df['train'] = train_acc
plot_df['test'] = test_acc
plot_df['x'] = range(1,20)

import matplotlib.pyplot as plt
plt.plot( 'x', 'test', data=plot_df, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'train', data=plot_df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.legend()

final_dt_clf = DecisionTreeClassifier(max_depth = 5)
final_dt_clf.fit(X_train, y_train)

from sklearn.tree import export_graphviz
import pydotplus

dot_data = export_graphviz(final_dt_clf, out_file=None,
                           feature_names=X_df.columns)
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("dt.pdf")

#####################################
from sklearn import metrics
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test,final_dt_clf.predict(X_test)).ravel()
print(confusion_matrix(y_test,final_dt_clf.predict(X_test)))


import scikitplot as skplt
y_pred_proba = final_dt_clf.predict_proba(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, [row[0] for row in y_pred_proba], pos_label=2)
skplt.metrics.plot_roc_curve(y_test, y_pred_proba)
plt.show()
