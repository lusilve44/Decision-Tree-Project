import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv('/workspace/Decision-Tree-Project/data/processed/df_data.csv')

X = df.drop(["Outcome"],axis=1)
y = df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.25)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train,y_train)

clf_entropy = DecisionTreeClassifier(random_state=42,criterion="entropy")
clf_entropy.fit(X_train,y_train)


# Hypertuning

parameters = {'criterion':['gini','entropy','log_loss'],'splitter':['best','random'],'max_depth':range(1,200,2),'min_samples_split': [2, 3, 4]}
clf_cv = GridSearchCV(DecisionTreeClassifier(), parameters)
clf_cv.fit(X_train,y_train)

clf_tunned = clf_cv.best_estimator_
clf_tunned.fit(X_train,y_train)


# Exporting models 

filename1 = '/workspace/Decision-Tree-Project/models/model_dt_gini.sav'
pickle.dump(clf, open(filename1, 'wb'))

filename2 = '/workspace/Decision-Tree-Project/models/model_dt_entropy.sav'
pickle.dump(clf_entropy, open(filename2, 'wb'))

filename3 = '/workspace/Decision-Tree-Project/models/model_dt_tunned.sav'
pickle.dump(clf_tunned, open(filename3, 'wb'))