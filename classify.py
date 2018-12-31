from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

data_train = pd.read_csv(r'train.csv')
data_test = pd.read_csv(r'test.csv')

nc = 6 # number of classes

# training data
X_train = data_train.T.iloc[:294]
y_train = data_train.T.iloc[294:]

# testing data
X_test = data_test

# Using Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train.T ,y_train.T)
y_pred = clf.predict(X_test)

# print(y_pred)
pd.DataFrame(y_pred,dtype=np.int8).to_csv('predictions.csv')