import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('adult.csv', names = ['age',
                                                  'workclass',
                                                  'fnlwgt',
                                                  'education',
                                                  'education-num',
                                                  'marital-status',
                                                  'occupation',
                                                  'relationship',
                                                  'race',
                                                  'gender',
                                                  'capital-gain',
                                                  'capital-loss',
                                                  'hours-per-week',
                                                  'native-country',
                                                  'salary'], na_values = ' ?')

X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
sim = SimpleImputer()
X[:, [0, 2, 4, 10, 11, 12]] = sim.fit_transform(X[:, [0, 2, 4, 10, 11, 12]])

temp = pd.DataFrame(X[:, [1, 3, 5, 6, 7, 8, 9, 13]])

temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()
temp[3].value_counts()
temp[4].value_counts()
temp[5].value_counts()
temp[6].value_counts()
temp[7].value_counts()

temp[0] = temp[0].fillna(' Private')
temp[1] = temp[1].fillna(' HS-grad')
temp[2] = temp[2].fillna(' Married-civ-spouse')
temp[3] = temp[3].fillna(' Prof-specialty')
temp[4] = temp[4].fillna(' Husband')
temp[5] = temp[5].fillna(' White')
temp[6] = temp[6].fillna(' Male')
temp[7] = temp[7].fillna(' United-States')


X[:, [1, 3, 5, 6, 7, 8, 9, 13]] = temp
#del(temp)

from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()

X[:, 1] = lab.fit_transform(X[:, 1])
X[:, 3] = lab.fit_transform(X[:, 3])
X[:, 5] = lab.fit_transform(X[:, 5])
X[:, 6] = lab.fit_transform(X[:, 6])
X[:, 7] = lab.fit_transform(X[:, 7])
X[:, 8] = lab.fit_transform(X[:, 8])
X[:, 9] = lab.fit_transform(X[:, 9])
X[:, 13] = lab.fit_transform(X[:, 13])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1, 3, 5, 6, 7, 8, 9, 13])
X = one.fit_transform(X)
X = X.toarray()
y = lab.fit_transform(y)
lab.classes_

dataset = pd.read_csv('adult.csv')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)

knn.score(X, y)
knn.score(X_test, y_test)
knn.score(X_train, y_train)

y_pred = knn.predict(X_test)

knn.score(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X, y)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y_test, y_pred, average = 'macro')
recall_score(y_test, y_pred, average = 'macro')
f1_score(y_test, y_pred, average = 'macro')