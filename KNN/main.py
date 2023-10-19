import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

knn = KNN(3, 'classification')
knn.fit(X, y)
predicts = knn.predict()
print(predicts)