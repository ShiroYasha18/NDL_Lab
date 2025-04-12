from sklearn.datasets import  load_iris
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (train_test_split)

X,y = load_iris(return_X_y=True)
X_train,X_test,Y_train,Y_test= train_test_split(X,y)
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=200).fit(X_train,Y_train)

print(accuracy_score(Y_test,clf.predict(X_test)))