from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset 1
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Design and train NN
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300).fit(X_train, y_train)

# Test and evaluate
pred = clf.predict(X_test)
print("Digits Dataset Accuracy:", accuracy_score(y_test, pred))
