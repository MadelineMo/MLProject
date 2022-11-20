import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from initialize import initalize
from sklearn.linear_model import Perceptron
from adaline import Adaline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from Grid import gridSearch

# Initialized data
# returns standardized data
print("Data Evaluation ---------------------")
X, y = initalize()

# split data into train and test sets (per y set)
X_training, X_test, y_training, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# validation data from training set
X_train, X_val, y_train, y_val = train_test_split(
    X_training, y_training, test_size=0.3, random_state=0)

print("Default Evaluations ---------------------")
# train and validate default parameters on data
per = Perceptron(eta0=0.1, max_iter=50)
per = per.fit(X=X_train, y=y_train)
per_score = per.score(X_val, y_val)
print("The accuracy of the default Perceptron is: ", per_score)

ad = Adaline(0.1, 10)
ad = ad.fit(X=X_train, y=y_train)
ad_predict = ad.predict(X=X_val)

adaline_correct = 0
for y, pred in zip(y_val, ad_predict):
    if y == pred:
        adaline_correct += 1
print("The accuracy of the default Adaline is: ", (adaline_correct/(y_val.size)))

log = LogisticRegression()
log = log.fit(X=X_train, y=y_train)
log_score = log.score(X=X_val, y=y_val)
print("The accuracy of the default Logistic Regression is: ", log_score)

k = KNeighborsClassifier()
k = k.fit(X=X_train, y=y_train)
k_score = k.score(X=X_val, y=y_val)
print("The accuracy of the default K Nearest Neighbor is: ", k_score)

# svc = SVC(max_iter=10)
# svc = svc.fit(X=X_train, y=y_train)
# s_score = svc.score(X_val, y_val)
# print("The accuracy of the default SVC is: ", s_score)

print("The best performing model for this data is Logistic Regression")
print("Refine Logistic Regression ---------------------")

# grid search for best parameters using cross fold validation
C = [0.1, 0.01, 0.001, 1, 10, 100, 1000]
iterations = [100, 200, 300, 400, 500]

max_accuracy, grid_best_model, scores_list, best_set = gridSearch(X_train, y_train, C, iterations)
print("---------Best model from grid search: Accuracy=", best_set[0], " C=", best_set[1], " Iterations=", best_set[2])

# best = grid_best_model(C=10, iterations=100)
# best = best.fit(X=X_training, y=y_training)
score = grid_best_model.score(X_test, y_test)
print("The final result:")
print("The accuracy of the 'best' Logisic Regression model based on the Grid Search utilizing the n-fold cross validation on the Test set is: ", score)

