from sklearn.model_selection import train_test_split
from initialize import initalize
from sklearn.linear_model import Perceptron
from adaline import Adaline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

svc = SVC(max_iter=10)
svc = svc.fit(X=X_train, y=y_train)
s_score = svc.score(X_val, y_val)
print("The accuracy of the default SVC is: ", s_score)

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X=X_train, y=y_train)
rf_score = rf.score(X_val, y_val)
print("The accuracy of the default Random Forest is: ", rf_score)

print("The best performing model for this data is Logistic Regression")
print("Refine Logistic Regression ---------------------")

# grid search for best parameters using cross fold validation
C = [0.1, 0.01, 0.001, 1, 10, 100, 1000]
iterations = [100, 200, 300, 400, 500]

logreg = LogisticRegression()
params = [{'C': C, 'max_iter': iterations}]
gs_log = GridSearchCV(estimator=logreg, param_grid=params, scoring='accuracy', cv=5)
gs_log.fit(X_train, y_train)
print("Best Parameters from GridSearchCV: ", gs_log.best_params_)
val_score = gs_log.score(X_val, y_val)
print("The accuracy of the 'best model' on the validation set is: ", val_score)
score = gs_log.score(X_test, y_test)
print("The final result:")
print("The accuracy of the 'best' Logisic Regression model based on the Grid Search utilizing the n-fold cross validation on the Test set is: ", score)

