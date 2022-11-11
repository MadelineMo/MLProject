import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from kneighbor import KNeighbors


df = pd.read_csv('weatherAUS.csv', header=0, encoding='utf-8')

# remove dates colum
df = df.drop(['Date'], axis=1)

# replace names with numbers
df = df.replace('Adelaide', 1)
df = df.replace('Albany', 2)
df = df.replace('Albury', 3)
df = df.replace('AliceSprings', 4)
df = df.replace('BadgerysCreek', 5)
df = df.replace('Ballarat', 6)
df = df.replace('Bendigo', 7)
df = df.replace('Brisbane', 8)
df = df.replace('Cairns', 9)
df = df.replace('Canberra', 10)
df = df.replace('Cobar', 11)
df = df.replace('CoffsHarbour', 12)
df = df.replace('Dartmoor', 13)
df = df.replace('Darwin', 14)
df = df.replace('GoldCoast', 15)
df = df.replace('Hobart', 16)
df = df.replace('Katherine', 17)
df = df.replace('Launceston', 18)
df = df.replace('MelbourneAirport', 20)
df = df.replace('Melbourne', 19)
df = df.replace('Mildura', 21)
df = df.replace('Moree', 22)
df = df.replace('MountGambier', 23)
df = df.replace('MountGinini', 24)
df = df.replace('Newcastle', 25)
df = df.replace('Nhil', 26)
df = df.replace('NorahHead', 27)
df = df.replace('NorfolkIsland', 28)
df = df.replace('Nuriootpa', 29)
df = df.replace('PearceRAAF', 30)
df = df.replace('Penrith', 31)
df = df.replace('PerthAirport', 33)
df = df.replace('Perth', 32)
df = df.replace('Portland', 34)
df = df.replace('Richmond', 35)
df = df.replace('Sale', 36)
df = df.replace('SalmonGums', 37)
df = df.replace('SydneyAirport', 38)
df = df.replace('Sydney', 39)
df = df.replace('Townsville', 40)
df = df.replace('Tuggeranong', 41)
df = df.replace('Uluru', 42)
df = df.replace('WaggaWagga', 43)
df = df.replace('Walpole', 44)
df = df.replace('Watsonia', 45)
df = df.replace('Williamtown', 46)
df = df.replace('Witchcliffe', 47)
df = df.replace('Wollongong', 48)
df = df.replace('Woomera', 49)

# replace north, south, east, west
df = df.replace('E', 1)
df = df.replace('ENE', 2)
df = df.replace('ESE', 3)
df = df.replace('N', 4)
df = df.replace('NA', 5)
df = df.replace('NE', 6)
df = df.replace('NNE', 7)
df = df.replace('NNW', 8)
df = df.replace('NW', 9)
df = df.replace('S', 10)
df = df.replace('SE', 11)
df = df.replace('SSE', 12)
df = df.replace('SSW', 13)
df = df.replace('SW', 14)
df = df.replace('W', 15)
df = df.replace('WNW', 16)
df = df.replace('WSW', 17)

# replace yes and no
df = df.replace('Yes', 1)
df = df.replace('No', -1)
df['RainToday'] = df['RainToday'].fillna(-1)
df['RainTomorrow'] = df['RainTomorrow'].fillna(-1)

# fill all empty boxes with 0
df = df.fillna(0)
df = df.astype(float)
data = df.to_numpy()

# sort data
y3 = data[:, 21]  # column 21
X3 = data[:, 0:21]

X_train, X_test, y_train, y_test = train_test_split(
    X3, y3, test_size=0.15, random_state=0)

X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=0.15, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_dev_std = sc.transform(X_dev)

# Question 1
# logistic regression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_std, y_train)

score = logisticRegr.score(X_dev_std, y_dev)
print("Logistic Regression default accuracy on the development set is: ", score)

#svc = SVC(max_iter=50)
#svc.fit(X_train_std, y_train)

#predictions4 = svc.predict(X_dev_std)
#score4 = svc.score(X_dev_std, y_dev)
#print("SVC default accuracy on development set is: ", score4)

# Question 2
# improve model
log = LogisticRegression(C=1)
log.fit(X_train_std, y_train)

s = log.score(X_dev_std, y_dev)
print("Logistic Regression C= 1 accuracy on the development set is: ", s)

log2 = LogisticRegression(C=0.001)
log2.fit(X_train_std, y_train)

s2 = log2.score(X_dev_std, y_dev)
print("Logistic Regression C= 0.001 accuracy on the development set is: ", s2)

log3 = LogisticRegression(C=0.01)
log3.fit(X_train_std, y_train)

s3 = log3.score(X_dev_std, y_dev)
print("Logistic Regression C= 0.01 accuracy on the development set is: ", s3)

log4 = LogisticRegression(C=0.1)
log4.fit(X_train_std, y_train)

s4 = log4.score(X_dev_std, y_dev)
print("Logistic Regression C= 0.1 accuracy on the development set is: ", s4)

log5 = LogisticRegression(C=10)
log5.fit(X_train_std, y_train)

s5 = log5.score(X_dev_std, y_dev)
print("Logistic Regression C= 10 accuracy on the development set is: ", s5)

log6 = LogisticRegression(C=100)
log6.fit(X_train_std, y_train)

s6 = log6.score(X_dev_std, y_dev)
print("Logistic Regression C= 100 accuracy on the development set is: ", s6)

log7 = LogisticRegression(C=1000)
log7.fit(X_train_std, y_train)

s7 = log7.score(X_dev_std, y_dev)
print("Logistic Regression C= 1000 accuracy on the development set is: ", s7)
print("Therefore the Logistic Regression with the best accuracy is the default value of C=1")

#svc2 = SVC(C=100, max_iter=50)
#svc2.fit(X_train_std, y_train)

#predictions5= svc2.predict(X_dev_std)
#score5 = svc2.score(X_dev_std, y_dev)
#print("SVC edited accuracy on development set is: ", score5)

# Question 3
# k-nearest neighbor classifer
kNear = KNeighbors
kNear.fit(self=kNear, X=X_train_std, y=y_train)
score2 = kNear.evaluate(self=kNear, X_test=X_dev_std, y_test=y_dev, k=5)
print("K-Nearest Neighbor accuracy on the development set is: ", score2)

# Question 4
# Dummy Classifer
dummy = DummyClassifier()
dummy.fit(X_train_std, y_train)
score3 = dummy.score(X_dev_std, y_dev)
print("Dummys accuracy on the development set is: ", score3)

# Question 5
# Compare best model on test set
finalscore1 = log.score(X_test_std, y_test)
print("The accuracy of Logistic Regression C=1 (the best performing and the default) is: ", finalscore1)
# predictions2 = log.predict(X_test_std)
finalscore2 = kNear.evaluate(self=kNear, X_test=X_test, y_test=y_test, k=5)
print("The accuracy of K-Nearest Neighbor on the Test set is: ", finalscore2)
finalscore3 = dummy.score(X_test_std, y_test)
print("The accuracy of the dummy model on the test set is:", finalscore3)
