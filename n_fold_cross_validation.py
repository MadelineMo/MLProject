import numpy as np
from sklearn.linear_model import LogisticRegression


def cross_validation(X_train, y_train, n=5, model=LogisticRegression):
    # split set by n
    Xfolds = np.array_split(X_train, n)
    yfolds = np.array_split(y_train, n)
    # v = validation number
    v = 0
    result = 0
    scores = [0]
    maxScore = 0
    bestModel = LogisticRegression

    # i = iteration
    for i in range(n):
        # j = merge and define validation set
        Xvalidation = Xfolds[v]
        yvalidation = yfolds[v]
        for j in range(n):
            if (v != 0 & j==0):
                XtrainFolded = Xfolds[0]
                ytrainFolded = yfolds[0]
                continue
            else:
                XtrainFolded = Xfolds[1]
                ytrainFolded = yfolds[1]

            if (j != v):
                XtrainFolded = np.concatenate((XtrainFolded, Xfolds[j]), axis = 0)
                ytrainFolded = np.concatenate((ytrainFolded, yfolds[j]), axis= 0)

        v = v + 1
        model.fit(XtrainFolded, ytrainFolded)
        result = model.score(Xvalidation, yvalidation)
        if result > maxScore:
            maxScore = result
            bestModel = model
        scores.append(result)

    return (scores, maxScore, bestModel)