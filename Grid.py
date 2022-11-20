from sklearn.linear_model import LogisticRegression
from n_fold_cross_validation import cross_validation


def gridSearch(X_train, y_train, LearningRates, iterations):
    max_accuracy = 0
    scores_list = [0]
    parameters = []
    # all possible combinations of parameters
    for i in LearningRates:
        for j in iterations:
            parameters.append((i, j))

    for k in range(len(parameters)):
        model = LogisticRegression(C= parameters[k][0], max_iter= parameters[k][1])
        scores, maxScore, bestModel = cross_validation(X_train=X_train, y_train=y_train, n=10, model=model)
        scores_list.append([maxScore, parameters[k][0], parameters[k][1]])

        if max_accuracy < maxScore:
            max_accuracy = maxScore
            best_model = bestModel
            best_set = [max_accuracy, parameters[k][0], parameters[k][1]]

    return(max_accuracy, best_model, scores_list, best_set)