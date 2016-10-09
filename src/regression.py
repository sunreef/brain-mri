from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from random import randint, random
import sys

def ridge(X, y):
    lowest_mse = sys.maxint
    best_alpha = sys.maxint

    for i in range(20):
        if i % 5 == 0:
            print("Ridge Loop: " + str(i))

        # Monte Carlo alpha
        alpha = random() * 5
        rls = Ridge(alpha=alpha, tol=0.1)

        scores = cross_val_score(rls, X, y, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=3)

        if (-scores.mean() < lowest_mse):
            lowest_mse = -scores.mean()
            best_alpha = alpha


    print("Best Ridge Regression Params:")
    print("Alpha: " +  str(alpha))
    print("Lowest Error: " + str(lowest_mse))

def lasso(X, y):
    lowest_mse = sys.maxint

    best_alpha = sys.maxint

    for i in range(20):
        if i % 5 == 0:
            print("Lasso Loop: " + str(i))

        # Monte Carlo alpha
        alpha = random() * 5
        lls = Lasso(alpha=alpha, tol=0.1)

        scores = cross_val_score(lls, X, y, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=3)

        if (-scores.mean() < lowest_mse):
            lowest_mse = -scores.mean()
            best_alpha = alpha

    print("Best Lasso Regression Params:")
    print("Alpha: " + str(alpha))
    print("Lowest Error: " + str(lowest_mse))