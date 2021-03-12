import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


def get_model_score(estimator, y_true, y_pred, x_train, y_train):
    accuracies = cross_val_score(estimator, X=x_train, y=y_train, cv=10)
    acc_mean = accuracies.mean()
    acc_std = accuracies.std()
    print(f" Mean accuracy: {acc_mean}, std: {acc_std}")
    print(f"Mean squared error: {np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))}")
    print(f"R2 score: {r2_score(y_true, y_pred)}")
