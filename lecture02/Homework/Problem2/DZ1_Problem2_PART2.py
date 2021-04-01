import numpy as np


def count_parametrs(y_true, y_predict, percent):
    if percent == None:
        percent = 50
    else:
        y_true = y_true[: y_true.shape[0] * percent // 100]
        y_predict = y_predict[: y_predict.shape[0] * percent // 100]
    y_hard = ((y_predict[:, 0] >= percent / 100)) * 1
    if np.sum((y_true == 1) & (y_predict[:, 0] == 1)) / np.sum(y_true == 1) < 0.5:
        y_hard = ((y_predict[:, 1] >= percent / 100)) * 1
    TP = np.sum((y_hard == 1) & (y_true == 1))
    FP = np.sum((y_hard == 1) & (y_true == 0))
    TN = np.sum((y_hard == 0) & (y_true == 0))
    FN = np.sum((y_hard == 0) & (y_true == 1))
    return TP, FP, TN, FN


def accuracy_score(y_true, y_predict, percent=None):
    TP, FP, TN, FN = count_parametrs(y_true, y_predict, percent)
    return (TP + TN) / (TP + TN + FP + FN)


def precision_score(y_true, y_predict, percent=None):
    TP, FP, TN, FN = count_parametrs(y_true, y_predict, percent)
    return TP / (TP + FP)


def recall_score(y_true, y_predict, percent=None):
    TP, FP, TN, FN = count_parametrs(y_true, y_predict, percent)
    return TP / (TP + FN)


def lift_score(y_true, y_predict, percent=None):
    TP, FP, TN, FN = count_parametrs(y_true, y_predict, percent)
    return TP / (TP + FP) / (TP + FN) / y_true.shape[0]


def f1_score(y_true, y_predict, percent=None):
    TP, FP, TN, FN = count_parametrs(y_true, y_predict, percent)
    return 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN))

#! Проверка кода:
# file = np.loadtxt("lecture02\Homework\Problem2\HW2_labels.txt", delimiter=",")
# y_predict, y_true = file[:, :2], file[:, -1]

# print("accuracy_score: ", accuracy_score(y_true, y_predict, percent=None))
# print("precision_score:  ", precision_score(y_true, y_predict, percent=None))
# print("recall_score: ", recall_score(y_true, y_predict, percent=None))
# print("lift_score: ", lift_score(y_true, y_predict, percent=None))
# print("f1_score: ", f1_score(y_true, y_predict, percent=None))
