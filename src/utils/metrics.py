from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

def auroc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return roc_auc_score(y_true, y_pred)

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return precision_score(y_true,y_pred, zero_division=0)

def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return recall_score(y_true,y_pred, zero_division=0)

def f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return f1_score(y_true,y_pred)

def auprc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return average_precision_score(y_true, y_pred)