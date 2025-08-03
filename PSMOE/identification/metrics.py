from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np
import torch
from utils import  get_sequence_vectors

def caculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    labels = list(map(int, labels))
    pred_prob = list(map(float, pred_prob))
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data

def evaluate_accuracy(data_iter, net, seq2vec, device):
    acc_sum, n = 0.0, 0
    for x, y, z in data_iter:
        x = x.to(device)
        y = y.to(device)
        vec = get_sequence_vectors(z, seq2vec, device)
        if vec is not None:
            outputs, _ = net.trainModel(x, vec)
            acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        else:
            print("Warning: No valid sequence vectors found in this batch. Skipping...")

    if n == 0:
        print("Warning: No valid batches were processed. Returning 0 accuracy.")
        return 0.0
    return acc_sum / n

