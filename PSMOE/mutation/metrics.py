import numpy as np
import torch


def calculate_metric(pred_y, labels):
    _, predicted = torch.max(pred_y, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    test_num = len(labels)
    K = 3
    class_counts = {cls: np.sum(labels == cls) for cls in [0, 1, 2]}
    x = np.array([class_counts[0], class_counts[1], class_counts[2]])
    confusion = np.zeros((3, 3), dtype=int)
    for pred, label in zip(predicted, labels):
        confusion[label, pred] += 1
    tp = confusion.diagonal()
    y = confusion.sum(axis=0)
    gcc_numerator = 0.0
    for i in range(3):
        for j in range(3):
            z_ij = confusion[i, j]
            e_ij = (x[i] * y[j]) / test_num
            gcc_numerator += (z_ij - e_ij) ** 2 / e_ij
    gcc = gcc_numerator / (test_num * (K - 1))
    metrics = {'GCC': gcc}
    for cls in [0, 1, 2]:
        fn = class_counts[cls] - tp[cls]
        fp = confusion[:, cls].sum() - tp[cls]
        tn = test_num - class_counts[cls] - fp
        se = tp[cls] / (tp[cls] + fn) if (tp[cls] + fn) != 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        ppv = tp[cls] / (tp[cls] + fp) if (tp[cls] + fp) != 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0
        metrics[f"{cls}_SE"] = se
        metrics[f"{cls}_SP"] = sp
        metrics[f"{cls}_PPV"] = ppv
        metrics[f"{cls}_NPV"] = npv
    acc = tp.sum() / test_num

    return metrics, acc, gcc