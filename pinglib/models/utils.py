#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   这里提供一些通用的功能方便组建多类模型

import numpy as np

#   从混淆矩阵得到每类的TP,TN,FP,FN
def confusion_to_TFPN(confusion):
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = np.sum(confusion) - TP - FP - FN
    return TP, TN, FP, FN


#   从多个混淆矩阵获得TFPN的序列
#   输入： C x C x T
#   输出： C x 4 x T
def confusions_to_TFPN(confusions):
    class_num = confusions.shape[0]
    series_length = confusions.shape[-1]
    result = np.zeros((class_num, 4, series_length))
    for t in range(series_length):
        TP, TN, FP, FN = confusion_to_TFPN(confusions[:, :, t])
        result[:, 0, t] = TP
        result[:, 1, t] = TN
        result[:, 2, t] = FP
        result[:, 3, t] = FN
    return result


def confusion_to_accuracy(confusion):
    epi = 1e-5
    accuracy = np.sum(np.diag(confusion)) / (np.sum(confusion) + epi)
    return accuracy


#   从混淆矩阵得到precision
def confusion_to_precision(confusion):
    epi = 1e-5
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    precision = TP / (TP + FP + epi)
    return precision


#   从混淆矩阵得到recall
def confusion_to_recall(confusion):
    epi = 1e-5
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    recall = TP / (TP + FN + epi)
    return recall


#   从混淆矩阵得到F1得分
def confusion_to_f1(confusion):
    epi = 1e-5
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    precision = TP / (TP + FP + epi)
    recall = TP / (TP + FN + epi)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


#   从混淆矩阵得到dice
def confusion_to_dice(confusion):
    epi = 1e-5
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    dice = 2 * TP / (TP + FN + TP + FP + epi)
    return dice


#   从混淆矩阵得到kappa
def confusion_to_kappa(confusion):
    n = np.sum(confusion)
    pa = np.sum(np.diag(confusion)) / n
    pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / (n * n)
    kappa = (pa - pe) / (1 - pe)
    return kappa


#   从混淆矩阵得到jaccard
def confusion_to_jaccard(confusion):
    epi = 1e-5
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    jaccard = TP / (TP + FN + FP + epi)
    return jaccard


#   从混淆矩阵得到各种指标
def confusion_to_all(confusion):
    epi = 1e-5
    accuracy = np.sum(np.diag(confusion)) / (np.sum(confusion) + epi)
    TP, TN, FP, FN = confusion_to_TFPN(confusion)
    precision = TP / (TP + FP + epi)
    recall = TP / (TP + FN + epi)
    f1 = 2 * precision * recall / (precision + recall)
    dice = 2 * TP / (TP + FN + TP + FP + epi)
    jaccard = TP / (TP + FN + FP + epi)
    result_dict = {'accuracy': accuracy,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1,
                   'dice': dice,
                   'jaccard': jaccard}
    return result_dict
