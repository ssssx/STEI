# -*- coding:utf-8 -*-
import numpy as np

def get_reshape(input):
    size = input.size()
    input = input.cpu()
    out = np.reshape(input.detach().numpy(), ((size[0], -1)))
    return out

def get_RMSE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    return np.sqrt(np.mean((label - result) ** 2))

def get_MAE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    return np.mean(np.abs(label - result))

def get_L1(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    return np.abs(label-result)

def get_RSE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    return np.sqrt(np.sum((label - result) ** 2)) / np.sqrt(np.sum((label - label.mean()) ** 2))

def CORR(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    u = ((label - label.mean(0)) * (result - result.mean(0))).sum(0)
    d = np.sqrt(((label - label.mean(0)) ** 2 * (result - result.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MSE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    return np.mean((result - label) ** 2)


def MAPE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    denominator = np.where( label != 0,  label, 1)
    return np.mean(np.abs((result - label) / denominator))



def MSPE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    denominator = np.where(label != 0, label, 1)
    return np.mean(np.square((result - label) / denominator))

def R2(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    tss = np.sum((label-np.mean(label))**2)
    rss = np.sum((label-result)**2)
    r2 = 1-rss/tss
    return r2

def MRE(label, result):
    label = get_reshape(label)
    result = get_reshape(result)
    denominator = np.where(label != 0, label, 1)
    return np.mean(np.abs((label-result)/denominator))




