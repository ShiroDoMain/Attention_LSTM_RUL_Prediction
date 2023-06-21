import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from tqdm import tqdm
from data_util import load_RUL, TestLoader


def myScore(y_ture, y_pred):
    score = 0
    for i in range(len(y_pred)):
        if y_ture[i] <= y_pred[i]:
            score += np.exp((y_pred[i] - y_ture[i]) / 10.0) - 1
        else:
            score += np.exp((y_ture[i] - y_pred[i]) / 13.0) - 1
    return score


def testing(model):
    rmse = 0
    score = 0
    j = 1
    result = []
    gt = TestLoader("datasets/test_FD001.txt")
    y_test = load_RUL("datasets/RUL_FD001.txt")
    group_test_loader = DataLoader(gt)
    progressbar = tqdm(group_test_loader, desc="[Test]")
    for x_test in progressbar:
        data_predict = 0
        seq_len, bs, input_dim = x_test.size()

        for t in range(1, bs-2):  # iterate to the end of each sequence
            # if t == bs - 1:
            #     X_test = np.append(x_test[:,98:, :], [[np.zeros(14)]], axis=1)
            # else:
            X_test = x_test[:, t - 1:t + 2, 2:]

            X_test_tensors = Variable(X_test)

            test_predict = model(X_test_tensors, t)
            data_predict = test_predict.data.numpy()[-1]

            # block for linearily decreasing the RUL after each iteration
            if data_predict - 1 < 0:
                data_predict = 0
            else:
                data_predict -= 1
        result.append(data_predict)
        rmse += np.power((data_predict - y_test[j - 1]), 2)
        j += 1
        
    score += myScore(y_test, result)
    rmse = np.sqrt(rmse / 100)
    return rmse, score
