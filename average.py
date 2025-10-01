import copy
import torch
from torch import nn

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])

    # Average the coefficients
    for k in range(w_avg["coef"].shape[0]):      # over classes
        for j in range(w_avg["coef"].shape[1]):  # over features
            value = w_avg["coef"][k, j]
            for i in range(1, len(w)):   # over clients
                value += w[i]["coef"][k, j] * s_num[i]/temp_sample_num
            value = value * temp_sample_num/total_sample_num
            w_avg["coef"][k, j] = value
    
    # Average the bias too
    if 'intercept' in w_avg:
        for k in range(w_avg["intercept"].shape[0]):
            value = w_avg["intercept"][k]
            for i in range(1, len(w)):   # over clients
                value += w[i]["intercept"][k] * s_num[i]/temp_sample_num
            value = value * temp_sample_num/total_sample_num
            w_avg["intercept"][k] = value

    return w_avg