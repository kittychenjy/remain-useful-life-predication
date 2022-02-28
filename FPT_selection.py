#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
# path='data/kurtosis_npy/Bearing1_1kurtosis_v.npy'
# seq_kurtosis=np.load(path)
# plt.plot(seq_kurtosis)
#
# # 使用论文An Improved Exponential Model for
# # Predicting Remaining Useful Life
# # of Rolling Element Bearings中的方法筛选

def get_fpt(seq_kurtosis):
    all_fpt = []
    fpt_index = []
    l = 0
    for i, kur in enumerate(seq_kurtosis):
        fpt = []
        if i > l and i<len(seq_kurtosis)-10:
            if i<400:
                t_seq = seq_kurtosis[0:400]
            else:
                t_seq = seq_kurtosis[0:400]
            mean = t_seq.mean()
            std = t_seq.std()
            flag=0
            for j in range(l + 1):
                if seq_kurtosis[i + j] > (mean + 3 * std):
                    flag=flag+1
            if j == flag-1:
                all_fpt.append(i)
                if len(all_fpt) > 1:
                    l = l + 1
                    flag=0
                    for j in range(l + 1):
                        if seq_kurtosis[i + j] > (mean + 3 * std):
                            flag=flag+1
                    if j == flag-1:
                        return i
                        break
                else:
                    l = l + 1
    return 1


def get_fpt0(seq_kurtosis):
    all_fpt = []
    fpt_index = []
    l = 1
    flag=0
    for i, kur in enumerate(seq_kurtosis):
        if i > l:
            flag = 0
            if i<400:
                t_seq = seq_kurtosis  #此处理解的早期阶段，理解为前200个点。
            else:
                t_seq = seq_kurtosis
            mean = t_seq.mean()
            std = t_seq.std()
            for j in range(l + 1):
                if seq_kurtosis[i - j] > (mean + 2 * std):
                    flag=flag+1
            if j+1 == flag :
                fpt=i
                return i
    return 1


