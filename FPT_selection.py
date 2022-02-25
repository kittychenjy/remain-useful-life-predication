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
        if i > 1:
            t_seq = seq_kurtosis[0:i]
            mean = t_seq.mean()
            std = t_seq.std()
            j = i
            for j in range(l + 1):
                if seq_kurtosis[i + j] > (mean + 3 * std):
                    ''
                else:
                    break
            if j == l:
                all_fpt.append(i)
                if len(all_fpt) > 1:
                    l = l + 1
                    for j in range(l + 1):
                        if seq_kurtosis[i + j] > (mean + 3 * std):
                            ''
                        else:
                            break
                    if j == l:
                        print(i)
                        break
                else:
                    l = l + 1






