
# 制作测试西交 和 phm的数据集，并打标签
#encoding = utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from feature_functions import VibrationSignal
from FPT_selection import get_fpt,get_fpt0
# for phm
def show(all_rms, all_kur, fpt, name):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=0, hspace=0.5)
    plt.title('rms')
    plt.plot(all_rms, color='b')
    plt.axvline(fpt, color='red')
    plt.subplot(2, 1, 2)
    plt.title('kurtosis')
    plt.plot(all_kur, color='b')
    plt.axvline(fpt, color='red')
    try:
        plt.savefig(ans_path + name + '.png')
    except:
        os.makedirs(ans_path)
        plt.savefig(ans_path + name + '.png')
def get_lab(all_data, name):
    all_kur = []
    all_rms = []
    length = len(all_data)
    for data in all_data:
        b = VibrationSignal(data.swapaxes(0, 1)[0])
        kur = b.get_kurtosis()
        rms = b.get_rms()
        all_kur.append(kur)
        all_rms.append(rms)
    fpt = get_fpt0(np.array(all_kur))
    show(all_rms, all_kur, fpt, name)
    lable = np.ones(length)
    life = length - fpt
    for i in range(fpt, length):
        lable[i] = (length - i) / life
    return lable
for con_name in ['Full_Test_Set']:
    # con_name= 'Learning_set'
    path='phm-ieee-2012-data-challenge-dataset-master/{}'.format(con_name)
    lable_path = 'phm0/phm-lable/{}/'.format(con_name)
    npy_path = 'phm0/phm-raw_data/{}/'.format(con_name)
    ans_path = 'phm0/ans/'
    for filename in os.listdir(path):
        all_data = []
        all_lab = []
        f_list= []
        file = path + '/' + filename
        file_list = os.listdir(file)
        if con_name=='Full_Test_Set':
            for content in file_list:
                if content.split('_')[0]=='acc':
                    f_list.append(content)
        #file_list.sort(key=lambda x: int(x[:-4]))  # 读取时发现排序有问题。
        for f_name in f_list:
            data_path = file + '/' + f_name
            data = pd.read_csv(data_path).values[:,4:6]
            all_data.append(data)
        all_data = np.array(all_data)
        all_lab = get_lab(all_data,name=filename)
        try:
            np.save(npy_path + filename + '_data.npy', all_data)
            np.save(lable_path + filename + '_lable.npy', all_lab)
        except:
            os.makedirs(npy_path)
            os.makedirs(lable_path)
            np.save(npy_path + filename + 'data.npy', all_data)
            np.save(lable_path + filename + 'lable.npy', all_lab)