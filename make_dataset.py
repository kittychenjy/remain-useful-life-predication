
# 制作测试西交 和 phm的数据集，并打标签
#encoding = utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import pandas as pd
from feature_functions import VibrationSignal
from FPT_selection import get_fpt
# for XJTU
con_name= '35Hz12kN'
path='XJTU-SY_Bearing_Datasets/{}'.format(con_name)
lable_path = 'lable/{}/'.format(con_name)
npy_path = 'raw_data/{}/'.format(con_name)




def get_lab(all_data):
    all_kur=[]
    length=len(all_data)
    for data in all_data:
        b=VibrationSignal(data.swapaxes(0,1)[0])
        kur=b.get_kurtosis()
        all_kur.append(kur)
    fpt=get_fpt(all_kur)
    lable=np.ones(length)
    life=length-fpt
    for i in range(fpt,length):
        lable[i]=(length-i)/life
    return lable


for filename in os.listdir(path):
    all_data = []
    all_lab = []
    file = path + '/' + filename
    file_list = os.listdir(file)
    file_list.sort(key=lambda x: int(x[:-4]))  # 读取时发现排序有问题。
    for f_name in file_list:
        data_path = file + '/' + f_name
        data = pd.read_csv(data_path).values
        all_data.append(data)
    all_data = np.array(all_data)
    all_lab = get_lab(all_data)
    try:
        np.save(npy_path + filename + '_data.npy', all_data)
        np.save(lable_path + filename + '_lable.npy', all_lab)
    except:
        os.makedirs(npy_path)
        os.makedirs(lable_path)
        np.save(npy_path + filename + 'data.npy', all_data)
        np.save(lable_path + filename + 'lable.npy', all_lab)
