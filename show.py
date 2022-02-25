# 测试西交 和 phm的数据集
#encoding = utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import pandas as pd
from feature_functions import VibrationSignal
path='XJTU-SY_Bearing_Datasets/35Hz12kN'
path_phm= 'phm-ieee-2012-data-challenge-dataset-master/Learning_set'
def get_feature(fe_name):
    # for XJTU
    ans_path='ans/tem_{}/'.format(fe_name)
    npy_path='data/{}_npy/'.format(fe_name)
    for filename in os.listdir(path):
        all_fe_h = []
        all_fe_v = []
        file=path+'/'+filename
        file_list=os.listdir(file)
        file_list.sort(key= lambda x:int(x[:-4])) #读取时发现排序有问题。
        for f_name in file_list:
            data_path=file+'/'+f_name
            data=pd.read_csv(data_path)
            b = VibrationSignal(data.values, signal_axis=0)
            rms=getattr(b,'get_{}'.format(fe_name))()
            all_fe_h.append(rms[0])
            all_fe_v.append(rms[1])
        all_fe_h=np.array(all_fe_h)
        all_fe_v=np.array(all_fe_v)
        try:
            np.save(npy_path + filename + '{}_h.npy'.format(fe_name), all_fe_h)
            np.save(npy_path + filename + '{}_v.npy'.format(fe_name), all_fe_v)
        except:
            os.makedirs(npy_path)
            np.save(npy_path + filename + '{}_h.npy'.format(fe_name), all_fe_h)
            np.save(npy_path + filename + '{}_v.npy'.format(fe_name), all_fe_v)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.5)
        plt.title(filename + '_{}_h'.format(fe_name))
        plt.plot(all_fe_h)
        plt.subplot(2, 1, 2)
        plt.title(filename + '_{}_v'.format(fe_name))
        plt.plot(all_fe_v)
        #plt.show()
        try:
            plt.savefig(ans_path + filename + '_{}.png'.format(fe_name))
        except:
            os.makedirs(ans_path)
            plt.savefig(ans_path + filename + '_{}.png'.format(fe_name))


get_feature('kurtosis')
feature_list=['smra','shape_factor',
              'crest_factor','impulse_factor',
              'clearance_factor','kurtosis_factor']
for fe_name in feature_list:
    get_feature(fe_name)






















def get_feature(fe_name):
    # for XJTU
    ans_path='ans/tem_{}/'.format(fe_name)
    npy_path='data/{}_npy/'.format(fe_name)
    for filename in os.listdir(path_phm):
        all_fe_h = []
        all_fe_v = []
        file=path+'/'+filename
        file_list=os.listdir(file)
        file_list.sort(key= lambda x:int(x[:-4])) #读取时发现排序有问题。
        for f_name in file_list:
            data_path=file+'/'+f_name
            data=pd.read_csv(data_path)
            b = VibrationSignal(data.values, signal_axis=0)
            rms=getattr(b,'get_{}'.format(fe_name))()
            all_fe_h.append(rms[0])
            all_fe_v.append(rms[1])
        all_fe_h=np.array(all_fe_h)
        all_fe_v=np.array(all_fe_v)
        try:
            np.save(npy_path + filename + '{}_h.npy'.format(fe_name), all_fe_h)
            np.save(npy_path + filename + '{}_v.npy'.format(fe_name), all_fe_v)
        except:
            os.makedirs(npy_path)
            np.save(npy_path + filename + '{}_h.npy'.format(fe_name), all_fe_h)
            np.save(npy_path + filename + '{}_v.npy'.format(fe_name), all_fe_v)
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0, hspace=0.5)
        plt.title(filename + '_{}_h'.format(fe_name))
        plt.plot(all_fe_h)
        plt.subplot(2, 1, 2)
        plt.title(filename + '_{}_v'.format(fe_name))
        plt.plot(all_fe_v)
        #plt.show()
        try:
            plt.savefig(ans_path + filename + '_{}.png'.format(fe_name))
        except:
            os.makedirs(ans_path)
            plt.savefig(ans_path + filename + '_{}.png'.format(fe_name))














for filename in os.listdir(path):
    file=path+'/'+filename
    for f_name in os.listdir(file):
        data_path=file+'/'+f_name
        data=pd.read_csv(data_path)
        # t = np.arange(0, 1, 1/1000)
        # load = np.cos(2 * np.pi * 100 * t) + np.random.randn(t.size)
        # a = np.array([[1, 2, 3],[3,6,9]])
        b = VibrationSignal(data.values[0:30000], signal_axis=0)
        bx, by = b.get_fft(30000)
        bx1, by1 = b.get_square_demodulation(30000)
        bx2, by2 = b.get_psd_by_correlation(30000)
        by2 = by2 / 30000
        bx3, by3 = b.get_psd_by_square(30000)
        bx4, by4 = b.get_psd_using_scipy(30000)
        # by3 = 10 * np.log(by3)
        # by3[0] = 0
        # by4 = 10*np.log(by4)
        # by4[0] = 0

        # plt.subplot(411)
        # plt.plot(load)
        #
        # plt.subplot(412)
        # plt.plot(bx, by)
        #
        plt.subplot(211)
        plt.plot(bx2, by2[:, 0])

        plt.subplot(212)
        plt.plot(bx4, by4[:, 0])
        plt.show()
        Hor = data['Horizontal_vibration_signals'].values
        Ver = data['Vertical_vibration_signals'].values
        all_rms.append()



        plt.plot(Hor)
        plt.show()
