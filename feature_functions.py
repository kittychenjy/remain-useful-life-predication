import numpy as np
from scipy import fft, stats, signal
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd


class VibrationSignal:

    def __init__(self, an_np_array, signal_axis=-1):

        self.shape = an_np_array.shape
        self.dimension = len(self.shape)
        self.axis = signal_axis
        self.signal_len = self.shape[self.axis]

        if self.dimension not in [1, 2]:
            raise ValueError('Input array should be 1-d or 2-d.')

        else:
            if self.axis not in [0, 1, -1]:
                raise IndexError('tuple index out of range')
            else: self.data = an_np_array

        self.max = np.amax(self.data, axis=self.axis)
        self.p2p = np.amax(self.data, axis=self.axis) - np.amin(self.data, axis=self.axis)
        self.mean = np.mean(self.data, axis=self.axis)
        self.var = np.var(self.data, axis=self.axis)
        self.std = np.std(self.data, axis=self.axis)
        self.skewness = stats.skew(self.data, axis=self.axis)
        self.kurtosis = stats.kurtosis(self.data, axis=self.axis)

    # ----------------------  some other features ------------------------------
    def get_rms(self):
        after_square = np.square(self.data)
        after_sum = np.sum(after_square, axis=self.axis)
        after_average = np.divide(after_sum, self.signal_len)
        rms_result = np.sqrt(after_average)

        return rms_result
    def get_kurtosis(self):
        return self.kurtosis
    def get_smra(self):  # 方根幅值
        after_abs = abs(self.data)
        after_sqrt = np.sqrt(after_abs)
        after_sum = np.sum(after_sqrt, axis=self.axis)
        after_average = np.divide(after_sum, self.signal_len)
        smra_result = np.square(after_average)

        return smra_result

    def get_shape_factor(self):  # 波形指标
        after_abs = abs(self.data)
        after_average = np.mean(after_abs, axis=self.axis)
        shape_factor = self.get_rms() / after_average

        return shape_factor

    def get_crest_factor(self):  # 峰值指标
        crest_factor = self.max / self.get_rms()

        return crest_factor

    def get_impulse_factor(self):  # 脉冲指标
        after_abs = abs(self.data)
        after_average = np.mean(after_abs, axis=self.axis)
        impulse_factor = self.max / after_average

        return impulse_factor

    def get_clearance_factor(self):  # 裕度指标
        clea_fac = self.max / self.get_smra()

        return clea_fac

    def get_kurtosis_factor(self):  # 峭度指标
        kur_fac = self.kurtosis / pow(self.get_rms(), 4)

        return kur_fac

    def get_waveform_entropy(self, AVP):
        k = np.sqrt((self.data ** 2).mean(axis=1)) / ((abs(self.data)).mean(axis=1))
        out2_m = np.zeros(len(k))
        for i in range(len(k)):
            if i >= AVP and i <= (len(k) - AVP):
                out2_m[i] = np.mean((k[i - AVP:i + AVP]) * np.log(k[i - AVP:i + AVP]))
            else:
                out2_m[i] = k[i] * np.log(k[i])
        # out2_m[np.isnan(out2_m)] = 0
        return out2_m

    #  ------------------------------ plotting sources --------------------------------------------
    @staticmethod
    def get_start_end_point(fs, num_dots, start_freq, end_freq):  # get fft start and end point index
        if end_freq is None:
            end_freq = int(fs / 2)
        if end_freq > fs / 2:
            raise ValueError("end frequency shouldn't be higher than half of fs.")
        if start_freq > end_freq:
            raise ValueError("start frequency shouldn't be higher than end frequency.")

        start_point = int(start_freq * num_dots / fs)
        end_point = int(end_freq * num_dots / fs)

        return start_point, end_point

    def get_fft(self, fs, start_freq=0, end_freq=None, data=None, dc_to_zero=True):  # fft 序列

        if data is None:
            data = self.data

        x_seq_whole = np.linspace(0, fs, self.signal_len, endpoint=False)
        fft_data_whole = abs(fft(data, axis=self.axis))

        start_point, end_point = self.get_start_end_point(fs, self.signal_len, start_freq, end_freq)
        x_seq = x_seq_whole[start_point:end_point]

        # modify zero frequency component to zero
        if dc_to_zero is True:
            if self.dimension == 2 and self.axis in [-1, 1]:
                fft_data_whole.transpose()[0] = 0
                fft_data = 2 * (fft_data_whole / self.signal_len)[:, start_point:end_point]

            else:
                fft_data_whole[0] = 0
                fft_data = 2 * (fft_data_whole / self.signal_len)[start_point:end_point]

        elif dc_to_zero is False:
            if self.dimension == 2 and self.axis in [-1, 1]:
                fft_data = 2 * (fft_data_whole / self.signal_len)[:, start_point:end_point]

            else:
                fft_data = 2 * (fft_data_whole / self.signal_len)[start_point:end_point]

        else:
            raise ValueError('dc_to_zero should either be True of False')

        return x_seq, fft_data

    def get_square_demodulation(self, fs, start_freq=0, end_freq=None):  # 平方解调序列
        data = np.square(self.data)

        return self.get_fft(fs, start_freq, end_freq, data)

    def get_psd_by_correlation(self, fs, start_freq=0, end_freq=None):  # 用自相关求psd序列（暂未正确scale幅值）
        if self.dimension == 1:
            data = np.correlate(self.data, self.data, 'same')

        elif self.axis == 0:
            data = np.zeros(self.shape)
            for i in range(self.shape[1]):
                data[:, i] = np.correlate(self.data[:, i], self.data[:, i], 'same')

        else:
            data = np.zeros(self.shape)
            for i in range(self.shape[0]):
                data[i, :] = np.correlate(self.data[i, :], self.data[i, :], 'same')

        return self.get_fft(fs, start_freq, end_freq, data)

    def get_psd_by_square(self, fs, start_freq=0, end_freq=None):  # 平方法求psd序列
        fft_x, fft_y = self.get_fft(fs, start_freq, end_freq, self.data, dc_to_zero=True)
        fft_y_square = np.square(fft_y)
        psd_array = fft_y_square * self.signal_len / (2 * fs)

        return fft_x, psd_array

    def get_psd_using_scipy(self, fs, start_freq=0, end_freq=None):  # 用scipy方法直接求psd序列
        psd_x_whole, psd_y_whole = signal.periodogram(self.data, fs, axis=self.axis)
        start_point, end_point = self.get_start_end_point(fs, self.signal_len, start_freq, end_freq)

        return psd_x_whole[start_point: end_point], psd_y_whole[start_point: end_point]

    def get_cepstrum_sequence(self):  # 倒频谱序列
        power_x, power_y = self.get_power_spectrum_sequence()


# for code testing
if __name__ == '__main__':

    # load1 = sio.loadmat('D:/Spindle_Program/Projects/Data_Repository/MT2/inner_0.6_0.04'
    #                    '/mat_data/20190422151037_inner_0.6_0.04_al7075_7000rpm_depth0.1_width3_feed1200.mat')['spindle_z']
    # load2 = sio.loadmat('D:/Spindle_Program/Projects/Data_Repository/MT2/inner_0.6_0.04'
    #                    '/mat_data/20190422151037_inner_0.6_0.04_al7075_7000rpm_depth0.1_width3_feed1200.mat')['spindle_x']
    # load = np.vstack((load2, load1))[:, 51200:76800]
    # load = load.transpose()
    # print(load.shape)
    load = pd.read_csv('../Data_Repository/PHM2010/c1/c1/c_1_001.csv').values[50000:100000]

    # t = np.arange(0, 1, 1/1000)
    # load = np.cos(2 * np.pi * 100 * t) + np.random.randn(t.size)

    # a = np.array([[1, 2, 3],[3,6,9]])
    b = VibrationSignal(load, signal_axis=0)
    bx, by = b.get_fft(50000)
    bx1, by1 = b.get_square_decomposition(50000)
    bx2, by2 = b.get_psd_by_correlation(50000)
    by2 = by2 / 50000
    bx3, by3 = b.get_psd_by_square(50000)
    bx4, by4 = b.get_psd_using_scipy(50000)
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
