import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# @Time    : 2023/6/6
# @Author  : xu bj
class Model:
    """
    LSTM模型类
    # data  数据
    # day   数据长度以天为单位 /天
    """
    data = []
    day = 0
    model_name = None
    ori = []
    pre = []
    __model_path = "Model"
    __fig_path = "Fig"
    __miss_number = 999999
    # __down_window = 3
    # __seq_length = 480
    __down_window = 15
    __seq_length = 96
    __nor_min = 0
    __nor_max = 0
    __nor_name = "nor_params.npy"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def train(self, data: [], day: int, batch_size: int = 128, epochs: int = 100):
        """
        模型训练函数
        :param data: 数据 切片类型
        :param day: 数据天数
        :param batch_size: 默认128
        :param epochs: 默认100
        :return:
        """
        self.data = data
        self.day = day
        # 数据校验
        self.__integrity_check()
        # 数据预处理
        train_X, test_X, train_Y, test_Y = self.__processor()
        # 设置 GPU,可根据电脑自行修改相关参数
        self.__gpu_init()
        model = self.__lstm(train_X.shape[1:], train_Y.shape[1])
        model.summary()
        # 之后若要介入模型训练可从这开始，比如返回loss之类的
        history = model.fit(
            train_X,
            train_Y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_X, test_Y),
        ).history
        # 检查是否存在目录，不存在则创建
        if not os.path.exists(self.__model_path):
            os.makedirs(self.__model_path)
        model.save(self.__model_path + "/" + self.model_name)
        if not os.path.exists(self.__model_path + "/" + self.model_name):
            os.makedirs(self.__model_path + "/" + self.model_name)
        self.__save_nor_param()
        print("模型训练完成")

    def predict(self, data: [], model_absDir: str = ""):
        """
        预测数据
        :param data:预测数据序列，不少于2天的数据
        :return: 原始数据（去除了第一天），预测数据
        """
        self.__gpu_init()
        my_model_path = self.__model_path + "/" + self.model_name  # 默认不使用绝对路径
        if model_absDir != "":  # 如果使用绝对路径，则需要添加
            my_model_path = model_absDir + "/" + my_model_path
        my_model = keras.models.load_model(my_model_path)
        ori_data = self.__down_sampling(data)
        # self.__get_nor_param()
        self.__nor_max = np.max(data)
        self.__nor_min = np.min(data)
        input_data = (ori_data - self.__nor_min) / (self.__nor_max - self.__nor_min)
        prepare_data = self.__pre_processor(input_data)
        pre = my_model.predict(prepare_data)
        res = pre.reshape(-1)
        self.pre = self.__denormalization(res)
        self.ori = ori_data[self.__seq_length:] # 抛弃了最初的第一天的数据，为的是输出的ori和pre是时间维度完全相同
        return self.ori, self.pre

    def seq_edit(self, seq, edit_num):
        for i in range(len(seq)):
            seq[i] = seq[i] - edit_num
        return seq

    def draw_comparison_fig(self, start_time: str, interval: int, fig_name: str):
        """
        绘制对比图
        :param start_time:起始时间
        :param interval: 横坐标间距，以日为单位
        :param fig_name：保存图片的名称
        :return:
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(50, 10))
        # 曲线 1，2
        time_stamp = pd.date_range(start=start_time, periods=len(self.pre), freq='15min')
        # time_stamp = pd.date_range(start=start_time, periods=len(self.pre), freq='3min') # 3-13
        L1, = plt.plot(time_stamp, self.ori, label='raw data')
        # print(self.pre)
        # print(self.ori)
        L2, = plt.plot(time_stamp, self.pre, color='red', label='forecast data')
        # L2, = plt.plot(time_stamp, self.seq_edit(self.pre,500000), color='red', label='forecast data')
        # 设置图例，按需修改
        plt.legend(handles=[L1, L2], labels=['raw data', 'forecast data'], loc='upper right', fontsize=30)
        # 设置横坐标
        # plt.ylabel("magnetic declination/°",fontproperties='SimHei', fontsize=30, labelpad=10)
        plt.ylabel("Amplitude/nT", fontproperties='SimHei', fontsize=30, labelpad=10)
        # 设置纵坐标
        plt.xlabel(u'Time/d', fontproperties='SimHei', fontsize=30, labelpad=10)
        # 设置坐标轴刻度大小
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        # 设置x轴参数，时间显示戳单位
        xfmt = matplotlib.dates.DateFormatter('%Y-%m-%d')
        # 设置x轴刻度间隔
        x_major_locator = plt.MultipleLocator(interval)

        # 获取坐标轴实例
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(x_major_locator)

        # 显示倾斜35度
        fig.autofmt_xdate(rotation=35)

        # 自动调整图周围空白
        fig.tight_layout()
        if not os.path.exists(self.__fig_path):
            os.makedirs(self.__fig_path)
        plt.savefig(self.__fig_path + "/" + fig_name + '.svg', dpi=300, format='svg')
        plt.show()

    def __gpu_init(self):
        """
        可根据电脑配置情况，进行修改
        :return:
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # 指定特定的 GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # 按需分配
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

    def __integrity_check(self):
        """
        检查数据完整性及是否缺失
        :return:
        """
        if len(self.data) == 0:
            raise RuntimeError("输入数据为空")
        if int(len(self.data) / 1440) != self.day:
            raise RuntimeError("输入数据与输入天数不对应")
        for i in range(len(self.data)):
            if self.data[i] == self.__miss_number:
                raise RuntimeError("输入数据有缺失项")

    def __processor(self):
        # 降采样
        arr = self.__down_sampling(self.data)
        # 归一化
        nor_data = self.__normalization(arr)
        # 得到训练数据
        train_X, test_X, train_Y, test_Y = self.__get_train_data(nor_data)
        train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
        test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
        return train_X, test_X, train_Y, test_Y

    def __down_sampling(self, data):
        """
        降采样
        :param data: 一维数据
        :return:
        """
        n = int(len(data))
        index = 0
        arr = []
        while index < n:
            tmp = data[index:index + self.__down_window]
            arr.append(np.mean(tmp))
            index += self.__down_window
        return arr

    def __normalization(self, data: []):
        """
        归一化[0,1]
        :param data:
        :return:
        """
        min = np.min(data)
        max = np.max(data)
        data = (data - min) / (max - min)
        self.__nor_max = max
        self.__nor_min = min
        return data

    def __denormalization(self, data: []):
        return data * (self.__nor_max - self.__nor_min) + self.__nor_min

    def __get_train_data(self, data: []):
        """
        得到训练数据
        :param data: 一维数据
        :return:
        """
        predict_length = 1
        data_process = []
        for index in range(len(data) - self.__seq_length - predict_length):
            data_process.append(data[index:index + self.__seq_length + predict_length])

        data_process = np.array(data_process)
        np.random.shuffle(data_process)
        # print(data_process.shape)
        x = data_process[:, :-predict_length]
        y = data_process[:, self.__seq_length:]
        # print(x.shape)
        # print(y.shape)
        # 切分比例
        split_len = int(len(data_process) * 0.75)
        train_X = x[:split_len]
        test_X = x[split_len:]
        train_Y = y[:split_len]
        test_Y = y[split_len:]

        return train_X, test_X, train_Y, test_Y

    def __lstm(self, input_shape, output_size):
        model = tf.keras.Sequential([
            layers.LSTM(100, activation='tanh', input_shape=(input_shape)),
            layers.RepeatVector(input_shape[0]),
            layers.LSTM(100, activation='tanh'),
            layers.Dense(100, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.0001),
                      loss='mse')
        return model

    def __get_nor_param(self):
        arr = np.load(self.__model_path + "/" + self.model_name + "/" + self.__nor_name)
        self.__nor_min = arr[0]
        self.__nor_max = arr[1]

    def __save_nor_param(self):
        params = [self.__nor_min, self.__nor_max]
        np.save(self.__model_path + "/" + self.model_name + "/" + self.__nor_name, params)

    def __pre_processor(self, data):
        data_process = []
        # 最后一个切片不用，因为无法验证
        for index in range(len(data) - self.__seq_length):
            data_process.append(data[index:index + self.__seq_length])
        data = np.array(data_process)
        return np.reshape(data, (data.shape[0], data.shape[1], 1))
