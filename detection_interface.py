import os
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates


# @Time    : 2023/6/7
# @Author  : xu bj
class Detect:
    ori = []
    pre = []
    res = []
    coordinate_X = []
    coordinate_Y = []
    __window = 30
    __dst_index = -40
    __seq_length = 96
    # __seq_length = 480
    __fig_path = "Fig"

    def __init__(self, detection_win: int):
        """
        数据初始化，从model的predict方法获取,两个序列的长度不小于31天
        :param ori:原始观测序列
        :param pre: 预测序列
        """
        self.__window = detection_win

    def abnormal_detect(self, ori: [], pre: [], dst_seq: [], type: int = 0):
        """
        异常检测方法，可以得到异常指数序列
        :param ori: 原始数据序列
        :param pre: 预测数据序列
        :param dst_seq: 地磁扰动指数序列，取每天最小值,若只是想获得异常检测结果则传空值，若想获得去除磁暴后的结果则必须输入
        :param type: 输入数据类型 0：非磁偏角，1：磁偏角
        :return: 处理后的时间序列，1个点为 1天,若去除了磁暴相当于计算了异常指数，序列为 0 或 1
        """
        self.ori = ori
        self.pre = pre
        self.__data_validate()
        data = np.abs(self.pre - self.ori)
        y = self.__get_day_mean(data)
        threshold = 2.5
        if type == 1:
            threshold = 2
        res = self.__slide_the_quartile(y, threshold)
        if len(dst_seq) > 0:
            # dst_seq维度是去掉开头若干天的数据，res的维度必须和dst_seq相同。
            res = self.__remove_magnetic_storm(res, dst_seq)
        self.res = res
        return np.array(self.res)

    def draw_abnormal_detection_fig(self, start_time: str, interval: int, fig_name: str):
        """
        画异常检测的直方图
        :param start_time: 横坐标开始时间
        :param interval: 横坐标间隔
        :param fig_name: 图片名
        :return:
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(30, 5))
        y_stamp = pd.date_range(start=start_time, periods=len(self.res), freq='1D')
        plt.bar(y_stamp, self.res)  # 内容绘制
        plt.ylabel(u'异常值', fontproperties='SimHei', fontsize=20, labelpad=10)
        plt.xlabel(u'时间', fontproperties='SimHei', fontsize=20, labelpad=10)
        plt.xticks(fontsize=18, rotation=300)
        plt.yticks(fontsize=18)
        xfmt = matplotlib.dates.DateFormatter('%m-%d')
        x_major_locator = plt.MultipleLocator(interval)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(x_major_locator)
        plt.tight_layout()
        if not os.path.exists(self.__fig_path):
            os.makedirs(self.__fig_path)
        plt.savefig(self.__fig_path + "/" + fig_name + ".svg", dpi=300, format='svg')
        plt.show()

    def coordinate_calculator(self, lon_start, lon_end, lat_start, lat_end):
        """
        计算网格矩阵边界，返回值留作日后开发
        :param lon_start: 开始经度
        :param lon_end: 结束经度
        :param lat_start: 开始纬度
        :param lat_end: 结束纬度
        :return:
        """
        coordinate_X = np.arange(lon_start, lon_end, 0.1)
        coordinate_Y = np.arange(lat_start, lat_end, 0.1)
        coordinate_X = [round(x, 2) for x in coordinate_X]
        coordinate_Y = [round(y, 2) for y in coordinate_Y]
        self.coordinate_X = coordinate_X
        self.coordinate_Y = coordinate_Y
        return coordinate_X, coordinate_Y

    def draw_heatmap(self, matrixs: [[]], longitudes: [], latitudes: [], fig_name: str, max_value: int = 0):
        """
        绘制累计异常热图
        :param matrixs: 区域台站的累计异常矩阵数组，3维
        :param longitudes: 台站经度数组
        :param latitudes: 台站纬度数组
        :param fig_name: 图片名
        :param max_value: 可选参数，异常上限，默认每个台站均为4分量
        :return:
        """
        fig = plt.figure(figsize=(15, 15), dpi=100)
        ax = plt.gca()
        matrix = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
        for i in range(len(matrixs)):
            matrix += self.__staion_matrix_calculator(matrixs[i], longitudes[i], latitudes[i])
            # matrix += self.__staion_matrix_seq_calculator(matrixs[i], longitudes[i], latitudes[i])
            # matrix += matrixs[i]
        # if max_value == 0:
        #     max_value += 4 * len(matrixs)
        max_value = 6
        # 设置横纵坐标
        ax.set_xticks(range(len(self.coordinate_X)))
        ax.set_xticklabels(self.coordinate_X)
        ax.set_yticks(range(len(self.coordinate_Y)))
        ax.set_yticklabels(self.coordinate_Y)

        # cmap设置颜色，aspect自动调整方块大小（绘图主体代码）
        im = ax.imshow(matrix, aspect='auto', alpha=0.7, vmax=max_value)

        # 增加右侧颜色刻度
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel(u'异常指数', rotation=-90, va="bottom", fontsize=30, fontproperties='SimHei')
        cbar.ax.tick_params(labelsize=20)

        # 设置坐标轴
        plt.ylabel(u'纬度', fontproperties='SimHei', fontsize=30)
        plt.xlabel(u'经度', fontproperties='SimHei', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        x_major_locator = plt.MultipleLocator(5)
        y_major_locator = plt.MultipleLocator(5)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        fig.tight_layout()
        plt.grid(True)
        if not os.path.exists(self.__fig_path):
            os.makedirs(self.__fig_path)
        plt.savefig(self.__fig_path + "/" + fig_name + ".svg", dpi=100, format='svg')
        plt.show()

    def __data_validate(self):
        """
        数据校验
        :return:
        """
        if len(self.pre) == 0 or len(self.ori) == 0:
            raise RuntimeError("输入数据为空")
        if len(self.pre) != len(self.ori):
            raise RuntimeError("输入数据长度未对齐")
        if len(self.pre) % 96 != 0:
            raise RuntimeError("输入数据长度不完整，1天应为96个点")

    def __distance_calculator(self, lon1, lat1, lon2, lat2):
        """
        计算两点间距离
        :param lon1:
        :param lat1:
        :param lon2:
        :param lat2:
        :return:
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        distance = c * r
        return distance

    def __get_day_mean(self, data):
        """
        获得一天的绝对平均值
        :param data: 数据
        :param window: 一天的数据点个数
        :return:
        """
        n = len(data)
        index = 0
        res = []
        while index < n:
            cur = data[index:index + self.__seq_length]
            cur = np.array(cur)
            cur = np.mean(cur)
            res.append(cur)
            index += self.__seq_length
        return np.array(res)

    def __slide_the_quartile(self, data, threshold):
        """
        求滑动四分位距值
        :param data: 数据，是预测值和实际值的差值，维度是预测维度
        :param threshold: 阈值，滑动四分位异常值判定的阈值。
        :return:
        """
        index = 0
        n = len(data)
        res = []
        while index < n - self.__window:
            tmp = data[index:index + self.__window]
            tmp = np.sort(tmp)
            Q1 = int(self.__window / 4)
            mid = int(self.__window / 2)
            Q3 = int(3 * self.__window / 4)
            IQR = tmp[Q3] - tmp[Q1]
            up = tmp[mid] + threshold * IQR
            lower = tmp[mid] - threshold * IQR
            point = data[index + self.__window]
            if point > up or point < lower:
                res.append(point)
            else:
                res.append(0)
            index += 1
        return np.array(res)

    def __remove_magnetic_storm(self, data, dst_seq):
        """
        计算异常指数
        :param data: 数据序列，1天1个点
        :param dst_seq: dst序列，取每日最小值
        :return:
        """
        # print(f"data:{len(data)}")
        # print(f"dst_len:{len(dst_seq)}")
        if len(data) != len(dst_seq):
            raise RuntimeError("地磁数据长度与磁爆数据长度未对齐")
        for i in range(len(data)):
            # 磁暴日直接判断为正常，如果不是磁暴日，就不会置零，这个时候如果四分位结果参数data处的值本来就不为0，则判定为异常。
            if dst_seq[i] == 1:
                data[i] = 0
            if data[i] != 0:
                data[i] = 1
        return data

    def __staion_matrix_seq_calculator(self, ab_res: [[]], longitude, latitude):
        """
        计算单台站异常指数矩阵序列
        :param coordinate_X: 网格 x 轴经度坐标
        :param coordinate_Y: 网格 y 轴纬度坐标
        :param ab_res: 单台站的所有分量的异常检测结果，2维切片
        :param longitude: 台站经度
        :param latitude: 台站纬度
        :return:
        """
        tmp = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
        for i in range(len(tmp)):
            for j in range(len(tmp[0])):
                if self.__distance_calculator(longitude, latitude, self.coordinate_X[j], self.coordinate_Y[i]) <= 200:
                    tmp[i][j] = 1
        pos = np.where(tmp == 1)
        res = []
        for index, row in ab_res.iterrows():
            sum = 0
            cur = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
            for i in range(len(row)):
                sum += row[i]
                cur[pos] = sum
            res.append(cur)
        tensor = np.array(res)
        return tensor

    def __staion_matrix_calculator(self, ab_res: [[]], longitude, latitude):
        """
        计算单台站累计异常指数矩阵
        :param coordinate_X: 网格 x 轴经度坐标
        :param coordinate_Y: 网格 y 轴纬度坐标
        :param ab_res: 单台站的所有分量的异常检测结果，2维切片，30天数据
        :param longitude: 台站经度
        :param latitude: 台站纬度
        :return:
        """
        # if len(ab_res) != 30:
        #     raise RuntimeError("输入异常检测序列不等于30天")
        tmp = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
        for i in range(len(tmp)):
            for j in range(len(tmp[0])):
                if self.__distance_calculator(longitude, latitude, self.coordinate_X[j], self.coordinate_Y[i]) <= 200:
                    tmp[i][j] = 1
        pos = np.where(tmp == 1)
        matrix = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
        for index in range(len(ab_res)):
            sum = 0
            row = ab_res[index]
            cur = np.zeros((len(self.coordinate_Y), len(self.coordinate_X)))
            for i in range(len(row)):
                sum += row[i]
                cur[pos] = sum
            matrix += cur
        return matrix
