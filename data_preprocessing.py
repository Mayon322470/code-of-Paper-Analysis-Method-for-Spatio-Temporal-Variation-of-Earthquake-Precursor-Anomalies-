import pandas as pd
import numpy as np


# @Time    : 2023/6/26
# @Author  : 周飞
class MagneticData:
    magneticdata_filepath = 'F:/magneticdata/magMinutes/UT'
    # 补全门槛
    complement_threshold = 0.2
    # qt界面直接从__file_dict里面拿了key，将keys作为下拉框的选项（选项会不会有点太多了？做到了有问题再说）
    # 必须只支持磁场3、4、5、7四种分量的训练的
    __stationid_dict = dict()
    __pointid_dict = dict()
    __type_dict = dict()
    __dictfile_path = 'dict-file/105211141.csv'


    def initial_dict(self):
        """
        初始化两个字典（文件索引字典，四分量和{'3123', '3124', '3125', '3127'}之间的转换字典）
        方便后续查找目标文件名（字典中仅保留标准磁场数据的文件夹名）
        """
        row_dict = pd.read_csv(self.__dictfile_path, sep=',', usecols=['STATIONID', 'POINTID', 'STATIONNAME', 'ITEMID'])
        row_dict = row_dict[['STATIONID', 'POINTID', 'STATIONNAME', 'ITEMID']]  # 确定读取出来的表中，列的先后顺序
        # 1、初始化文件索引字典
        for i in range(row_dict.shape[0]):
            # 获取第i行的所有列
            current_station = row_dict.loc[i][:]
            # 消除读取出的数字开头的0
            if not current_station[0][0].isalpha():
                current_station[0] = str(int(current_station[0]))
            # 寻找磁场文件
            if current_station[3] in {'3123', '3124', '3125', '3127'}:
                # 存两个map，《台站名称，文件号》，《台站名称，《台站编号1，台站编号2……》》
                self.__stationid_dict[current_station[2]] = current_station[0]
                if current_station[2] not in self.__pointid_dict:
                    temp_set = set()
                    temp_set.add(current_station[1])
                    self.__pointid_dict[current_station[2]] = temp_set
                else:
                    self.__pointid_dict[current_station[2]].add(current_station[1])

        # 2、初始化分量转换的字典
        self.__type_dict['垂直分量'] = '3123'
        self.__type_dict['水平分量'] = '3124'
        self.__type_dict['磁偏角'] = '3125'
        # self.__type_dict['总场'] = '3127'
        return self.__stationid_dict, self.__pointid_dict, self.__type_dict

    def data_complement(self, row_data: [], complement_threshold: float = 0.2):
        """
        插值补全
        :param row_data:列表类型的原始数据
        :param complement_threshold:补全门槛，一般设置为20%，表示超过20%数据没有就补不了
        :return:返回补全好的list类型的数据，可能为[]空，表示未达到补全门槛。
        """
        x_complement = []
        x = []
        y = []
        for i in range(1440):
            if row_data[i] == 999999:
                x_complement.append(i)
            else:
                x.append(i)
                y.append(row_data[i])
        if len(x_complement) > 1440 * complement_threshold:
            return []
        y_complement = np.interp(x_complement, x, y)
        for i in range(len(x_complement)):
            row_data[x_complement[i]] = round(y_complement[i], 2)
        return row_data

    def get_file_data(self, magneticdata_filepath, station_folder: str, data_time: str, component_type: str):
        """
        获取目标文件数据，返回值为list类型
        :param magneticdata_filepath: 数据文件的绝对路径，例：F:/magneticdata/magMinutes/UT后面（/13001_1）。
        :param station_folder:目标文件夹的名称，例：13001_1。这个格式的文件名直接由__file_dict中的value获得即可。
        :param data_time:目标日期，输入格式必须为'%Y-%m-%d'，即年月日，中间以连接符-连接。
        :param component_type:分量,接收三种情况，3123（水平），3124（垂直），3125（磁偏角），直接输入数字即可。
        :return:返回list类型的数据
        """
        self.magneticdata_filepath = magneticdata_filepath
        # 拼接目标路径
        target_path = self.magneticdata_filepath + '/' + station_folder
        data_time_array = data_time.split('-')
        data_time = ''
        for time in data_time_array:
            data_time += time
        target_path = target_path + '/' + data_time + '_' + str(component_type) + '.txt'
        # 读取文件数据
        res = pd.read_csv(target_path, sep=' ', header=None)
        res.columns = ['time', 'data']
        res = res.loc[:, 'data'].to_list()
        do_fit = False
        abnormal_count = 0
        for i in range(len(res)):
            if res[i] == 999999:
                # print(data_time)
                # print(i)
                abnormal_count += 1
                do_fit = True
        if abnormal_count > len(res) * self.complement_threshold:
            # 未达到补全要求
            print(f"数据不全，缺失过多无法补全。（{station_folder}台站{component_type}分量{data_time}日）")
            return []
        # de_fit为True表示需要进行补全，会运行到本行说明达到补全要求
        if do_fit == True :
            print(f"{station_folder}台站{component_type}分量在{data_time}日中缺失数据，但是达到补全标准")
            list_res = self.data_complement(self, res, 0.2)
            return list_res
        else:
            # do_fit为false表示数据是全的，不需要补全。
            return res


    def fit(data_raw, f_n=48):
        # n阶傅里叶拟合（48阶）
        if f_n == 0:
            return data_raw
        l = len(data_raw)
        x = range(1, l + 1)
        x = np.array(x)
        l = l - 1
        y = np.array(data_raw)
        y_fit = np.zeros(len(data_raw))
        a0 = 1 / l * np.trapz(y, x)
        for i in range(1, f_n + 1):
            bn = (2 / l) * np.trapz(y * np.sin(2 * i * np.pi * x / l), x)
        an = (2 / l) * np.trapz(y * np.cos(2 * i * np.pi * x / l), x)
        y_fit = y_fit + bn * np.sin(2 * i * np.pi / l * x) + an * np.cos(2 * i * np.pi / l * x)
        data_fit = y_fit + a0
        return data_fit
