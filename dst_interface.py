import numpy as np
import pandas as pd
from datetime import datetime


# @Time    : 2023/6/9 16:11
# @Author  : xu bj
class Dst:
    dst_path = None

    def __init__(self, dst_path: str):
        self.dst_path = dst_path

    def get_dst_abnormal_day(self, start_time, end_time):
        """
        获取时间区间内，磁暴日期，当日最低小于-40认为发生磁爆
        :param start_time:
        :param end_time:
        :return:
        """
        df = self.__get_dst_df(start_time, end_time)
        time_stamp = []
        res = []
        for index, row in df.iterrows():
            tmp = np.array(row[1:])
            min = np.min(tmp)
            if min <= -40:
                time_stamp.append(row[0])
                res.append(1)
            else:
                res.append(0)
        return time_stamp, res

    def __get_dst_series(self, start_time, end_time):
        """
        获取磁爆序列
        :param start_time:
        :param end_time:
        :return:
        """
        df = self.__get_dst_df(start_time, end_time)
        df = df.drop(['mean'], axis=1)
        series = []
        for i in range(len(df)):
            #     series.extend(df.iloc[i].values.tolist())
            cur = df.iloc[i].values.tolist()
            cur.pop(0)
            series.extend(cur)
        days = (datetime.strptime(end_time, '%Y-%m-%d') - datetime.strptime(start_time, '%Y-%m-%d')).days
        time_stamp = pd.date_range(start=start_time + " 01:00:00", periods=(days + 1) * 24, freq='1h')
        return series, time_stamp

    def __get_dst_df(self, start_time, end_time):
        df = self.__dst_reader()
        st = datetime.strptime(start_time, '%Y-%m-%d')
        ed = datetime.strptime(end_time, '%Y-%m-%d')
        df['time'] = pd.to_datetime(df['time'])
        df = df[(df['time'] >= st) & (df['time'] <= ed)]
        return df.reset_index(drop=True)

    def __dst_reader(self):
        return pd.read_csv(self.dst_path, index_col=0)
