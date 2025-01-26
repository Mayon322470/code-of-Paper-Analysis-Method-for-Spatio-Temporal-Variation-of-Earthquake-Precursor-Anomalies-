import numpy as np
import pandas as pd
from datetime import datetime,date

import model_interface
import dst_interface
import detection_interface


# @Time    : 2023/6/9
# @Author  : xu bj
def get_csv_by_time(filepath, start_time, end_time):
    """
    :param filepath:
    :param start_time:
    :param end_time:
    :return:
    """
    st = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    ed = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(filepath, index_col=0)
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= st) & (df['time'] <= ed)]
    return df


def day_calculator(start_time, end_time):
    days = (datetime.strptime(end_time, '%Y-%m-%d') - datetime.strptime(start_time, '%Y-%m-%d')).days + 1
    return days


def train_example():
    # 模型训练，以垂直分量为例，训练使用GPU若要使用cpu自行去除__gpu_init方法
    # 此处为使用示例，数据的输入自行处理，需要保证无缺失且完整，处理为list输入即可
    file_path = "example-data/62004_1_completed.csv"
    start_time = '2021-05-15 00:00:00'
    end_time = '2021-07-15 23:59:00'
    df = get_csv_by_time(file_path, start_time, end_time)
    train_ori_data = df['vertical'].values.tolist()
    model = model_interface.Model("test-example")
    day = day_calculator('2021-05-15', '2021-07-15')
    model.train(train_ori_data, day)


def predict_and_draw_example():
    # 使用模型预测，同样以垂直分量为例
    # 因为是示例，所以只使用2个月数据进行训练，且周期较短，所以拟合精度不算特别高
    # 但是已经明显能够看到两个曲线的趋势是一致的，数据已经基本拟合，只需增加数据量和训练的epoch即可
    file_path = "example-data/62004_1_completed.csv"
    start_time = '2021-07-15 00:00:00'
    end_time = '2021-09-14 23:59:00'
    df = get_csv_by_time(file_path, start_time, end_time)
    pre_ori_data = df['vertical'].values.tolist()
    model = model_interface.Model("test-example-U1")
    ori, pre = model.predict(pre_ori_data)  # 此处返回观测值和预测值，用于后续异常检测
    model.draw_comparison_fig('2021-07-16', 5, 'vertical-example')


def detect_example():
    # 异常检测示例，同样以垂直分量为例
    mgn_file_path = "example-data/62004_1_completed.csv"
    mgn_start_time = '2021-07-15 00:00:00'
    mgn_end_time = '2021-09-14 23:59:00'
    df = get_csv_by_time(mgn_file_path, mgn_start_time, mgn_end_time)
    pre_ori_data = df['vertical'].values.tolist()
    model = model_interface.Model("test-example")
    ori, pre = model.predict(pre_ori_data)

    dst_file_path = "example-data/dst_2019-2022.csv"
    # 滑动四分位距需要前30天数据作为窗口,所以时间从预测开始时间往后30天
    dst_start_time = '2021-08-15'
    start_time = '2021-07-15'


    date_differ = date.fromisoformat(dst_start_time) - date.fromisoformat(start_time)
    print(date_differ)


    dst_end_time = '2021-09-14'
    dst = dst_interface.Dst(dst_file_path)
    time_stamp, dst_res = dst.get_dst_abnormal_day(dst_start_time, dst_end_time)  # 判断磁暴，1为当日有磁暴，0为无磁暴
    detect = detection_interface.Detect(detection_win=30)
    res = detect.abnormal_detect(ori, pre, dst_res, 0)   # 该结果为去除磁暴异常的结果,用于后续区域异常指数热图计算(可以不去磁暴，但若要进行后续计算必须去磁爆)
    detect.draw_abnormal_detection_fig(dst_start_time, 5, 'abnormal_detect-example')


def heatmap_example():
    # 区域异常检测示例
    # 该文件结果相当于abnormal_detect检测的 4个分量的结果，这里仅做示范此示例与之前示例没有关联
    file_path = "example-data/62004_1.csv"
    df = pd.read_csv(file_path, index_col=0)
    detect = detection_interface.Detect(detection_win=30)
    coordinate_X, coordinate_Y = detect.coordinate_calculator(97, 104, 36, 40)  # 返回生成的矩阵网格横纵坐标
    matrix = np.array(df.values)
    matrixs = np.reshape(matrix, (1, matrix.shape[0], matrix.shape[1]))  # 3维数据，表示多个台站数据，此处只有单台站但也须将数据升维
    longitudes = [101.036] # 台站经度list，数量及顺序对应 matrixs
    latitudes = [38.77] # 台站纬度list
    detect.draw_heatmap(matrixs, longitudes, latitudes, "heatmap-example")


# def draw_four_type():
#     # 绘制单个台站四分量，时间内的原始数据


if __name__ == '__main__':
    #########################################################
    # Model类相关示例
    # 训练示例
    # train_example()
    # 预测及绘图示例
    predict_and_draw_example()
    # print("example.py,3")
    ##############################################################
    # Detect类相关示例
    # 异常检测结果示例
    # detect_example()
    # 区域异常指数热图示例
    heatmap_example()
