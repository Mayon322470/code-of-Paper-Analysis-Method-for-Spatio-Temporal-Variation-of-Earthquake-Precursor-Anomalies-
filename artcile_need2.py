import csv
import os
import pandas as pd
import numpy as np
import data_preprocessing
import model_interface
import datetime
import dst_interface
import detection_interface


class use_interface:
    # 类中定义了实验室当前情景读取数据的方法，以及需要的参数
    """
    1、往下两个函数，实现了预测绘图、滑动四分位离群点绘图
    """
    def plot_predict(self, magneticdata_filepath, start_time: str, end_time: str, station, point_id, type):
        """
        第一张结果图，函数内完成了 1）读取原始数据，2）进行数据预测，3）绘制预测曲线，一系列操作。
        :param type: 全部统一使用3123类的写法
        """
        # 获取数据并进行预测
        station_folder = station + '_' + point_id
        # 遍历开始时间到结束时间之间的所有日期。
        date_list = self.getEveryDay(start_time, end_time)
        pre_ori_data = []
        magneticData = data_preprocessing.MagneticData
        # 对输入的地址，因为复制过来的地址都是\分隔的，如F:\magneticdata\magMinutes\UT，做一个字符串字符替换
        magneticdata_filepath = magneticdata_filepath.replace('\\', '/')
        for date in date_list:
            pre_ori_data += magneticData.get_file_data(magneticData, magneticdata_filepath, station_folder, date, type)
        model = model_interface.Model("test-example-{}".format(type))
        ori, pre = model.predict(pre_ori_data)

        # 绘图，第2个参数是日期间隔长度，第三个参数是图片名称
        # 第一个参数是绘图开始日期，需要注意，网络得到的ori、pre是不包括第一天的，所以日期是start的后一天。
        model.draw_comparison_fig(self.get_tomorrow(start_time), 5, f"{station_folder}_{type}type_{start_time}_{end_time}")
        return ori, pre

    def plot_detect(self, ori, pre, start_time, end_time, detection_win, type, station_id):
        """
        :param detection_win:滑动窗长度
        :param type: 表示分量类型（字符串），'3123'表示垂直分量，'3124'表示水平分量，'3125'表示磁偏角
        """
        # 异常检测示例，同样以垂直分量为例
        dst_file_path = "example-data/dst_2019-2022.csv"
        # 滑动四分位距需要前30天数据作为窗口,所以时间从预测开始时间往后30天
        dst_start_time = (datetime.datetime.strptime(start_time, "%Y-%m-%d") +
                          datetime.timedelta(days=detection_win)).strftime("%Y-%m-%d")
                          # datetime.timedelta(days=detection_win + 1)).strftime("%Y-%m-%d")
        dst_end_time = end_time
        dst = dst_interface.Dst(dst_file_path)
        time_stamp, dst_res = dst.get_dst_abnormal_day(dst_start_time, dst_end_time)  # 判断磁暴，1为当日有磁暴，0为无磁暴
        detect = detection_interface.Detect(detection_win)
        detect_type = 0
        if type == '3125':
            detect_type = 1
        res = detect.abnormal_detect(ori, pre, dst_res,detect_type)
        # 该结果为去除磁暴异常的结果,用于后续区域异常指数热图计算(可以不去磁暴，但若要进行后续计算必须去磁爆)

        # 绘图，注意，绘图的开始日期就是start_time往后滑动四分位滑窗长天的日期。
        detect.draw_abnormal_detection_fig(dst_start_time, 5, f"abnormal_{station_id}_{type}type_{start_time}_{end_time}")
        # 保存结果数据
        return res

    """
    2、往下3个函数，是日期相关的工具类
    """
    def getEveryDay(self,begin_date, end_date):
        """
        获取两个日期间的所有日期
        """
        date_list = []
        begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        while begin_date <= end_date:
            date_str = begin_date.strftime("%Y-%m-%d")
            date_list.append(date_str)
            begin_date += datetime.timedelta(days=1)
        return date_list

    def get_yesterday(self, date:str):
        """
        计算给出日期的前一天
        """
        return (datetime.datetime.strptime(date,'%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    def get_tomorrow(self, date:str):
        """
        计算给出日期的后一天
        """
        return (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    """
    3、四分位结果存储成csv文件的两个函数
    """
    def create_csv(self, file_name, start_date, end_date):
        # 先创建文件并存储日期列，此处由于3127分量（总场）默认全0，所以也一起创建了
        with open(f"example-data/{file_name}.csv", "a", encoding="utf-8", newline="") as f:
            # 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f)
            # 构建列表头
            name = ['date', '3127']
            csv_writer.writerow(name)
            # 写入csv文件内容
            date_list = zip(self.getEveryDay(start_date, end_date))
            # 处理维度，需要按列将数据写入csv文件
            # 使用csv库来存储数据，只能遍历元素在每一行的该列写入
            for cur_date in date_list:
                csv_writer.writerow(cur_date)
            print("创建文件成功，并已写好日期列")
            # 5. 关闭文件
            f.close()

    def res_save_csv(self, res, type, file_name):
        df = pd.read_csv(f"example-data/{file_name}.csv")
        df[f"{type}"] = res
        print(f"example-data/{file_name}.csv已更新{type}分量的数据")
        df.to_csv(f"example-data/{file_name}.csv", index=False)


    def heatmap_example(self, file_dir, csv_filename, detection_win, longitudes, latitudes):
        # 区域异常检测示例
        # 该文件结果相当于abnormal_detect检测的 4个分量的结果，这里仅做示范此示例与之前示例没有关联
        # 参数格式示例：1）file_path = "example-data/62004_1.csv"，
        #               2）longitudes = [101.036] # 台站经度list，数量及顺序对应 matrixs
        #               3）latitudes = [38.77] # 台站纬度list
        file_path = f"{file_dir}/{csv_filename}.csv"
        df = pd.read_csv(file_path, index_col=0)
        detect = detection_interface.Detect(detection_win)
        coordinate_X, coordinate_Y = detect.coordinate_calculator(97, 104, 36, 40)  # 返回生成的矩阵网格横纵坐标
        matrix = np.array(df.values)
        matrixs = np.reshape(matrix, (1, matrix.shape[0], matrix.shape[1]))  # 3维数据，表示多个台站数据，此处只有单台站但也须将数据升维
        detect.draw_heatmap(matrixs, longitudes, latitudes, f"heatmap_{csv_filename}")


if __name__ == '__main__':
    # 测试，先测试代码是否会因为数据缺失，造成异常判定不准确。
    # 基本参数设置（测试只需要修改这里的参数）
    selfObj = use_interface()
    magneticdata_filepath = 'F:/magneticdata/magMinutes/UT'
    # 1楚雄
    # station = '53005'
    # point_id = '2'
    # start_date = '2021-01-01'
    # # start_date = '2021-03-01'
    # end_date = '2021-05-21'

    # # 2永胜
    # station = '53006'
    # point_id = '3'

    # 3云龙
    # station = '53008'
    # point_id = '5'

    # 4射阳（沿海应该无地震，看能不能跟前几个台站有些差别，能够判出来无异常）
    # 异常值为4，3个分量3个月，也相当于一个分量1个异常值，相当于几乎没有。
    # station = '32023'
    # point_id = '2'
    # start_date = '2021-08-17'
    # end_date = '2022-02-17'
    # end_date = '2022-02-01'
    # 5宿迁台
    # 异常值为6
    # station = '32025'
    # point_id = '3'
    # 6海安台
    # 异常值为4（如果是21211017到20211217的话，异常值为0）
    # station = '32028'
    # point_id = '2'

    # 若干反例（无地震）
    # station = '32022'
    # point_id = '1'

    # 7山丹台
    # station = '62004'
    # point_id = '1'
    # start_date = '2021-10-16'
    # end_date = '2022-01-09'
    # 8古丰台（数据不全，当前代码无法运行，做不全，但很明显也符合。）
    # station = '62149'
    # point_id = '3'
    # start_date = '2021-11-01'
    # end_date = '2022-01-09'

    # 9（超出200km，没算）（22年四川有多处地震，可作为实验样例）
    # station = '51016'
    # point_id = '2'

    # 10道孚台51015-1，3，4（22年四川有多处地震，可作为实验样例）
    # 2022-09-05，6.8级，传感器距离震中约190km
    # 2022-06-01，6.1级
    # 只有道孚台4站有查出8个异常
    station = '51015'
    point_id = '4'
    # start_date = '2022-01-10'
    # # start_date = '2022-07-02'
    # end_date = '2022-05-29'
    start_date = '2022-06-05'
    # # start_date = '2022-07-02'
    end_date = '2022-09-05'

    # 经纬度距离计算：https://www.lddgo.net/convert/distance

    # 11丽江53174-1（数据缺失太多），2（异常值7，从0401开始，合理），3（异常值10）（原20210320缺数，我用0319的数据复制的）（2020年09月01日至2021年01月31日）
    # station = '53174'
    # # point_id = '2'
    # point_id = '3'
    # start_date = '2021-02-01'
    # # start_date = '2021-03-01'
    # end_date = '2021-05-21'

    # 12四川2022-06-10地震，台站62097-2（缺数太严重，完全没法补全），3（也缺数），4（缺）
    # station = '62097'
    # # point_id = '2'
    # point_id = '4'
    # start_date = '2022-02-15'
    # # start_date = '2021-03-15'
    # end_date = '2022-06-15'

    # 13大武，63019-8（异常值20，看了日均，确实比预测明显低一点，但是原始数据肉眼很难观测出来），9，b（0527缺数，复制的0526数据，异常值21）。
    # 地震同12
    # station = '63019'
    # # point_id = '2'
    # point_id = 'b'

    # 14仍是道孚台51015-1、3、4、5，这次跟前一次地震不一样，这次地震是12条的2022-06-10的地震
    # 1站：7个异常，但是预测下来拟合的太好了，能有7个异常纯属巧合。后试了20220315开始，异常就变成3了，严格来说，这个没查出来（也可能是没有异常）
    # 3站：14个异常
    # 4站：异常多的普遍都是第2分量，也确实是2分量预测的效果不好，我怀疑可能正常的波形也容易出现异常。9个（可靠性存疑）。
    # 5站：缺数。
    # station = '51015'
    # # point_id = '2'
    # point_id = '5'

    # 15地震2021-05-22青海果洛，台站大武，63019-8，9，b
    # 8站：地震后的天数检测出不少异常，比如1分量的5.23-5.30。算了，反正0101开始或者0122开始，都检测出超10个异常。
    # 9站：缺数严重
    # b站：7个
    # station = '63019'
    # # point_id = '9'
    # point_id = 'b'
    # start_date = '2021-01-01'
    # # start_date = '2021-03-15'
    # end_date = '2021-05-22'

    # 16若干反例（无地震）
    # station = '32023'
    # point_id = '2'

    # station = '32028'
    # point_id = '2'

    # station = '32022'
    # point_id = '1'

    # station = '32040'
    # point_id = '2'
    #
    # station = '32019'
    # point_id = 'A'
    #
    # station = '32019'
    # point_id = '3'

    # station = '34004'
    # point_id = '1'

    # station = '34004'
    # point_id = '2'

    # start_date = '2020-01-01'
    # end_date = '2020-03-01'
    # start_date = '2020-02-01'
    # end_date = '2020-04-01'
    # start_date = '2020-03-01'
    # end_date = '2020-05-01'
    # start_date = '2020-04-01'
    # end_date = '2020-06-01'
    # start_date = '2020-05-01'
    # end_date = '2020-07-01'
    # start_date = '2020-06-01'
    # end_date = '2020-08-01'

    # start_date = '2021-08-20'
    # start_date = '2021-09-20'
    # end_date = '2021-12-20'
    # start_date = '2021-10-20'
    # end_date = '2021-12-20'
    # start_date = '2021-11-20'
    # end_date = '2022-01-20'
    # start_date = '2021-11-17'
    # end_date = '2022-02-17'



    type_list = ['3123', '3124', '3125']
    detection_win = 30
    station_longitudes = [101.036]
    station_latitudes = [38.77]

    # 由基本参数推出的参数，用于后续预测和异常检测
    # 预测得到的数据，是从输入原始数据的第二天开始的。所以predict_start_date是start_date的后一天。
    predict_start_date = selfObj.get_tomorrow(start_date)
    dst_start_date = (datetime.datetime.strptime(predict_start_date, "%Y-%m-%d") +
                      datetime.timedelta(days=detection_win)).strftime("%Y-%m-%d")
    csv_filename = f"{station}_{point_id}_{dst_start_date}_{end_date}"

    # 先创建文件
    if not os.path.exists(f"example-data/{csv_filename}.csv"):
        # 先创建csv文件并写好日期列
        selfObj.create_csv(csv_filename, dst_start_date, end_date)
        # 再添加全为0的3127总场列
        # 创建全为0的list
        zero_list_len = len(selfObj.getEveryDay(dst_start_date, end_date))
        zero_list = [0 for i in range(zero_list_len)]
        # 将全0列添加进文件中
        selfObj.res_save_csv(zero_list, '3127', csv_filename)

    abnormal_num = 0
    for type in type_list:
        # 开始运行模型并绘图
        print(f"---------------{csv_filename}的{type}分量开始预测----------------")
        ori, pre = selfObj.plot_predict(magneticdata_filepath, start_date, end_date, station, point_id, type)
        res = selfObj.plot_detect(ori, pre, predict_start_date, end_date, detection_win, type, f"{station}_{point_id}")
        print(f"----------------{csv_filename}的{type}分量预测结束。--------------")
        # 保存数据
        selfObj.res_save_csv(res, type, csv_filename)
        # 统计res中1的个数，也就是异常值个数/天数
        for i in res:
            if i == 1:
                abnormal_num += 1

    print(f"异常值个数为：{abnormal_num}")
    # 存储结果进文件保存
    with open("example-data/test_log.txt", "a", encoding='utf-8') as f:
        f.write(f"{csv_filename}异常值个数为：{abnormal_num}\n")  # 自带文件关闭功能，不需要再写f.close()
    selfObj.heatmap_example(f"example-data", csv_filename, detection_win, station_longitudes, station_latitudes)

