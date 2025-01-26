# 工程文件说明
## model_interface.py
model为模型类，里面主要为与LSTM-AE模型相关的方法，输入的数据1天应有1440个点  
主要方法为：
1. 模型训练函数：train(data: [], day: int, batch_size: int = 128, epochs: int = 100)  
模型训练函数说明  
:param data: 数据 切片类型  
:param day: 数据天数  
:param batch_size: 可选参数，默认128  
:param epochs: 可选参数，默认100  

2. 数据预测函数：predict(data)调用模型获取预测数据  
预测数据  
:param data:预测数据序列，不少于2天的数据   
:return: 原始数据（去除了第一天），预测数据    
3. 画预测值与观测值的对比图函数：draw_comparison_fig(start_time: str, interval: int, fig_name: str)  
绘制对比图  
:param start_time:起始时间  
:param interval: 横坐标间距，以日为单位  
:param fig_name：保存图片的名称  
   
## detection_interface.py
Detect为地震前兆异常检测类，里面主要为异常检测方法，输入数据可由model类处理后得到    
主要方法为：
1. 异常检测方法：def abnormal_detect(self, ori: [], pre: [], dst_seq: [], type: int = 0):    
异常检测方法，可以得到异常指数序列  
:param ori: 原始数据序列  
:param pre: 预测数据序列  
:param dst_seq: 地磁扰动指数序列，取每天最小值,若只是想获得异常检测结果则传空值，若想获得去除磁暴后的结果则必须输入  
:param type: 输入数据类型 0：非磁偏角，1：磁偏角  
:return: 处理后的时间序列，1个点为 1天,若去除了磁暴相当于计算了异常指数，序列为 0 或 1  
2. 绘制异常检测直方图：draw_abnormal_detection_fig(self, start_time: str, interval: int, fig_name: str)  
画异常检测的直方图  
:param start_time: 横坐标开始时间  
:param interval: 横坐标间隔  
:param fig_name: 图片名  
3. 计算网格边界：coordinate_calculator(lon_start, lon_end, lat_start, lat_end)  
计算网格矩阵边界    
:param lon_start: 开始经度  
:param lon_end: 结束经度  
:param lat_start: 开始纬度  
:param lat_end: 结束纬度  
:return: 横坐标点切片，纵坐标点切片  
4. 绘制区域异常指数热图：draw_heatmap(self, matrixs: [[]], longitudes: [], latitudes: [], fig_name: str, max_value: int = 0)    
绘制累计异常热图  
:param matrixs: 区域台站的累计异常矩阵数组，3维  
:param longitudes: 台站经度数组  
:param latitudes: 台站纬度数组  
:param fig_name: 图片名  
:param max_value: 可选参数，异常上限，默认每个台站均为4分量  

## dst_interface.py
处理磁暴数据的类，数据可从日本京都地磁中心获取，可根据自己需求进行修改

## example.py、run_example文件夹、article_need2.py
example.py是示例文件，article_need等几个文件是论文实验代码。

## Model
模型文件存放的位置

## Fig
图片存放位置


## example-data 
测例数据及部分计算结果的存放位置，

# 环境配置
tensorflow 2.6(根据自身显卡修改tensorflow配置)
keras 2.6
python 3.7  
numpy  
pandas  
matplotlib
    
# 开发者注意事项  
若之后程序有修改，请写上自己的大名，并在修改处注明时间，方便后人查看  
**write by 徐邦杰 2023年6月**


