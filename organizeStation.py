# -*- coding:utf-8 -*-
import xlrd
data = xlrd.open_workbook('成分信息.xls') # 打开xls文件
table = data.sheets()[0]
nrows = table.nrows      # 获取表的行数

for i in range(nrows):   # 循环逐行打印
    # 跳过第一行
    if i == 0:
        continue
