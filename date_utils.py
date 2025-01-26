import datetime



def getEveryDay(begin_date, end_date):
    """
    获取两个日期间的所有日期（包含首尾）
    """
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


def date_format_conversion(ori_date):
    # 把“年-月-日”格式的日期转换成“年月日”
    str_arr = ori_date.split('-')
    return str_arr[0] + str_arr[1] + str_arr[2]


def get_tomorrow(date: str):
    """
    计算给出日期的后一天（输入日期格式为%Y-%m-%d）
    """
    return (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

