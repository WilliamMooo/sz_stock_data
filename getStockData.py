import requests
import random
import pandas as pd

class SZ(object):
    # 初始化数据
    def __init__(self):
        self.flag = 0 # 0为分时线；1为日线；2为周线；3为月线
        self.code = '000001' # 股票代码
        self.random = random.uniform(0,1)
        self.server = 'http://www.szse.cn/api/market/ssjjhq/'
        self.timeData = 'getTimeData?random='+str(self.random)+'&marketId=1&code='
        self.dayLine = 'getHistoryData?random='+str(self.random)+'&cycleType=32&marketId=1&code='
        self.weekLine = 'getHistoryData?random='+str(self.random)+'&cycleType=33&marketId=1&code='
        self.monthLine = 'getHistoryData?random='+str(self.random)+'&cycleType=34&marketId=1&code='

    # 获取数据
    def getPicData(self):
        if self.flag == 0:
            # 分时数据
            url = self.server + self.timeData + self.code
            label = ['时间','最新','均价','涨跌','涨幅/%','成交量/手','成交额/万元']
            # df.to_csv('分时数据.csv', index=False, encoding='gbk')
        elif self.flag == 1:
            # 日线数据
            url = self.server + self.dayLine + self.code
            label = ['时间','开盘','最高','最低','收盘','涨跌','涨幅/%','成交量/手','成交额/万元']
            # df.to_csv('日线数据.csv', index=False, encoding='gbk')
        elif self.flag == 2:
            # 周线数据
            url = self.server + self.weekLine + self.code
            label = ['时间','开盘','最高','最低','收盘','涨跌','涨幅/%','成交量/手','成交额/万元']
            # df.to_csv('周线数据.csv', index=False, encoding='gbk')
        elif self.flag == 3:
            # 月线数据
            url = self.server + self.monthLine + self.code
            label = ['时间','开盘','最高','最低','收盘','涨跌','涨幅/%','成交量/手','成交额/万元']
            # df.to_csv('月线数据.csv', index=False, encoding='gbk')
        else:
            print('参数不正确')
        req = requests.get(url=url)
        # print(url)
        originData = req.json()
        data = originData['data']['picupdata']
        df = pd.DataFrame(data, columns=label)
        return df

    # 设置参数
    # flag为获取数据类型,code为股票代码
    def setCondition(self, flag, code):
        self.flag = flag
        self.code = code

if __name__ == "__main__":
    sz = SZ()
    sz.setCondition(0, '200553')
    print(sz.getPicData())