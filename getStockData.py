import requests
import random
import pandas as pd
import matplotlib.pyplot as plt

def calculatreMA(df, ma):
    df["MA"+str(ma)] = df["收盘"].rolling(ma).mean()

def calculatreMACD(df):
    df["sema"]=df["收盘"].ewm(span=12).mean()
    df["lema"]=df["收盘"].ewm(span=26).mean()
    df["diff"]=df["sema"]-df["lema"]
    df["dea"]=df["diff"].ewm(span=9).mean()
    df["macd"]=2*(df["diff"]-df["dea"])

class SZ(object):
    # 初始化数据
    def __init__(self):
        self.lineType = 0 # 图线类型：0为分时线；1为日线；2为周线；3为月线
        self.code = "000001" # 股票代码
        self.random = str(random.uniform(0,1))
        self.server = "http://www.szse.cn/api/market/ssjjhq/"
        self.timeData = "getTimeData?random="+self.random+"&marketId=1&code="
        self.dayLine = "getHistoryData?random="+self.random+"&cycleType=32&marketId=1&code="
        self.weekLine = "getHistoryData?random="+self.random+"&cycleType=33&marketId=1&code="
        self.monthLine = "getHistoryData?random="+self.random+"&cycleType=34&marketId=1&code="

    # 设置参数
    # lineType为获取数据类型,code为股票代码
    def setCondition(self, lineType, code):
        self.lineType = lineType
        self.code = code

    # 获取市场总貌即板块信息,返回数据类型为dict
    # basicmap为信息类型：0为市场总貌，1为深市主板，2为中小企业板，3为创业板
    def getOverview(self,basicmap):
        url = "http://www.szse.cn/api/report/index/overview/onepersistentday/szse?"+self.random
        req = requests.get(url=url)
        originData = req.json()["result"]["basicmap"]
        if basicmap == 0:
            data = originData["main"]
        elif basicmap == 1:
            data = originData["cdd"]
        elif basicmap == 2:
            data = originData["lcd"]
        elif basicmap == 3:
            data = originData["nmk"]
        ret = {}
        for i in data:
            ret[i["name"]] = i["value"]
        return ret

    # 获取单只股票数据,返回数据类型为dataframe
    def getStockData(self):
        self.random = str(random.uniform(0,1))
        if self.lineType == 0:
            # 分时数据
            url = self.server + self.timeData + self.code
            label = ["时间","最新","均价","涨跌","涨幅","成交量","成交额/万元"]
            # df.to_csv("分时数据.csv", index=False, encoding="gbk")
        elif self.lineType == 1:
            # 日线数据
            url = self.server + self.dayLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额/万元"]
            # df.to_csv("日线数据.csv", index=False, encoding="gbk")
        elif self.lineType == 2:
            # 周线数据
            url = self.server + self.weekLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额/万元"]
            # df.to_csv("周线数据.csv", index=False, encoding="gbk")
        elif self.lineType == 3:
            # 月线数据
            url = self.server + self.monthLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额/万元"]
            # df.to_csv("月线数据.csv", index=False, encoding="gbk")
        else:
            print("参数不正确")
        req = requests.get(url=url)
        originData = req.json()
        data = originData["data"]["picupdata"]
        df = pd.DataFrame(data, columns=label)
        return df

    # 获取行情数据,返回数据类型为dataframe
    # marketType行情数据类型:0为深证成指；1为深证综指；2为中小版指；3为创业版指 
    # lineType图线类型：0为分时线；1为日线；2为周线；3为月线
    def getMarketData(self, marketType, lineType):
        self.random = str(random.uniform(0,1))
        #判断行情类型
        if marketType == 0:
            self.code = "399001"
        elif marketType == 1:
            self.code = "399106"
        elif marketType == 2:
            self.code = "399005"
        elif marketType == 3:
            self.code = "399006"
        #判断图线类型
        self.lineType = lineType
        if lineType == 0:
            url = self.server + self.timeData + self.code
            label = ["时间","指数","涨跌","涨幅","成交量","成交额"]
        elif lineType == 1:
            url = self.server + self.dayLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额"]
        elif lineType == 2:
            url = self.server + self.weekLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额"]
        elif lineType == 3:
            url = self.server + self.monthLine + self.code
            label = ["时间","开盘","收盘","最低","最高","涨跌","涨幅","成交量","成交额"]
        else:
            print("参数错误")
        req = requests.get(url=url)
        originData = req.json()
        data = originData["data"]["picupdata"]
        df = pd.DataFrame(data, columns=label)
        return df

# 以下为一些使用实例
if __name__ == "__main__":
    sz = SZ() # 初始化对象实例
    sz.setCondition(1, "000001") # 设置个股单只股票查询
    df = sz.getStockData() # 查询单只股票
    calculatreMA(df,5) # 计算MA5线
    calculatreMA(df,10) # 计算MA10线
    calculatreMA(df,20) # 计算MA20线
    calculatreMACD(df) #计算macd线
    # 时间筛选
    df["时间"] = pd.to_datetime(df["时间"])
    df = df[df["时间"] > pd.to_datetime("20180904")]
    # 简单画图K线图
    KlineData = pd.DataFrame()
    KlineData["MA5"] = df["MA5"]
    KlineData["MA10"] = df["MA10"]
    KlineData["MA20"] = df["MA20"]
    KlineData["时间"] = df["时间"]
    KlineData.set_index(["时间"], inplace=True)
    KlineData.plot()
    plt.show()
    df.to_csv("日线数据.csv", index=False, encoding="gbk")
    print(df) # 在控制台中打印