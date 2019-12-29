import requests
import random
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors

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
        # 判断图线类型
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

class Predicter(object):
    def __init__(self):
        self.origin_data = pd.DataFrame()
        self.time = pd.DataFrame()
        self.close_price = pd.DataFrame()
        self.data = pd.DataFrame()
        self.data_p = pd.DataFrame()
        self.to_predict_data = pd.DataFrame()

    def read_data(self, origin):
        self.origin_data = origin
        # 时间线处理,去掉第一天
        self.time = pd.to_datetime(self.origin_data['时间'])
        self.time.drop(index=0, inplace=True)
        self.time.loc[len(self.time)+1] = self.time.loc[len(self.time)]+pd.Timedelta(days=1)
        # 去掉第一天的收盘价
        self.close_price = self.origin_data['收盘']
        self.close_price.drop(index=0, inplace=True)
        # 筛选数据指标
        self.data = self.origin_data[['开盘','收盘','最低','最高','涨跌','涨幅','成交量','成交额/万元']]
        # 主成分分析
        self.data_p = self.pca_process(self.data, 0.95)
        self.to_predict_data = pd.DataFrame(self.data_p)
        self.data_p.drop(index=len(self.data_p)-1, inplace=True)

    def try_different_method(self, model, train_data, train_target, predict_data):
        model.fit(train_data,train_target)
        predict_price = model.predict(predict_data)
        return predict_price

    def pca_process(self, origin_data, proportion):
        if proportion > 1:
            return []
        pca = PCA()
        data = scale(origin_data) # 数据标准化
        data_p = pca.fit(data).transform(data) #pca
        data_r = pd.DataFrame()
        weight = 0
        i = 0
        while weight < proportion:
            weight += pca.explained_variance_ratio_[i]
            data_r[i] = data_p[:,i]
            i += 1
        return data_r

    def draw_pic(self, time, predict_data, real_data):
        # 输出实际值和预测值
        pic_data = pd.DataFrame()
        pic_data['real'] = pd.Series(real_data)
        pic_data['predict'] = pd.Series(predict_data)
        pic_data['time'] = pd.Series(time)
        pic_data.set_index(['time'], inplace=True)
        print(pic_data)
        plt.show()

    def draw_epsilon(self, time, predict_data, real_data):
        pic_data = pd.DataFrame()
        time.drop(index=len(time), inplace=True)
        pic_data['time'] = time
        predict_data = pd.Series(predict_data)
        predict_data.drop(index=len(predict_data)-1, inplace=True)
        epsilon = real_data-predict_data
        epsilon.drop(index=len(epsilon)-1, inplace=True)
        epsilon.drop(index=0, inplace=True)
        pic_data['epsilon'] = epsilon
        pic_data.set_index(['time'], inplace=True)
        pic_data.plot()
        plt.show()

    def Linear_regression(self):
        model = linear_model.LinearRegression()
        predict_data = self.try_different_method(model, self.data_p, self.close_price, self.to_predict_data)
        self.draw_pic(self.time, predict_data, self.close_price)
        self.draw_epsilon(self.time, predict_data, self.close_price)

    def adaboost(self):
        model = ensemble.AdaBoostRegressor(n_estimators=50) #这里使用50个决策树
        predict_data = self.try_different_method(model, self.data_p, self.close_price, self.to_predict_data)
        self.draw_pic(self.time, predict_data, self.close_price)
        self.draw_epsilon(self.time, predict_data, self.close_price)

    def knn(self):
        model = neighbors.KNeighborsRegressor()
        predict_data = self.try_different_method(model, self.data_p, self.close_price, self.to_predict_data)
        self.draw_pic(self.time, predict_data, self.close_price)
        self.draw_epsilon(self.time, predict_data, self.close_price)

root = tk.Tk()
root.title("股票交易市场")
# 接下来是控件
root.geometry('900x500')
# 画布
C = tk.Canvas(root, background='white', width=900, height=500)
C.pack()
# 输入的框
stock_num=tk.StringVar()
stock_num.set('输入您要查询的股票代码')
tk.Entry(root, relief='groove', textvariable=stock_num).place(x=200, y=20)

# 函数回调

# 以下为一些使用实例
# 时间筛选
def A():
    if __name__ == "__main__":
        sz = SZ() # 初始化对象实例
        c = stock_num.get()
        sz.setCondition(0,code=c)
        # if c == "399001":
        #     HQSJ0()
        df = sz.getStockData() # 查询单只股票
        print(df)
        x_data = df["时间"].values
        y_data = df["最新"].values
        x_ticks = []
        y_ticks = []
        for i in range(0,6):
            x_ticks.append(len(df)*i/6)
            y_ticks.append(float(min(y_data))/2*i)
        plt.xticks(x_ticks)
        # plt.yticks(y_ticks)
        plt.plot(x_data, y_data)
        plt.show()
        print(df) # 在控制台中打印
def XX1():
    x=1
    B(x)
def XX2():
    x=2
    B(x)
def XX3():
    x=3
    B(x)
def HQ0():
    stock_num.set('399001')
def HQ1():
    stock_num.set('399106')
def HQ2():
    stock_num.set('399005')
def HQ3():
    stock_num.set('399006')
#def yes():
    #stock_num.set(stock_num.get())

def B(x):
    if __name__ == "__main__":
        sz = SZ() # 初始化对象实例
        c = stock_num.get()
        sz.setCondition(lineType=x,code=c)
        if c == '399001' or c == '399106' or c == '399005' or c == '399006':
            HQSJ(x)
        else:   
            df = sz.getStockData() # 查询单只股票
            calculatreMA(df,5) # 计算MA5线
            calculatreMA(df,10) # 计算MA10线
            calculatreMA(df,20) # 计算MA20线
            calculatreMACD(df) #计算macd线
            df["时间"] = pd.to_datetime(df["时间"])
            df = df[df["时间"] > pd.to_datetime("20180904")]
            KlineData = pd.DataFrame()
            KlineData["MA5"] = df["MA5"]
            KlineData["MA10"] = df["MA10"]
            KlineData["MA20"] = df["MA20"]
            # KlineData["macd"] = df["macd"]
            KlineData["时间"] = df["时间"]
            KlineData.set_index(["时间"], inplace=True)
            KlineData.plot()
            plt.show()
            print(df) # 在控制台中打印
def HQSJ(x):
    if __name__ == "__main__":
        sz2 = SZ() # 初始化对象实例
        hq = stock_num.get()
        mk=0
        if hq == "399106":
            mk=1
        elif hq =="399005":
            mk=2
        elif hq =="399006":
            mk=3
        df = sz2.getMarketData(lineType=x,marketType=mk)
        calculatreMA(df,5) # 计算MA5线
        calculatreMA(df,10) # 计算MA10线
        calculatreMA(df,20) # 计算MA20线
        calculatreMACD(df) #计算macd线
        df["时间"] = pd.to_datetime(df["时间"])
        df = df[df["时间"] > pd.to_datetime("20180904")]
        KlineData = pd.DataFrame()
        KlineData["MA5"] = df["MA5"]
        KlineData["MA10"] = df["MA10"]
        KlineData["MA20"] = df["MA20"]
        # KlineData["macd"] = df["macd"]
        KlineData["时间"] = df["时间"]
        KlineData.set_index(["时间"], inplace=True)
        KlineData.plot()
        plt.show()
        print(df) # 在控制台中打印
# def HQSJ0():
#     if __name__ == "__main__":
#         sz2 = SZ() # 初始化对象实例
#         df = stock_num.get()
#         mk=0
#         if hq == "399001":
#             mk=0
#         x_data = df["时间"].values
#         y_data = df["最新"].values
#         x_ticks = []
#         y_ticks = []
#         for i in range(0,6):
#             x_ticks.append(len(df)*i/6)
#             y_ticks.append(float(min(y_data))/2*i)
#         plt.xticks(x_ticks)
#         # plt.yticks(y_ticks)
#         plt.plot(x_data, y_data)
#         plt.show()
#         print(df) # 在控制台中打印

def xxyc():
    if __name__ == "__main__":
        sz = SZ() # 初始化对象实例
        c = stock_num.get()
        sz.setCondition(1,code=c)
        df = sz.getStockData() # 查询单只股票
        p = Predicter()
        p.read_data(df)
        p.Linear_regression()
def adyc():
    if __name__ == "__main__":
        sz = SZ() # 初始化对象实例
        c = stock_num.get()
        sz.setCondition(0,code=c)
        df = sz.getStockData() # 查询单只股票
        p = Predicter()
        p.read_data(df)
        p.adaboost()
def knnyc():
    if __name__ == "__main__":
        sz = SZ() # 初始化对象实例
        c = stock_num.get()
        sz.setCondition(0,code=c)
        df = sz.getStockData() # 查询单只股票
        p = Predicter()
        p.read_data(df)
        p.knn()
        
        





# 按键
label1 = tk.Button(root, text='分时', width=15, height=2, bg="white", font=("宋体", 15, "bold"),command=A)
label1.place(x=150, y=50)

label2 = tk.Button(root, text='日线', width=15, height=2, bg="white", font=("宋体", 15, "bold"),command=XX1)
label2.place(x=300, y=50)

label3 = tk.Button(root, text='周线', width=15, height=2, bg="white", font=("宋体", 15, "bold"),command=XX2)
label3.place(x=450, y=50)

label4 = tk.Button(root, text='月线', width=15, height=2, bg="white", font=("宋体", 15, "bold"),command=XX3)
label4.place(x=600, y=50)

bn1 = tk.Button(root, text='深证成指', width=15, height=2, bg="white", font=("宋体", 10, "bold"),command=HQ0)
bn1.place(x=0, y=100, anchor='nw')

bn2 = tk.Button(root, text='深证综指', width=15, height=2, bg="white", font=("宋体", 10, "bold"),command=HQ1)
bn2.place(x=0, y=200, anchor='nw')

bn3 = tk.Button(root, text='中小板指', width=15, height=2, bg="white", font=("宋体", 10, "bold"),command=HQ2)
bn3.place(x=0, y=300, anchor='nw')

bn4 = tk.Button(root, text='创业板指', width=15, height=2, bg="white", font=("宋体", 10, "bold"),command=HQ3)
bn4.place(x=0, y=400, anchor='nw')

bn5 = tk.Button(root, text='线性预测', width=15, height=2, fg="white", bg='blue', font=("楷体", 7, "bold"),command=xxyc)
bn5.place(x=350, y=20, anchor='nw')

bn6 = tk.Button(root, text='adaboost预测', width=15, height=2, fg="white", bg='blue', font=("楷体", 7, "bold"),command=adyc)
bn6.place(x=400, y=20, anchor='nw')

bn7 = tk.Button(root, text='knn预测', width=15, height=2, fg="white", bg='blue', font=("楷体", 7, "bold"),command=knnyc)
bn7.place(x=450, y=20, anchor='nw')

root.mainloop()