import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors

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
        # 时间线处理,去掉第一天，
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
        pic_data['time'] = time
        pic_data['real'] = real_data
        pic_data['predict'] = predict_data
        pic_data.set_index(['time'], inplace=True)
        print(pic_data)
        pic_data.plot()
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


if __name__ == "__main__":
    # 从文件读取原始数据
    origin_data = pd.read_csv("日线数据.csv", encoding="gbk")

    p = Predicter()

    # 传入数据
    p.read_data(origin_data)

    # 线性回归
    p.Linear_regression()

    # adaboost
    p.adaboost()

    # knn
    p.knn()