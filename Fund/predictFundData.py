import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

class Predicter(object):
    def __init__(self):
        # 从文件中读取原始数据
        self.fund_code = "000001"
        self.origin_data = pd.read_csv(self.fund_code+"日线数据.csv", encoding="gbk")
        # 绘图数据
        self.pic_data = pd.DataFrame()
        self.pic_data['time'] = self.origin_data['时间']
        self.pic_data['real'] = self.origin_data['收盘']
        # 切分训练集和验证集
        self.origin_data.drop(['时间'], axis=1, inplace=True)
        self.data_process = self.pca_process(self.origin_data.values, 0.95)
        self.train_data = np.array([])
        self.valid_data = np.array([])
        self.train_data = self.data_process[:128]
        self.valid_data = self.data_process[128:]
        self.train_target = self.origin_data['收盘'].values[1:129]
        
    def pca_process(self, data, proportion):
        if proportion > 1:
            return []
        pca = PCA()
        data = scale(data) # 数据标准化
        data = pca.fit_transform(data)
        weight = 0
        i = 0
        while weight < proportion:
            weight += pca.explained_variance_ratio_[i]
            i += 1
        # print(pca.explained_variance_ratio_) # 主成分占比
        # print(pca.explained_variance_) # 特征根
        # print(np.linalg.eig(pca.get_covariance())) # 协方差矩阵
        return data[:,0:i]
    
    def OLS(self):
        #  普通最小二乘
        model = linear_model.LinearRegression()
        model.fit(self.train_data, self.train_target)
        # self.fit_data = model.predict(self.train_data)
        self.predict_data = model.predict(self.valid_data)

    def Ridge(self):
        # 岭回归
        model = linear_model.Ridge(alpha=40)
        # model = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0, 20.0, 40.0, 50.0])
        model.fit(self.train_data, self.train_target)
        # self.fit_data = model.predict(self.train_data)
        self.predict_data = model.predict(self.valid_data)
        # self.showModelAlpha(model)

    def LASSO(self):
        # LASSO回归
        model = linear_model.Lasso(alpha=0.01)
        # model = linear_model.LassoCV(alphas=[0.001, 0.005, 0.01, 0.1, 1.0, 10.0, 50.0])
        model.fit(self.train_data, self.train_target)
        self.fit_data = model.predict(self.train_data)
        self.predict_data = model.predict(self.valid_data)
        # self.showModelAlpha(model)

    def ElasticNet(self):
        # 弹性网络
        model = ElasticNet(alpha=0.03)
        # model = ElasticNetCV(alphas=[0.001, 0.005, 0.03, 0.1, 1.0, 10.0, 50.0])
        model.fit(self.train_data, self.train_target)
        # self.fit_data = model.predict(self.train_data)
        self.predict_data = model.predict(self.valid_data)
        # self.showModelAlpha(model)

    def showModelAlpha(self, model):
        print(model.alpha_)

    def fit_predictPic(self):
        # 预测和拟合曲线
        # self.pic_data.drop(index=0, inplace=True)
        # self.predict_data = self.predict_data[:-1]
        # self.pic_data.drop(index=len(self.pic_data)-1, inplace=True)
        pic_data = self.pic_data
        pic_data.set_index(['time'],inplace=True)
        fit_predict = np.concatenate((self.fit_data, self.predict_data))
        pic_data['fit_predict'] = fit_predict
        pic_data.plot()
        plt.show()
        return pic_data

    def predictPic(self):
        # 预测曲线
        pic_data = self.pic_data
        # self.pic_data.drop(index=len(self.pic_data)-1, inplace=True)
        for i in range(len(self.train_data)):
            pic_data.drop(index=i, inplace=True)
        # self.predict_data = self.predict_data[:-1]
        print(len(self.predict_data), len(pic_data))
        pic_data.set_index(['time'],inplace=True)
        pic_data['predict'] = self.predict_data
        pic_data.plot()
        plt.show()
        return pic_data


if __name__ == "__main__":
    p = Predicter()
    # p.OLS()
    # p.Ridge()
    # p.LASSO()
    p.ElasticNet()
    
    # p.fit_predictPic()
    # p.predictPic()

    # 误差分析
    data = p.predictPic()
    print(data)
    y_true = data['real'].values
    y_pred = data['predict'].values
    # 解释方差分
    EVS = explained_variance_score(y_true, y_pred)
    print("解释方差分:", EVS)
    # 平均绝对误差
    MAE = mean_absolute_error(y_true, y_pred)
    print("平均绝对误差:", MAE)
    # 均方误差
    MSE = mean_squared_error(y_true, y_pred)
    print("均方误差:", MSE)
    # 均方对数误差
    MSLE = mean_squared_log_error(y_true, y_pred)
    print("均方对数误差:", MSLE)
    # 中位数绝对误差
    MAE_ = median_absolute_error(y_true, y_pred)
    print("中位数绝对误差:", MAE_)
    # 决定系数、R方
    r2 = r2_score(y_true, y_pred)
    print("决定系数:", r2)
