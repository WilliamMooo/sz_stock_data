import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors

def try_different_method(model, train_data, train_target, predict_data):
    model.fit(train_data,train_target)
    predict_price = model.predict(predict_data)
    return predict_price

def pca_process(origin_data, proportion):
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

def draw_pic(time, predict_data, real_data):
    # 输出实际值和预测值
    pic_data = pd.DataFrame()
    pic_data['time'] = time
    pic_data['real'] = real_data
    pic_data['predict'] = predict_data
    pic_data.set_index(['time'], inplace=True)
    pic_data.plot()
    plt.show()

def draw_epsilon(time, predict_data, real_data):
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


if __name__ == "__main__":
    origin_data = pd.read_csv("日线数据.csv", encoding="gbk")

    # 时间线处理,去掉第一天，
    time = pd.to_datetime(origin_data['时间'])
    time.drop(index=0, inplace=True)
    time.loc[len(time)+1] = time.loc[len(time)]+pd.Timedelta(days=1)

    # 去掉第一天的收盘价
    close_price = origin_data['收盘']
    close_price.drop(index=0, inplace=True)

    # 筛选数据指标
    data = origin_data[['开盘','收盘','最低','最高','涨跌','涨幅','成交量','成交额/万元']]

    # 主成分分析
    data_p = pca_process(data, 0.95)
    to_predict_data = pd.DataFrame(data_p)
    data_p.drop(index=len(data_p)-1, inplace=True)

    # 线性回归
    # model = linear_model.LinearRegression()
    # predict_data = try_different_method(model, data_p, close_price, to_predict_data)
    # draw_pic(time, predict_data, close_price)
    # draw_epsilon(time, predict_data, close_price)

    # adaboost
    # model = ensemble.AdaBoostRegressor(n_estimators=50) #这里使用50个决策树
    # predict_data = try_different_method(model, data_p, close_price, to_predict_data)
    # draw_pic(time, predict_data, close_price)
    # draw_epsilon(time, predict_data, close_price)

    # knn
    model = neighbors.KNeighborsRegressor()
    predict_data = try_different_method(model, data_p, close_price, to_predict_data)
    draw_pic(time, predict_data, close_price)
    draw_epsilon(time, predict_data, close_price)

