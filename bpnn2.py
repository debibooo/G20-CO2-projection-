import shap as shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from shap.plots._beeswarm import summary_legacy
from sklearn.inspection import partial_dependence  # 直接获取pdp数组的方法
from scipy.interpolate import splev, splrep  # 数据平滑插值
# Back-Propagation Neural Networks
import pandas as pd
import math
import random

import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
inpath = 'C:\\Users\\PC\\OneDrive - 中山大学\\文档\\ABC-SVR\\data'
ems = pd.read_excel(inpath + '\\ems_g20.xlsx', sheet_name=0)
emfactor = pd.read_excel(inpath + '\\emfactor_g20.xlsx', sheet_name=0, header=0)
countries = np.asarray(ems.loc[:, 'Country Name'])
future_x = pd.read_excel(inpath + '\\future_g20_SSP4_45.xlsx', sheet_name='BAU_wi_NDC', header=0)
r2_test_list = []
r2_train_list=[]
rmse_test_list = []
rmse_train_list = []
mae_test_list = []
mae_train_list = []
mape_test_list = []
mape_train_list = []
test_y_pre_list = np.zeros(shape=(44, 10))
train_y_list = np.zeros(shape=(44, 40))
train_y_pre_list = np.zeros(shape=(44, 40))
test_y_list = np.zeros(shape=(44, 10))
future = np.zeros(shape=(44, 11))


#train_yy_pre_list = np.zeros(shape=(44, 40))
#test_y_pre_list = np.zeros(shape=(44, 10))

#train_yy_list = np.zeros(shape=(44, 40))
#test_y_list = np.zeros(shape=(44, 10))
#future = np.zeros(shape=(44, 11))
pat1=[]
pat2=[]
fut=[]
index = 0
# ----------------------main code---------------------------------------------------------------------------------------

for country in list(countries):


    x = emfactor[emfactor['Country Name'] == country]
    y = ems[ems['Country Name'] == country]
    x_future = future_x[future_x['Country Name'] == country]
    data = x.drop(['Series Name', 'Series Code', 'Country Name', 'Country Code'], axis=1)
    fut_x = x_future.drop(['Series Name', 'Series Code', 'Country Name', 'Country Code'], axis=1)
    data = data.replace(0, np.nan)
    k = 7

    ##-------删掉缺失了60%的因子---------
    for i in list(data.index.values):
        x1 = data.loc[i, :]
        num = x1.isna().sum(axis=0)
        if num > 30:
            data = data.drop(index=i, axis=0)
            fut_x = fut_x.drop(index=i, axis=0)
            k = k - 1
    fut_xx = np.asarray(fut_x.T)
    feature = np.asarray(x.loc[data.index, 'Series Name'])  # feature名称
    feature_code = np.asarray(x.loc[data.index, 'Series Code'])
    colu=np.concatenate((['year','co2'],feature_code),axis=0)
    print('country:', country, 'feature:', feature_code, feature.shape,'colu',colu)
    print('x:', data, 'x_future:', fut_x)


    ###------------随机森林补全NAN值----------------------------
    def NAN_completion(data):
        data2 = np.asarray(data)
        x_missing_reg = pd.DataFrame(data2.copy().T)
        y_full1 = np.asarray(y.drop(['Country Name', 'Country Code'], axis=1))
        y_full = pd.DataFrame(y_full1.copy().T)
        sortindex = np.argsort(x_missing_reg.isnull().sum(axis=0).values)
        print(x_missing_reg.isnull().sum(axis=0), sortindex)
        for i in sortindex:
            # 构建我们的新特征矩阵（没有被选中去填充的特征+原始的标签）和新标签（被选中去填充的特征）
            df = x_missing_reg
            fillc = df.iloc[:, i]
            df = pd.concat([df.iloc[:, df.columns != i], y_full], axis=1)
            m = fillc.isnull().sum()
            if m == 0:
                continue
            else:
                # 在新特征矩阵中，对含有缺失值的列，进行0的填补
                df_0 = SimpleImputer(missing_values=np.nan
                                     , strategy="constant"
                                     , fill_value=0
                                     ).fit_transform(df)

                y_train = fillc[fillc.notnull()]
                y_test = fillc[fillc.isnull()]
                x_train = df_0[y_train.index, :]
                x_test = df_0[y_test.index, :]

                # 用随机森林回归来填补缺失值
                rfc = RandomForestRegressor(n_estimators=100,random_state=101)
                rfc.fit(x_train, y_train)
                Ypredict = rfc.predict(x_test)
                x_missing_reg.loc[x_missing_reg.iloc[:, i].isnull(), i] = Ypredict
                print('num:', x_missing_reg.isnull().sum())
        data_x = np.asarray(x_missing_reg)
        data_y = np.asarray(y_full)
        year=pd.DataFrame(np.arange(1970,2020),dtype=int)
        df2 = pd.concat([pd.DataFrame(feature), x_missing_reg.T], axis=1)
        dataset=pd.concat([year,y_full,x_missing_reg],axis=1)
        dataset.columns = colu

        return data_x, data_y, df2, dataset


    data_x, data_y, df2 ,dataset = NAN_completion(data)
    print('df2:', df2,'dataset:',dataset)
    scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler1.fit(data_x)
    scaler2.fit(data_y)
    data_xx = scaler1.transform(data_x)
    data_yy = scaler2.transform(data_y)
   # fut_xxx = scaler1.transform(fut_xx)
    X_train, X_test, Y_train, Y_test = train_test_split(data_xx, data_yy, test_size=0.2,
                                                        random_state=101)  # 101-110十次随机实验
    train_xx= pd.DataFrame(X_train, columns=feature)

    print('train_xx',train_xx)

    random.seed(0)

    # calculate a random number where:  a <= rand < b
    def rand(a, b):
     return (b-a)*random.random() + a

    # our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
    def sigmoid(x):
     return math.tanh(x)

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(y):
     return 1.0 - y**2

    class Unit:
      def __init__(self, length):

        self.weight = [rand(-0.2, 0.2) for i in range(length)]
        self.change = [0.0] * length
        self.threshold = rand(-0.2, 0.2)
        #self.change_threshold = 0.0
      def calc(self, sample):
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold
        self.output = sigmoid(partsum)
        return self.output
      def update1(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
        #self.threshold = rateN * factor + rateM * self.change_threshold + self.threshold
        #self.change_threshold = factor
      def get_weight(self):
        return self.weight[:]
      def set_weight(self, weight):
        self.weight = weight[:]


    class Layer:
      def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]
        self.output = [0.0] * output_length
        self.ilen = input_length
      def calc(self, sample):
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
      def update2(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update1(diff, rate, factor)
      def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.ilen)]
      def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights
      def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])



    class BPNNet:

      def __init__(self, ni, nh, no):

        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no
        self.hlayer = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)

      def calc(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs[:] + [1.0]

        # hidden activations
        self.ah = self.hlayer.calc(self.ai)
        # output activations
        self.ao = self.olayer.calc(self.ah)


        return self.ao[:]


      def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]

        # calculate error terms for hidden
        hidden_deltas = [dsigmoid(ah) * error for ah, error in zip(self.ah, self.olayer.get_error(output_deltas))]

        # update output weights
        self.olayer.update2(output_deltas, rate, factor)

        # update input weights
        self.hlayer.update2(hidden_deltas, rate, factor)
        # calculate error
        return sum([0.5*(t-o)**2 for t, o in zip(targets, self.ao)])   #返回的误差和mse有什么关系？？BPNN主要使用梯度下降的原理,

      def train(self, xx, yy, iterations=500, N=0.1, M=0.1):
        train_x=np.asarray(xx)
        train_y=np.asarray(yy)
        pat1 = []
        for i in range(0, 40):
              train_x_p = train_x[i, :].tolist()
              train_y_p = train_y[i].tolist()
              pat1_p = [train_x_p, train_y_p]
              pat1.append(pat1_p)

        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p1 in pat1:

                inputs = p1[0]
                targets = p1[1]
                self.calc(inputs)
                error = error + self.update(targets, N, M)
                # 将train所有的error累加
            if i % 100 == 0:
                print('error %-.10f' % error)
            self.save_weights('tmp.weights')
      def save_weights(self, fn):
        weights = {
                "olayer":self.olayer.get_weights(),
                "hlayer":self.hlayer.get_weights()
                }
        with open(fn, "wb") as f:
            pickle.dump(weights, f)
      def load_weights(self, fn):
            with open(fn, "rb") as f:
                weights = pickle.load(f)
                self.olayer.set_weights(weights["olayer"])
                self.hlayer.set_weights(weights["hlayer"])

      def test(self, patterns):

        test_y_pre=[]
        for p in patterns:
            print(p[0], '->', self.calc(p[0]))
            test_y_pre.append(self.calc(p[0]))

        return test_y_pre
      def predict(self, patterns):
        future_y=[]
        for ps in patterns:
            future_y.append(self.calc(ps))
        future_y = np.asarray(future_y)
        return future_y

    def demo():

      M=data_x.shape[1]
      N=round((M+1)/2)
      bpnn = BPNNet(M, N, 1)
    # train it with some patterns
      bpnn.train(X_train,Y_train)

     # Generate colormap through matplotlib
#      newCmap = LinearSegmentedColormap.from_list("", ['#515151', '#F14040'])
    # Create object that can calculate shap values
#      explainer = shap.KernelExplainer(bpnn.predict,train_xx)
    # Calculate Shap values
#      shap_values = explainer.shap_values(train_xx)
#      shap.summary_plot(shap_values[0],train_xx,
#                        feature_names=feature_code,
#                        show=False, plot_type='dot',cmap=newCmap)
#      plt.rcParams['font.sans-serif'] = 'Arial'
#      q=k*40
#      value=np.asarray(shap_values[0]).reshape(1,q)
#      max_value=np.max(value)
#      print('max',max_value)
#      plt.xlim([-1.2*max_value,1.2*max_value])
#      plt.title(str(country),fontsize=20)
#      plt.savefig(inpath + '\emission plot\\' + str(country) + '.jpg', dpi=1200, bbox_inches='tight',
#                  pad_inches=0.1)
#      plt.close()


    # test it

      test_y_pred=np.asarray(bpnn.test(X_test))
      train_yy_pred=np.asarray(bpnn.test(X_train))


      print('train_y_pred', np.asarray(train_yy_pred).shape)
      print('test_y_pred', np.asarray(test_y_pred).shape)
     # predict it
      future_y=np.asarray(bpnn.predict(fut))
     # r2 score and rmse

      future_y2 = scaler2.inverse_transform(future_y.reshape(1, -1))
      test_y_pre2 = scaler2.inverse_transform(test_y_pred.reshape(1, -1))
      test_y2 = scaler2.inverse_transform(Y_test.reshape(1, -1))
      train_yy2=scaler2.inverse_transform(Y_train.reshape(1,-1))
      train_yy_pre2=scaler2.inverse_transform(train_yy_pred.reshape(1,-1))
      r2_test = round(1 - mean_squared_error(test_y2, test_y_pre2) / np.var(test_y2), 3)
      r2_train = round(1 - mean_squared_error(train_yy2, train_yy_pre2) / np.var(train_yy2), 3)
      rmse_test = round(mean_squared_error(test_y2, test_y_pre2)** 0.5, 3)/1000000
      rmse_train = round(mean_squared_error(train_yy2, train_yy_pre2)** 0.5, 3)/1000000
      mae_test = np.mean(np.abs(test_y2 - test_y_pre2)).round(3)/1000000
      mae_train = np.mean(np.abs(train_yy2 - train_yy_pre2)).round(3)/1000000
      mape_test = np.mean(np.abs((test_y2 - test_y_pre2) / test_y2)) * 100
      mape_train = np.mean(np.abs((train_yy2 - train_yy_pre2) / train_yy2)) * 100
      test_y_list[index, :] = np.asarray(test_y2)
      test_y_pre_list[index, :] = np.asarray(test_y_pre2)
      future[index, :] = np.asarray(future_y2)
      train_y_list[index,:]=np.asarray(train_yy2)
      train_y_pre_list[index,:]=np.asarray(train_yy_pre2)
      r2_test_list.append(r2_test)
      r2_train_list.append(r2_train)
      rmse_test_list.append(rmse_test)
      rmse_train_list.append(rmse_train)
      mae_test_list.append(mae_test)
      mae_train_list.append(mae_train)


      mape_test_list.append(mape_test)
      mape_train_list.append(mape_train)

    index = index + 1
    if __name__ == '__main__':
      demo()


result = pd.DataFrame({'ccountry': countries, 'R2': np.asarray(r2_test_list), 'R2_train':np.asarray(r2_train_list),
                       'RMSE_test': np.asarray(rmse_test_list),'RMSE_train': np.asarray(rmse_train_list),
                       'MAE_test': np.asarray(mae_test_list),'MAE_train': np.asarray(mae_train_list),
                       'MAPE_test': np.asarray(mape_test_list),'MAPE_train': np.asarray(mape_train_list)})
test_y_list=pd.DataFrame(test_y_list)
#future = pd.DataFrame(future)
#test_y_list=pd.DataFrame(test_y_list)
#test_y_pre_list=pd.DataFrame(test_y_pre_list)
#train_yy_pre_list=pd.DataFrame(train_yy_pre_list)
#train_yy_list=pd.DataFrame(train_yy_list)
writer = pd.ExcelWriter(r'C:\\Users\PC\OneDrive - 中山大学\文档\ABC-SVR\data\result\BPNN\1x.xlsx')
result.to_excel(writer, 'Rscore')
#future.to_excel(writer, 'future_y')
#test_y_list.to_excel(writer, 'test_y')
#test_y_pre_list.to_excel(writer, 'test_y_pre')
#train_yy_list.to_excel(writer,'train_y')
#train_yy_pre_list.to_excel(writer,'train_y_pre')
writer.save()
writer.close()