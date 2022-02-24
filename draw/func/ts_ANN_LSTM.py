import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import adam_v2
from keras.layers import LSTM
import base64
from io import BytesIO


def draw_ANN_LSTM(file_name, starttime, endtime, sel_prctise, object):
    df = pd.read_csv('draw/static/file_upload/' + file_name, sep=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date'], drop=True)
    #print(df.head(10))
    plt.figure(figsize=(10, 6))
    df = df[object]  # 样例使用'Adj Close'
    df = df.loc[starttime:endtime]
    df.plot()
    #plt.show()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    #图一：选定关键词随时间的关系图
    imd_1 = "data:image/png;base64," + ims 

    # split_date = pd.Timestamp('2018-01-01')
    # train = df.loc[:split_date]
    # test = df.loc[split_date:]
    #将数据集划分成训练集与测试集
    train_size = int(float(sel_prctise)*df.shape[0])
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    plt.figure(figsize=(10, 6))
    ax = train.plot()
    test.plot(ax=ax)
    plt.legend(['train', 'test'])
    #plt.show()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    #图二：在图一的基础上将原来的图划分成训练集和测试集
    imd_2 = "data:image/png;base64," + ims 

    #对数据进行归一化处理,将数据值缩放到[-1,1]范围内
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train.values.reshape(-1, 1))
    test_sc = scaler.transform(test.values.reshape(-1, 1))
    #通过数据平移的方式提取数据标签
    predict_days = 1
    X_train = train_sc[:-predict_days]
    y_train = train_sc[predict_days:]
    X_test = test_sc[:-predict_days]
    y_test = test_sc[predict_days:]

    nn_model = Sequential()


    nn_model.add(Dense(12, input_dim=1, activation='relu'))
    nn_model.add(Dense(1))
    #print(nn_model.summary())
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1,
                        verbose=1, callbacks=[early_stop], shuffle=False)
    y_pred_test_nn = nn_model.predict(X_test)
    y_train_pred_nn = nn_model.predict(X_train)
    # print("The R2 score on the Train set is:\t{:0.3f}".format(
    #     r2_score(y_train, y_train_pred_nn)))
    # print("The R2 score on the Test set is:\t{:0.3f}".format(
    #     r2_score(y_test, y_pred_test_nn)))
    train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
    test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

    for s in range(1, 2):
        train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
        test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

    X_train = train_sc_df.dropna().drop('Y', axis=1)
    y_train = train_sc_df.dropna().drop('X_1', axis=1)

    X_test = test_sc_df.dropna().drop('Y', axis=1)
    y_test = test_sc_df.dropna().drop('X_1', axis=1)

    X_train = X_train.values
    y_train = y_train.values

    X_test = X_test.values
    y_test = y_test.values

    X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # print('Train shape: ', X_train_lmse.shape)
    # print('Test shape: ', X_test_lmse.shape)

    lstm_model = Sequential()
    lstm_model.add(LSTM(7, input_shape=(
        1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history_lstm_model = lstm_model.fit(X_train_lmse, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    y_pred_test_lstm = lstm_model.predict(X_test_lmse)
    y_train_pred_lstm = lstm_model.predict(X_train_lmse)
    # print("The R2 score on the Train set is:\t{:0.3f}".format(
    #     r2_score(y_train, y_train_pred_lstm)))
    # print("The R2 score on the Test set is:\t{:0.3f}".format(
    #     r2_score(y_test, y_pred_test_lstm)))
    #模型比较
    nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
    lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)
    # print('NN MSE: %f' % nn_test_mse)
    # print('LSTM MSE: %f' % lstm_test_mse)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_nn, label='NN')
    plt.title("ANN's Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Adj Close Scaled')
    plt.legend()
    #plt.show()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    #图三：ANN模型预测效果
    imd_3 = "data:image/png;base64," + ims
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test_lstm, label='LSTM')
    plt.title("LSTM's Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Adj Close scaled')
    plt.legend()
    #plt.show()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    #图四：LSTM模型预测效果
    imd_4 = "data:image/png;base64," + ims

    info_list = [r2_score(y_train, y_train_pred_nn), r2_score(y_test, y_pred_test_nn), r2_score(
        y_train, y_train_pred_lstm), r2_score(y_test, y_pred_test_lstm), nn_test_mse, lstm_test_mse]
    imd_list = [imd_1, imd_2, imd_3, imd_4]
    return [info_list,imd_list]
