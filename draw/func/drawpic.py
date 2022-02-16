import statsmodels.api as smapi
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
import statsmodels.api as sm
#from .ts_ARIMA import imd_2
import itertools
import pandas as pd
import matplotlib.pylab as plt
plt.switch_backend('agg')
import numpy as np
import base64
from io import BytesIO
import os
os.getcwd()
# TSA from Statsmodels
#from statsmodels.tsa.statespace.sarimax import SARIMAX


def draw_pic():
    #决定画布内容
    #x = np.linspace(0, 15, 10)
    #y = x * 2
    #plt.plot(x, y)
    #CSV_FILE_PATH='draw/static/file_upload/test_1.xlsx'
    #milkproduction = pd.read_excel(CSV_FILE_PATH, index_col=0)
    milkproduction = pd.read_csv('draw/static/file_upload/milkproduction.csv', sep=',', index_col=0)
    milkproduction.plot(figsize=(12, 8))
    #plt.legend(bbox_to_anchor=(1.25, 0.5))
    plt.title("Monthly Milk Production ")
    #plt.show()
    #下面是plt的编码过程
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd_1 = "data:image/png;base64," + ims
    
    ts = pd.Series(np.array(milkproduction['production'].astype('float64')),
                   index=pd.period_range('196201', '197512', freq='M'))


    ts.head()


    def test_stationarity(timeseries):
        #滚动平均
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).mean()
        ts_diff = timeseries - timeseries.shift()

        orig = timeseries.plot(color='blue', label='Original')
        mean = rolmean.plot(color='red', label='Rolling Mean')
        std = rolstd.plot(color='black', label='Rolling Std')
        diff = ts_diff.plot(color='green', label='Diff 1')

        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        #plt.show(block=False)
        buffer = BytesIO()
        plt.savefig(buffer)
        plot_data = buffer.getvalue()
        imb = base64.b64encode(plot_data)  # 对plot_data进行编码
        ims = imb.decode()
        imd = "data:image/png;base64," + ims
        #adf检验
        #print('Result of Dickry-Fuller test')
        dftest = adfuller(timeseries, autolag='AIC')
        #dfoutput = pd.Series(dftest[0:4], index=[
        #'Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
        #for key, value in dftest[4].items():
        #dfoutput['Critical value(%s)' % key] = value
        #print(dfoutput)
        return imd

    
    imd_2 = test_stationarity(ts)

    imd_list=[imd_1,imd_2]
    return imd_list


    
