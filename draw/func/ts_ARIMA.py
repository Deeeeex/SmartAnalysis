import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

from pmdarima.datasets.stocks import load_msft
from pandas.plotting import lag_plot
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from io import BytesIO

# print(f"Using pmdarima {pm.__version__}")


def draw_ARIMA(file_name, starttime, endtime, sel_prctise, object):
    df = pd.read_csv('draw/static/file_upload/' + file_name, sep=',')
    df.head()
    len_row = len(df)

    train_len = int(df.shape[0] * 0.8)
    train_data, test_data = df[:train_len], df[train_len:]

    y_train = train_data[object].values
    y_test = test_data[object].values

# print(f"{train_len} train samples")
# print(f"{df.shape[0] - train_len} test samples")


    fig, axes = plt.subplots(3, 2, figsize=(10, 16))
    plt.title('MSFT Autocorrelation plot')

# The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)
    ]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        lag_plot(df[object], lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}")
   # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    # 图一：选定关键词随时间的关系图
    imd_1 = "data:image/png;base64," + ims


    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print(f"Estimated differencing term: {n_diffs}")

    auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, error_action="ignore", max_p=6,
                     max_order=None, trace=True)
    print(auto.order)


    model = auto  # seeded from the model we've already fit


    def forecast_one_step():
        fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
        return (
            fc.tolist()[0],
            np.asarray(conf_int).tolist()[0])


    forecasts = []
    confidence_intervals = []

    for new_ob in y_test:
        fc, conf = forecast_one_step()
        forecasts.append(fc)
        confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
        model.update(new_ob)

# print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
# print(f"SMAPE: {smape(y_test, forecasts)}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --------------------- Actual vs. Predicted --------------------------
    axes[0].plot(y_train, color='blue', label='Training Data')
    axes[0].plot(test_data.index, forecasts, color='green', marker='o',
             label='Predicted Y')

    axes[0].plot(test_data.index, y_test, color='red', label='Actual Y')
    axes[0].set_title('Microsoft Prediction')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Y')

    axes[0].set_xticks(np.arange(0, len_row, 12).tolist(), df['Date'][0:len_row:12].tolist())
    axes[0].legend()

# ------------------ Predicted with confidence intervals ----------------
    axes[1].plot(y_train, color='blue', label='Training Data')
    axes[1].plot(test_data.index, forecasts, color='green',label='Predicted')

    axes[1].set_title('Predictions & Confidence Intervals')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(object)

    conf_int = np.asarray(confidence_intervals)
    axes[1].fill_between(test_data.index,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.9, color='orange',
                     label="Confidence Intervals")

    axes[1].set_xticks(np.arange(0, len_row, 12).tolist(), df['Date'][0:len_row:12].tolist())
    axes[1].legend()
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    # 图一：选定关键词随时间的关系图
    imd_2 = "data:image/png;base64," + ims

    info_list = [n_diffs]
    imd_list = [imd_1, imd_2]
    return [info_list, imd_list]
