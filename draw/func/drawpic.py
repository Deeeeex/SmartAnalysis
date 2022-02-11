import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import base64
from io import BytesIO


def draw_pic():
    x = np.linspace(0, 15, 10)
    y = x * 2
    plt.plot(x, y)
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    return imd
