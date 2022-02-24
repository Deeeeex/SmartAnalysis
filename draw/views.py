from .func.ts_ARIMA import draw_ARIMA
from .func.ts_ANN_LSTM import draw_ANN_LSTM
from django.shortcuts import render
from django.conf import settings
import os
import pandas as pd
import numpy as np

def homepage(request):
    context = {

    }
    return render(request, 'draw/home.html', context)

def filetransfer(request):
    return render(request, "draw/file_transfer.html", locals())


def file_print(request):
    if request.method == "POST":
        #获取上传的文件及其相关信息
        file_obj = request.FILES.get("up_file")
        starttime = request.POST.get("starttime", None)
        endtime = request.POST.get("endtime", None)
        sel_prctise = request.POST.get("sel_prctise", None)
        object = request.POST.get("object", None)
        model = request.POST.get("sel_model", None)
        #获取当前项目的路径
        base = str(settings.BASE_DIR)
        # 文件本地存储地址
        path = os.path.join(base, 'draw/static/file_upload', file_obj.name)
        # 将文件写入本地静态文件夹
        with open(path, 'wb') as f1:
            for i in file_obj.chunks():
                f1.write(i)
        #待改进：支持读入csv,xls,xlsx等表格文件或者导入数据库
        # 读取上传的文件(内存中)
        df = pd.read_csv('draw/static/file_upload/'+file_obj.name)
        # 获取列名和值
        header = df.columns.to_list()
        dfa=np.array(df)
        rows=dfa.tolist()
        # df1_head = df.columns.values.tolist()
        # df1_values = df.values.tolist()
        if model=='ARIMA':
            imd_list = draw_ARIMA(file_obj.name, starttime, endtime, sel_prctise, object)
        elif model=='ANN_LSTM':
            ret_list = draw_ANN_LSTM(file_obj.name, starttime, endtime, sel_prctise, object)
            imd_list=ret_list[1]
            info_list=ret_list[0]
        # 返回列名和值的列表
        context = {
            'data_head':header,
            'data_values': rows,
            'info_list':info_list,
            'img_list':imd_list,
        }
    return render(request, 'draw/file_print.html', context)

