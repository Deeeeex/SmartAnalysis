from .func.drawpic import draw_pic
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
        #获取上传的文件
        file_obj = request.FILES.get("up_file")
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
        imd_list = draw_pic(file_obj.name)
        # 返回列名和值的列表
        context = {
            'data_head':header,
            'data_values': rows,
            'img_list': imd_list,
        }
    return render(request, 'draw/file_print.html', context)

