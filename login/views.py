from django.shortcuts import render, HttpResponse
from django.http import HttpResponseRedirect
from login import models
# Create your views here.
user_list = []


def index(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        majority = request.POST.get('majority')
        #数据保存到数据库
        models.UserInfo.objects.create(user=username, pwd=password, majr=majority)

    #数据库读取数据
    user_list = models.UserInfo.objects.all()
    print(user_list)
    return render(request, 'index.html', {'data': user_list})

def info_list(request):
    user_list = models.UserInfo.objects.all()
    print(user_list)
    return render(request, 'info_list.html',{'data':user_list})

def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        majority = request.POST.get('majority')
        #数据保存到数据库
        models.UserInfo.objects.create(user=username, pwd=password, majr=majority)
        return HttpResponseRedirect('/info/list/')#重定向至info_list.html

    #数据库读取数据
    user_list = models.UserInfo.objects.all()
    print(user_list)
    return render(request, 'register.html', {'data': user_list})
