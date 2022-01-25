from django.shortcuts import render, HttpResponse
from login import models
# Create your views here.
user_list = []


def index(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        #数据保存到数据库
        models.UserInfo.objects.create(user=username, pwd=password)

    #数据库读取数据
    user_list = models.UserInfo.objects.all()
    print(user_list)
    return render(request, 'index.html', {'data': user_list})
