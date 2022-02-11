from django.conf import settings
from django.urls import path,re_path
from . import views
from testdj import settings
from django.views.static import serve

app_name = "tasks"
urlpatterns = [
    # Create a task
    path('create/', views.task_create, name='task_create'),

    # Retrieve task list
    path('', views.task_list, name='task_list'),

    # Retrieve single task object
    re_path(r'^(?P<pk>\d+)/$', views.task_detail, name='task_detail'),   #url映射 正则表达式

    # Update a task
    re_path(r'^(?P<pk>\d+)/update/$', views.task_update, name='task_update'),

    # Delete a task
    re_path(r'^(?P<pk>\d+)/delete/$', views.task_delete, name='task_delete'),

    #重写admin页面的控制面板
    path('dashboard/', views.dashboard, name='dashboard'),

    #上传一个文件
    path('upload/', views.upload,name="upload"),
    
    #查看自己上传过的文件（理论上也可以下载）
    re_path(r'media/(?P<path>.*)', serve,{'document_root': settings.MEDIA_ROOT}),
]
