from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('home', views.homepage, name='homepage'),
    path('filetransfer', views.filetransfer, name='filetransfer'),
    path('fileprint', views.file_print, name='fileprint'),

]
