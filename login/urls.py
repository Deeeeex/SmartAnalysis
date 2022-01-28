from django.urls import path
from . import views

app_name = "login"
urlpatterns = [
    path('index/', views.index, name='index'),
    path('info/list/', views.info_list),
    path('register/', views.register),
]
