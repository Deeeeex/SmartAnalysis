from django.contrib import admin
from .models import UserInfo
from tasks.models import Task


admin.site.site_header = 'TimeSeries管理后台'  # 设置header
admin.site.site_title = 'TimeSeries管理后台'   # 设置title
admin.site.index_title = 'TimeSeries管理后台'
# Register your models here.
admin.site.register(UserInfo)
admin.site.register(Task)

