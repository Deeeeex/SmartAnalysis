from django.db import models

# Create your models here.


class UserInfo(models.Model):
    user = models.CharField(max_length=32,null=True)
    pwd = models.CharField(max_length=32,null=True)
    majr = models.CharField(max_length=32,null=True)
    
