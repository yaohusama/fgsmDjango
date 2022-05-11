from django.db import models
from django.contrib.auth.models import AbstractUser


from django.db import models

# Create your models here.


class User(models.Model):
    name = models.CharField(max_length=32,default=u"root")
    password = models.CharField(max_length=32,default=u"123456")

    class Meta:
        db_table = 'userT'
        app_label = 'users'