from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import AbstractUser


from django.db import models

# Create your models here.


class UserC(models.Model):
    name = models.CharField(max_length=32,default=u"root")
    password = models.CharField(max_length=32,default=u"123456")
    fileName=models.CharField(max_length=32,default=u"data")
    CreateTime = models.DateField(help_text='操作记录时间', blank=True, null=True)
    algorithm = models.CharField(max_length=32,default=u"fgsm")
    def __str__(self):
        return self.name
    class Meta:
        db_table = 'recordT'
        app_label = 'app02'