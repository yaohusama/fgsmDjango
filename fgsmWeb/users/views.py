from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from users import models
from app02 import models as modelsC




def index(request):
    return render(request, 'index.html')
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render,redirect
import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
newfileName=''
def upload_file(request):
    if request.method == "POST":
        newfile=request.FILES.get('newfile',None)

        if not newfile:
            return HttpResponse('提交无效，没有文件上传！')
        to_path=open(os.path.join(BASE_DIR,'uploadfile',newfile.name),'wb+')
        global newfileName
        newfileName=newfile.name
        print(newfile.name)
        for chunk in newfile.chunks():
            to_path.write(chunk)
        to_path.close()
        return HttpResponse('上传成功！')
    else:
        return HttpResponse('非表单提交访问！')
def login1(request):
    return render(request,'upload.html')


from django.http import HttpResponse, Http404
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import shutil
import zipfile

def download(request):
    # 创建zip压缩包
    zip_file = zipfile.ZipFile(
        os.path.join(BASE_DIR, 'download', newfileName), 'w', compression=zipfile.ZIP_DEFLATED#'file/key_zip.zip'
    )

    # 将key文件写入zip压缩包中
    for dir_name in os.listdir('download/'+newfileName.split(".")[0]):
        if not os.path.isfile(os.path.join('download/'+newfileName.split(".")[0], dir_name)):
           # zip_file.write(os.path.join('file/key/', dir_name), dir_name, zipfile.ZIP_STORED)
        # else:
            for file_name in os.listdir(os.path.join('download/'+newfileName.split(".")[0], dir_name)):
                zip_file.write(
                    os.path.join('download/'+newfileName.split(".")[0], os.path.join(dir_name, file_name)),
                    os.path.join(dir_name, file_name),
                    zipfile.ZIP_STORED
                )
    zip_file.close()
    filename =  newfileName
    try:
        download_path = open(os.path.join(BASE_DIR, 'download', filename), 'rb')
        d = HttpResponse(download_path)
        d['content_type'] = 'application/octet-stream'
        d['Content-Disposition'] = 'attachment;filename=' + filename
        return d
    except:
        raise Http404('下载文件' + filename + '失败！')
from datetime import datetime
from .fgsm import mainfgsm
@csrf_exempt
def fgsmU(request):
    data=request.body
    import json
    res=data.decode('utf-8')
    res1=json.loads(res)
    modelsC.UserC.objects.using('db_a').create(name=res1['name'], password=res1['password'],fileName=newfileName,algorithm="fgsm",CreateTime=datetime.now())
  
    imgPath=mainfgsm(newfileName).replace("\\","/")
    d=HttpResponse("/"+imgPath)
    return d
from .jsma import mainjsma
@csrf_exempt
def jsmaU(request):
    data = request.body
    import json
    res = data.decode('utf-8')
    res1 = json.loads(res)
    modelsC.UserC.objects.using('db_a').create(name=res1['name'], password=res1['password'], fileName=newfileName,
                                               algorithm="jsma", CreateTime=datetime.now())
    imgPath=mainjsma(newfileName).replace("\\","/")
    d=HttpResponse("/"+imgPath)
    return d
def login(request):
    if request.method == 'POST':
        name = request.POST.get('username')
        password = request.POST.get('password')
        if name=="root" and password=="123456":
            user_list = models.User.objects.all()
            return render(request, 'manage_user.html', locals())
        user_obj = models.User.objects.using('db_b').filter(name=name, password=password).first()
        contextU={}
        context1={}
        context1["name"]=name
        context1["password"]=password
        contextU["info"]=context1
        if not user_obj:
            return HttpResponse('该用户不存在')
        return render(request, 'upload.html',context1)#HttpResponse('登录成功')
    return render(request, 'login.html')


# 注册视图函数
def register(request):
    if request.method == 'POST':
        # 获取前端提交的用户名
        name = request.POST.get('username')
        # 获取前端提交的用户密码
        password = request.POST.get('password')
        # 查找出相应用户信息，filter返回的也是QuerySet对象，first方法取出其中第一个数据对象，可以为空
        user_obj = models.User.objects.using('db_b').filter(name=name, password=password).first()
        # 用户不存在则在数据库中创建相应用户信息记录，并重定位至管理员界面
        if not user_obj:
            models.User.objects.using('db_b').create(name=name, password=password)
            return redirect('/login/')
        return HttpResponse('该用户已存在')
    return render(request, 'register.html')


# 用户管理界面视图函数
def manage_user(request):
    # 取出所有用户信息，all方法取出来的是QuerySet对象
    user_list = models.User.objects.using('db_b').all()
    # render渲染前端页面，locals将当前函数所有变量都发送至前端
    return render(request, 'manage_user.html', locals())
def manage_userC(request):
    # 取出所有用户信息，all方法取出来的是QuerySet对象
    user_list = modelsC.UserC.objects.using('db_a').all()
    # render渲染前端页面，locals将当前函数所有变量都发送至前端
    return render(request, 'manage_userC.html', locals())

# 删除用户信息视图函数
def delete_user(request):
    user_id = request.GET.get('user_id')
    models.User.objects.using('db_b').filter(id=user_id).delete()
    return redirect('/manage_user/')


# 编辑用户信息视图函数
def edit_user(request):
    user_id = request.GET.get('user_id')
    user_obj = models.User.objects.filter(id=user_id).first()
    if request.method == 'POST':
        name = request.POST.get('username')
        password = request.POST.get('password')
        models.User.objects.using('db_b').filter(id=user_id).update(name=name, password=password)
        return redirect('/manage_user/')
    return render(request, 'edit_user.html', locals())
def delete_userc(request):
    user_id = request.GET.get('user_id')
    modelsC.UserC.objects.using('db_a').filter(id=user_id).delete()
    return redirect('/manage_userC/')


# 编辑用户信息视图函数
def edit_userc(request):
    user_id = request.GET.get('user_id')
    user_obj = models.User.objects.filter(id=user_id).first()
    if request.method == 'POST':
        name = request.POST.get('username')
        password = request.POST.get('password')
        fileName=request.POST.get('fileName')
        algorithm=request.POST.get('algorithm')
        modelsC.UserC.objects.using('db_a').filter(id=user_id).update(name=name, password=password,fileName=fileName,CreateTime=datetime.now(),algorithm=algorithm)
        return redirect('/manage_userC/')
    return render(request, 'edit_userC.html', locals())