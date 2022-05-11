"""django_auth_example URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin

from users import views
urlpatterns = [
    url('system/', views.login1,name="system"),
    url('^uploadfile/', views.upload_file),
    url('download/',views.download,name='download'),
    url('fgsm/',views.fgsmU,name='fgsm'),
    url('jsma/',views.jsmaU,name='jsma'),
    # url('uploadxtt/',views.project_upload,name='download'),
    url(r'^file/', views.login1),
    url(r'^admin/', admin.site.urls),
    url(r'^login/', views.login,name="login"),
    url(r'^register/', views.register),
    url(r'^manage_user/', views.manage_user),
    url(r'^manage_userC/', views.manage_userC),
    url(r'^delete_user/', views.delete_user),
    url(r'^edit_user/', views.edit_user),
    url(r'^delete_userC/', views.delete_userc),
    url(r'^edit_userC/', views.edit_userc),
]


