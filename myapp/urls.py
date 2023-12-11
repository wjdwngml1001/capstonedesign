from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('result/<int:image_id>/', views.result, name='result'),
]