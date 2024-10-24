from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page, name='home'),
    path('audio', views.upload_audio, name='audio'),
    path('form/', views.data_extraction_from_user, name='features'),
    path('audio_result/<str:prediction>/<str:title>', views.audio_result, name='audio_result'),
    path('result/<str:pred>', views.result, name='result')
]
