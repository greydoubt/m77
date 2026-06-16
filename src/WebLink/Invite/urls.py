from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
  path('select/', views.selectDateAndTime, name="pageOne"),
  path('info/', views.enterInformation, name="pageTwo"),
  path('confirm/', views.confirmation, name="pageThree"),
  path('save-selected-time/', views.save_selected_time, name='save_selected_time'),
]

  
