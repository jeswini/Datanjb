from django.urls import include, path
from ActionApp import views

urlpatterns=[
    path('action/',views.actionApi)
]