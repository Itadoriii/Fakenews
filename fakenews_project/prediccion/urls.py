from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analizar', views.predecir, name='predecir'),  # coincidir con action="/analizar"
]
