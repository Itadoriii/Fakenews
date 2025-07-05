from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_noticia_view, name='predecir'),  # ← raíz "/"
    path('predict/', views.predict_noticia_view, name='predict_noticia'),
]
