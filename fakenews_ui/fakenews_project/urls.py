from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('prediccion.urls')),  # Aquí se redirige la raíz a tu app prediccion
]
