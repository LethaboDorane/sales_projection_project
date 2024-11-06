# sales_projection_project/urls.py
from django.urls import path
from projection_app import views

urlpatterns = [
    path('', views.sales_projection_view, name='sales_projection'),
]

urlpatterns = [
    path('', views.sales_projection_view, name='sales_projection'),
    path('download-image/', views.download_image, name='download_image'),  # Add this line
]
