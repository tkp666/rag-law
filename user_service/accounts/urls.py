from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('history/', views.get_history, name='get_history'),
    path('history/<int:history_id>/', views.delete_history, name='delete_history'),
    path('save-query/', views.save_query, name='save_query'),
    path('csrf/', views.get_csrf_token, name='csrf_token'),
]