"""
URL configuration for the Articles Classifier API endpoints.
"""

from django.urls import path

from . import views

urlpatterns = [
    path('classify/', views.classify_article, name='classify_article'),
]
