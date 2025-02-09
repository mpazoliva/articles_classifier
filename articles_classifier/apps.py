"""
Application configuration for the Articles Classifier Django app.
"""

from django.apps import AppConfig


class ArticlesClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = "articles_classifier"
