"""
Admin configuration for managing the Article model in the Django admin panel.
"""

from django.contrib import admin

from articles_classifier.models import Article

admin.site.register(Article)
