"""
Database model for storing research articles with their titles, abstracts, and categories.
"""

from django.db import models


class Article(models.Model):
    title = models.CharField(max_length=250)
    abstract = models.TextField()
    category = models.CharField(max_length=50)
