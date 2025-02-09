"""
Serializer for the Article model, defining how data is transformed to and from JSON.
"""

from rest_framework import serializers

from .models import Article


class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'abstract', 'category']
