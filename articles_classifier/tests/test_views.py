"""
Unit tests for the `classify` API endpoint in the Articles Classifier application.

This test suite verifies the behavior of the `ClassifyArticleView` by:
- Mocking the classification model using `unittest.mock.patch` to isolate API behavior.
- Testing responses for valid and invalid payloads.

Test Cases:
1. `test_classify_article_valid_payload`: Ensures the API returns a valid category when given a proper abstract.
2. `test_classify_article_empty_abstract`: Checks that an empty abstract results in a 400 Bad Request response.
3. `test_classify_article_missing_abstract`: Ensures missing abstract fields return a 400 Bad Request response.

Usage:
- Run the tests using Django’s test runner: python manage.py test articles_classifier.tests
"""

from unittest.mock import patch

from django.test import TestCase
from rest_framework.test import APIClient


class TestClassifyArticleView(TestCase):
    def setUp(self):
        self.client = APIClient()

        # Patching ClassifierFactory instead of classifier_instance
        patcher = patch('articles_classifier.views.ClassifierFactory.get_classifier')
        self.mock_classifier_factory = patcher.start()
        self.addCleanup(patcher.stop)

        # Mocking the classify method inside the returned instance
        self.mock_classifier = self.mock_classifier_factory.return_value
        self.mock_classifier.classify.return_value = ("cs", {"cs": 0.9, "math": 0.1})

    def test_classify_article_valid_payload(self):
        response = self.client.post('/api/classify/', {"abstract": "This is about AI."}, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('predicted_category', response.json())

    def test_classify_article_empty_abstract(self):
        response = self.client.post('/api/classify/', {"abstract": ""}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_classify_article_missing_abstract(self):
        response = self.client.post('/api/classify/', {}, format='json')
        self.assertEqual(response.status_code, 400)
