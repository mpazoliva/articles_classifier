"""
Views for classifying research articles based on their abstracts using a machine learning model.
"""

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from articles_classifier.ml.classifier import ClassifierFactory

MODEL_PATH = settings.MODEL_PATH

FULL_CATEGORY_NAMES = {
    'phys': 'Physics', 'math': 'Mathematics', 'cs': 'Computer Science',
    'q-bio': 'Quantitative Biology', 'q-fin': 'Quantitative Finance',
    'stat': 'Statistics', 'eess': 'Electrical Engineering and Systems Science', 'econ': 'Economics'
}


@api_view(['POST'])
def classify_article(request):
    """
      Classifies a research article based on its abstract.

      This view accepts a JSON payload containing an 'abstract' field and uses a machine learning
      model to predict the most relevant research category. The model is retrieved using a
      Singleton-based Factory Pattern to ensure efficient inference.

      Args:
          request (HttpRequest): The HTTP request object containing the article abstract.

      Returns:
          Response: A JSON response with the following fields:
              - predicted_category (str): The predicted category label (e.g., "cs" for Computer Science).
              - full_category_name (str): The human-readable category name.
              - probabilities (dict): Confidence scores for each category.
              - message (str): A user-friendly message indicating the predicted category.

      Raises:
          400 Bad Request:
              - If the 'abstract' field is missing.
              - If the model encounters an invalid input.
          500 Internal Server Error:
              - If an unexpected error occurs during classification.
      """

    abstract = request.data.get('abstract')
    if not abstract:
        return Response({'error': "The 'abstract' field is required."}, status=status.HTTP_400_BAD_REQUEST)

    classifier = ClassifierFactory.get_classifier(model_type="bert", model_path=MODEL_PATH)

    try:
        predicted_category, probabilities = classifier.classify(abstract)
    except ValueError as ve:
        return Response({'error': str(ve)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({'error': f"Unexpected error: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    full_category_name = FULL_CATEGORY_NAMES.get(predicted_category, predicted_category)

    return Response({
        'predicted_category': predicted_category,
        'full_category_name': full_category_name,
        'probabilities': probabilities,
        'message': f"Your abstract belongs to the category: {full_category_name}."
    }, status=status.HTTP_200_OK)
