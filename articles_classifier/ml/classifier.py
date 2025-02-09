"""
This module defines two classes for classifying research article abstracts
into predefined categories using a Transformers-based model trained on the arXiv dataset
(see jupyter_notebooks/classification_model.ipynb).

Usage:
    - Use ClassifierFactory.get_classifier(model_type, model_path) to retrieve an instance of the Classifier.
    - Call classify(abstract) on a Classifier instance to classify a given abstract into a category.

Categories:
    1 - 'cs' (Computer Science)
    2 - 'econ' (Economics)
    3 - 'eess' (Electrical Engineering and Systems Science)
    4 - 'math' (Mathematics)
    5 - 'phys' (Physics)
    6 - 'q-bio' (Quantitative Biology)
    7 - 'q-fin' (Quantitative Finance)
    8 - 'stat' (Statistics)
"""

import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from utils.clean_text import clean_text


class Classifier:
    """
    This class implements a singleton pattern to ensure only one instance of the classifier is created.
    It loads a model for sequence classification and provides a method to classify abstracts.
    """
    _instance = None  # Singleton instance

    def __new__(cls, model_path=None, tokenizer_name='bert-base-uncased', categories=None):
        """ Ensure only one instance is created (Singleton) """
        if cls._instance is None:
            cls._instance = super(Classifier, cls).__new__(cls)
            cls._instance._initialize(model_path, tokenizer_name, categories)
        return cls._instance

    def _initialize(self, model_path, tokenizer_name, categories):
        """ Initializes the classifier """
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.categories = categories or ['cs', 'econ', 'eess', 'math', 'phys', 'q-bio', 'q-fin', 'stat']

    def classify(self, abstract):
        """
        Classifies the given abstract into one of the predefined categories.

        Args: abstract (str): The research abstract text to classify.

        Returns: tuple[str, dict[str, float]]
        A tuple containing:
            - The predicted category (str).
            - A dictionary with category labels as keys and confidence scores as values.

        Raises:
            ValueError: If `abstract` is not a valid non-empty string.
            Exception: If an error occurs during model inference.
        """
        if not isinstance(abstract, str) or not abstract.strip():
            raise ValueError("Input 'abstract' must be a nonempty string.")

        try:
            cleaned_abstract = clean_text(abstract)
            inputs = self.tokenizer(cleaned_abstract, return_tensors='tf', truncation=True, padding=True,
                                    max_length=512)
            outputs = self.model(inputs)
            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]

        except Exception as e:
            raise Exception(f"Error during model inference: {e}") from e

        predicted_label_index = int(np.argmax(probabilities))
        predicted_category = self.categories[predicted_label_index]
        probs = {cat: float(prob) for cat, prob in zip(self.categories, probabilities)}

        return predicted_category, probs


class ClassifierFactory:
    """
    This class implements a factory pattern to manage classifier instances,
    ensuring that only one instance per model type is created.
    """
    _instances = {}

    @staticmethod
    def get_classifier(model_type="bert", model_path=None):
        """ Returns a classifier instance based on model_type."""
        if model_type not in ClassifierFactory._instances:
            if model_type == "bert":
                ClassifierFactory._instances[model_type] = Classifier(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        return ClassifierFactory._instances[model_type]
