"""
Unit tests for the `Classifier` class in the Articles Classifier application.

This test suite verifies the behavior of the `Classifier` model by:
- Mocking the Hugging Face transformer model, tokenizer, and text preprocessing function.
- Testing classification behavior with valid and invalid inputs.
- Ensuring error handling mechanisms work correctly.

Test Cases:
1. `test_classify_valid_input`:
   - Ensures the classifier returns the expected category and probability distribution for a valid abstract.
   - Verifies probabilities sum to 1.

2. `test_classify_invalid_input_empty_string`:
   - Checks that an empty string input raises a `ValueError`.

3. `test_classify_invalid_input_non_string`:
   - Ensures that non-string inputs (e.g., `None`) raise a `ValueError`.

4. `test_exception_in_inference`:
   - Simulates an exception during tokenization to verify the classifier's error handling.

Usage:
- Run the tests using: python -m unittest path/to/test_classifier.py

"""
import unittest
from unittest import mock

import tensorflow as tf

from articles_classifier.ml.classifier import Classifier


# Creating dummy classes to simulate the model and tokenizer.
class DummyModel:
    def __call__(self, inputs):
        # Simulating logits for one instance. Assuming there are 8 categories (as per the default in Classifier)
        # Returning logits that will make index 1 the highest (so predicted category 'econ' if categories not changed).
        logits = tf.constant([[1.0, 2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0]])
        # Returning an object with a 'logits' attribute.
        return type("DummyOutput", (), {"logits": logits})


class DummyTokenizer:
    def __call__(self, text, return_tensors, truncation, padding, max_length):
        # Returning a dummy tokenized output.
        return {"input_ids": tf.constant([[101, 102]]), "attention_mask": tf.constant([[1, 1]])}


# Creating a dummy clean_text function that returns the text unchanged.
def dummy_clean_text(text: str) -> str:
    return text


class TestClassifier(unittest.TestCase):

    def setUp(self):
        # Patching the model and tokenizer creation so that they return our dummy objects.
        patcher_model = mock.patch(
            "articles_classifier.ml.classifier.TFAutoModelForSequenceClassification.from_pretrained",
            return_value=DummyModel()
        )
        patcher_tokenizer = mock.patch(
            "articles_classifier.ml.classifier.AutoTokenizer.from_pretrained",
            return_value=DummyTokenizer()
        )
        # Patching the clean_text function to avoid calling spaCy.
        patcher_clean = mock.patch(
            "articles_classifier.ml.classifier.clean_text",
            side_effect=dummy_clean_text
        )
        self.addCleanup(patcher_model.stop)
        self.addCleanup(patcher_tokenizer.stop)
        self.addCleanup(patcher_clean.stop)
        self.mock_model = patcher_model.start()
        self.mock_tokenizer = patcher_tokenizer.start()
        self.mock_clean = patcher_clean.start()

        # Initializing the classifier (the parameters don't matter because of the patches)
        self.classifier = Classifier(model_path="dummy_path")

    def test_classify_valid_input(self):
        # Using a valid abstract.
        abstract = "This is a test abstract for classification."
        predicted_category, probabilities = self.classifier.classify(abstract)

        # According to our dummy logits, the highest score is at index 1.
        # Default categories are: ['cs', 'econ', 'eess', 'math', 'phys', 'q-bio', 'q-fin', 'stat']
        self.assertEqual(predicted_category, "econ")
        self.assertIsInstance(probabilities, dict)
        self.assertEqual(set(probabilities.keys()), set(self.classifier.categories))
        # Checking that probabilities sum to 1 (approximately)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=4)

    def test_classify_invalid_input_empty_string(self):
        # Testing that an empty string raises ValueError.
        with self.assertRaises(ValueError):
            self.classifier.classify("   ")

    def test_classify_invalid_input_non_string(self):
        # Testing that a non-string input raises ValueError.
        with self.assertRaises(ValueError):
            self.classifier.classify(None)

    def test_exception_in_inference(self):
        # Patching the __call__ method on the DummyTokenizer class itself
        with mock.patch.object(type(self.classifier.tokenizer), '__call__', side_effect=Exception("Dummy error")):
            with self.assertRaises(Exception) as context:
                self.classifier.classify("A valid abstract")
            # Checking if the exception message contains the expected text.
            self.assertIn("Error during model inference", str(context.exception))


if __name__ == '__main__':
    unittest.main()
