# ArticlesClassifier

**ArticlesClassifier** is a Django REST Framework (DRF) API project for classifying research article abstracts into predefined categories using a Hugging Face NLP model.

The API accepts an abstract, cleans and preprocesses it using spaCy, and returns the predicted category along with prediction probabilities.

The default model is a BERT-based model pretrained on the arXiv dataset (see 'jupyter_notebooks/classification_model.ipynb'), but others can be added.

The predefined categories are based on arXiv as follows:

- `cs`: Computer Science
- `econ`: Economics
- `eess`: Electrical Engineering and Systems Science
- `math`: Mathematics
- `phys`: Physics
- `q-bio`: Quantitative Biology
- `q-fin`: Quantitative Finance
- `stat`: Statistics

## Project Structure

```
ArticlesClassifier/
├── .venv/                          # Virtual environment folder
├── articles_classifier/            # Django app folder containing: admin, apps, classifier, models, serializers, urls, views, and tests folder
├── config/                         # Django project configuration containing: asgi, settings, urls, and wsgi
├── jupyter_notebooks/              # Notebooks for exploratory data analysis and model training with folders containing the data and the model
├── utils/                          # Utility functions: clean_text.py
├── db.sqlite3                      # SQLite database file
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── manage.py                       # Django management script
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd ArticlesClassifier
   ```

2. **Create and Activate a Virtual Environment:**

    - On macOS/Linux:
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```

    - On Windows:
      ```bash
      python -m venv .venv
      .venv\Scripts\activate
      ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Configure the Model Path:**
   Open the file `config/settings.py` and add (or modify) the following line to define the model path:
   ```python
   import os
   BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   MODEL_PATH = os.path.join('BASE_DIR', "jupyter_notebooks", "base_model")
   ```

6. **Set Up the Django Database:**
   Run the migrations to initialize the SQLite database:
   ```bash
   python manage.py migrate
   ```

## Running the Application

1. **Start the Django Development Server:**
   From the root directory (`ArticlesClassifier/`), run:
   ```bash
   python manage.py runserver
   ```
   The API endpoint will be accessible at:
   ```
   http://127.0.0.1:8000/api/classify/
   ```

2. **Test the API Endpoint:**
   You can test the API by running the provided tests script:
   ```bash
   python api_test.py
   ```
   This script sends a sample abstract to your API and prints the response, which includes the predicted category and
   associated probabilities.

## API Endpoint Details

- **URL:** `/api/classify/`
- **Method:** POST
- **Content-Type:** `application/json`
- **Description:** Classifies a research article abstract into a predefined category.

## Sample Request Payload and Response

### Request Payload

```json
{
  "abstract": "A set of analog electronics boards for serial readout of silicon strip sensors was fabricated. A commercially available amplifier is mounted on a homemade hybrid board to receive analog signals from silicon strip sensors. Also, another homemade circuit board is fabricated to translate amplifier control signals into a suitable format and provide bias voltage to the amplifier as well as to the silicon sensors. We discuss technical details of the fabrication process and performance of the circuit boards we developed."
}
```

### Response

```json
{
  "predicted_category": "cs",
  "full_category_name": "Computer Science",
  "probabilities": {
    "cs": 0.8,
    "econ": 0.1,
    "eess": 0.05,
    "math": 0.02,
    "phys": 0.01,
    "q-bio": 0.01,
    "q-fin": 0.005,
    "stat": 0.005
  },
  "message": "Your abstract belongs to the category: Computer Science."
}
```

## Testing

The project includes multiple testing scripts to ensure the API and classifier work as expected.

### 1. **Single Abstract API Test**

To send a single abstract to the API and check the classification response:

```bash
python api_test.py
```

This script:

- Sends a POST request with a single abstract.
- Prints the predicted category and probabilities.

### 2. **Batch API Testing**

To test the API with multiple abstracts from a JSON file:

```bash
python api_tests.py
```

Before running this, ensure:

- The test payload file exists at `jupyter_notebooks/data_arxiv_articles/test_payloads.json`.
- The file contains a list of abstracts in JSON format.

### 3. **Unit Tests for the Classifier**

To run unit tests for the `Classifier` class:

```bash
python -m unittest test_classifier.py
```

This test suite:

- Mocks the Hugging Face model and tokenizer.
- Validates classification behavior.
- Ensures error handling works correctly.

### 4. **Unit Tests for the API Endpoint**

To test the `/api/classify/` endpoint:

```bash
python manage.py test articles_classifier.tests
```

This test suite:

- Mocks the classifier to isolate API behavior.
- Ensures proper responses for valid and invalid inputs.

## Contact

For questions or contributions, please contact **María Paz Oliva** (mpazoliva23@gmail.com).
