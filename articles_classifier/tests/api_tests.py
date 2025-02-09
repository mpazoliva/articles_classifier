"""
This script reads a JSON file containing multiple research abstracts, sends each abstract
to the classification API, and displays the response.

Usage:
- Ensure the Django server is running locally on port 8000.
- Place the test abstracts in a JSON file formatted as a list of dictionaries,
  where each dictionary contains an "abstract" key.
- Run this script to evaluate the API's performance on multiple abstracts.

Expected Behavior:
- Reads a JSON file containing abstracts.
- Sends each abstract in a separate POST request.
- Prints the classified category for each abstract if the request is successful.
- Displays an error message if a request fails.
"""

import json

import requests

# Defining the API endpoint URL
url = "http://127.0.0.1:8000/api/classify/"

# Loading abstracts from the JSON file
with open(
        "/Users/mariapazoliva/PycharmProjects/ArticlesClassifier/jupyter_notebooks/data_arxiv_articles/test_payloads.json",
        "r", encoding="utf-8") as file:
    data = json.load(file)

# Ensuring the data is a list of dictionaries
if isinstance(data, list):
    abstracts = [entry["abstract"] for entry in data if "abstract" in entry]
else:
    raise ValueError("The JSON file does not contain a list of abstracts.")

# Iterating through each abstract and send a request to the API
for i, abstract in enumerate(abstracts, start=1):
    payload = {"abstract": abstract}
    response = requests.post(url, json=payload)

    # Printing results
    if response.status_code == 200:
        print(f"Response for abstract {i}:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Request {i} failed with status code {response.status_code}")
        print(response.text)
