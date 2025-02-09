"""
This script sends a sample research abstract to the classification API and displays the response.

Usage:
- Ensure the Django server is running locally on port 8000.
- Run this script to evaluate the API's performance with a sample input.

Expected Behavior:
- Sends a POST request with an abstract.
- Prints the classified category if the request is successful.
- Displays an error message if the request fails.
"""

import json

import requests

# Defining API endpoint URL (since running the server locally use localhost and the appropriate port)
url = "http://127.0.0.1:8000/api/classify/"

# Sampling JSON payload
payload = {
    "abstract": "A set of analog electronics boards for serial readout of silicon strip\nsensors was fabricated. A commercially available amplifier is mounted on a\nhomemade hybrid board in order to receive analog signals from silicon strip\nsensors. Also, another homemade circuit board is fabricated in order to\ntranslate amplifier control signals into a suitable format and to provide bias\nvoltage to the amplifier as well as to the silicon sensors. We discuss\ntechnical details of the fabrication process and performance of the circuit\nboards we developed."
}

# Sending the POST request to the API endpoint.
response = requests.post(url, json=payload)

# Checking if the request was successful.
if response.status_code == 200:
    print("Response received:")
    print(json.dumps(response.json(), indent=2))
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
