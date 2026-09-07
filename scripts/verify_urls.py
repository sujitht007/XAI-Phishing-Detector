import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from app import evaluate_url

tests = [
    "http://konu.edu",
    "http://konu.edu/login",
    "aa",
    "https://mit.edu",
    "https://kongu.ac.in",
    "https://chatgpt.com/",
    "https://openai.com/",
    "http://192.168.1.1/login",
    "http://paypal-secure-login.com/",
]

for url in tests:
    result = evaluate_url(url)
    print(f"{url!r} -> {result['best_prediction']} ({result['best_confidence']})")
