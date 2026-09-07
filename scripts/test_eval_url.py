from app import evaluate_url
import json

if __name__ == '__main__':
    print(json.dumps(evaluate_url('http://konu.edu'), indent=2))
