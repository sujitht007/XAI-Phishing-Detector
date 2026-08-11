from app import evaluate_url

urls = [
    "https://kongu.ac.in/",
    "http://paypal-secure-login.com/",
    "http://192.168.1.1/secure",
    "https://google.com/",
]

for u in urls:
    r = evaluate_url(u)
    print("URL:", u)
    for m in r["model_results"]:
        mark = " <-- BEST" if m["name"] == r["best_model"] else ""
        print(
            f"  {m['name']}: pred={m['prediction']}, conf={m['confidence']}, "
            f"robust={m['robustness']}, complex={m['complexity']}, score={m['score']}{mark}"
        )
    print()
