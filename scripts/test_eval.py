from app import evaluate_email, evaluate_url

print("=== URL tests ===")
for url in ["https://kongu.ac.in/", "http://paypal-secure-login.com/"]:
    r = evaluate_url(url)
    print(url, "->", r["best_prediction"], "| best:", r["best_model"])

print("\n=== Email tests ===")
emails = [
    ("Semester Exam Schedule", "registrar@kongu.ac.in", "Dear students, exam schedule is on the portal."),
    ("URGENT: Verify PayPal", "security@paypa1-secure.com", "Account suspended! Click http://fake.com/login NOW!!!"),
    ("Team meeting", "ceo@gmail.com", "Send me your phone number urgently for a wire transfer."),
]
for subject, sender, body in emails:
    r = evaluate_email(subject, sender, body)
    print(sender, "->", r["best_prediction"], "| best:", r["best_model"])
