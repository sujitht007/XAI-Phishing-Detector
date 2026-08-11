import os

from flask import Flask, jsonify, render_template, request

from src.email_features import (
    EMAIL_FEATURE_COLUMNS,
    extract_features_from_email,
    is_clearly_legitimate_email,
)
from src.evaluator import load_model_bundle, run_evaluation
from src.features import FEATURE_COLUMNS, extract_features_from_url, is_clearly_legitimate
from src.phone_features import (
    PHONE_FEATURE_COLUMNS,
    extract_features_from_phone,
    is_clearly_legitimate_phone,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

url_models, url_x_train, url_explainer_cache = load_model_bundle(BASE_DIR)
email_models, email_x_train, email_explainer_cache = load_model_bundle(BASE_DIR, "email_")
phone_models, phone_x_train, phone_explainer_cache = load_model_bundle(BASE_DIR, "phone_")


def evaluate_url(url: str):
    features = extract_features_from_url(url)
    trusted_override = is_clearly_legitimate(features)
    trusted_note = (
        " This URL belongs to a trusted institutional domain (.ac.in, .edu, .gov, etc.) "
        "with no phishing indicators, so it is treated as legitimate."
        if trusted_override
        else ""
    )
    return run_evaluation(
        url_models,
        url_x_train,
        url_explainer_cache,
        features,
        FEATURE_COLUMNS,
        "URL",
        url,
        trusted_override,
        trusted_note,
        "URL",
    )


def evaluate_email(subject: str, sender: str, body: str):
    features = extract_features_from_email(subject, sender, body)
    trusted_override = is_clearly_legitimate_email(features)
    trusted_note = (
        " This email comes from a trusted institutional sender with no phishing indicators, "
        "so it is treated as legitimate."
        if trusted_override
        else ""
    )
    display = f"From: {sender or 'N/A'} | Subject: {subject or 'N/A'}"
    return run_evaluation(
        email_models,
        email_x_train,
        email_explainer_cache,
        features,
        EMAIL_FEATURE_COLUMNS,
        "Email",
        display,
        trusted_override,
        trusted_note,
        "email",
    )


def evaluate_phone(phone_number: str):
    features = extract_features_from_phone(phone_number)
    trusted_override = is_clearly_legitimate_phone(features)
    trusted_note = (
        " This phone number looks like a valid international contact number with no phishing flags, "
        "so the system treats it as legitimate."
        if trusted_override
        else ""
    )
    return run_evaluation(
        phone_models,
        phone_x_train,
        phone_explainer_cache,
        features,
        PHONE_FEATURE_COLUMNS,
        "Phone",
        phone_number,
        trusted_override,
        trusted_note,
        "phone",
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", analysis_type="url")


@app.route("/analyze", methods=["POST"])
def analyze():
    analysis_type = request.form.get("analysis_type", "url").strip().lower()

    if analysis_type == "email":
        subject = request.form.get("email_subject", "").strip()
        sender = request.form.get("email_sender", "").strip()
        body = request.form.get("email_body", "").strip()
        if not body and not subject:
            return render_template(
                "index.html",
                error="Please enter at least an email subject or body.",
                analysis_type="email",
            ), 400
        result = evaluate_email(subject, sender, body)
    elif analysis_type == "phone":
        phone_number = request.form.get("phone_text", "").strip()
        if not phone_number:
            return render_template(
                "index.html",
                error="Please enter a phone number to analyze.",
                analysis_type="phone",
            ), 400
        result = evaluate_phone(phone_number)
    else:
        url = request.form.get("url_text", "").strip()
        if not url:
            return render_template(
                "index.html",
                error="Please enter a URL to analyze.",
                analysis_type="url",
            ), 400
        result = evaluate_url(url)

    return render_template("result.html", **result)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/predict", methods=["OPTIONS", "POST"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.is_json:
        payload = request.get_json(silent=True) or {}
        analysis_type = payload.get("analysis_type", "url").strip().lower()
        if analysis_type == "email":
            subject = payload.get("subject", "")
            sender = payload.get("sender", "")
            body = payload.get("body", "")
            if not body and not subject:
                return jsonify({"error": "Please provide email subject or body."}), 400
            return jsonify(evaluate_email(subject, sender, body))
        if analysis_type == "phone":
            phone_number = payload.get("phone", "")
            if not phone_number:
                return jsonify({"error": "Please provide a phone number."}), 400
            return jsonify(evaluate_phone(phone_number))
        url = payload.get("url", "")
        if not url:
            return jsonify({"error": "Please provide a URL."}), 400
        return jsonify(evaluate_url(url))

    analysis_type = request.form.get("analysis_type", "url").strip().lower()
    if analysis_type == "email":
        subject = request.form.get("email_subject", "")
        sender = request.form.get("email_sender", "")
        body = request.form.get("email_body", "")
        if not body and not subject:
            return jsonify({"error": "Please provide email subject or body."}), 400
        return jsonify(evaluate_email(subject, sender, body))
    if analysis_type == "phone":
        phone_number = request.form.get("phone_text", "")
        if not phone_number:
            return jsonify({"error": "Please provide a phone number."}), 400
        return jsonify(evaluate_phone(phone_number))

    url = request.form.get("url_text", "")
    if not url:
        return jsonify({"error": "Please provide a URL."}), 400
    return jsonify(evaluate_url(url))


if __name__ == "__main__":
    app.run(debug=True)
