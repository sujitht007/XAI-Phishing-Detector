import os

from flask import Flask, jsonify, render_template, request

from src.email_features import (
    EMAIL_SENDER_FEATURE_COLUMNS,
    extract_features_from_sender,
    heuristic_sender_check,
    analyze_email_address,
)
from src.evaluator import (
    email_explainer_cache,
    email_models,
    email_models_loaded,
    email_x_train,
    phone_explainer_cache,
    phone_models,
    phone_x_train,
    run_evaluation,
    url_explainer_cache,
    url_models,
    url_x_train,
)
from src.features import FEATURE_COLUMNS, extract_features_from_url, is_clearly_legitimate
from src.phone_features import (
    PHONE_FEATURE_COLUMNS,
    extract_features_from_phone,
    is_clearly_legitimate_phone,
    is_valid_phone,
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_url(url: str):
    features = extract_features_from_url(url)
    trusted_override = is_clearly_legitimate(features)
    trusted_note = (
        " This URL matches a known legitimate domain with no phishing indicators."
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


def evaluate_email(sender: str):
    # Prefer ML model if available; otherwise use dedicated email-address-only heuristic pipeline
    if email_models_loaded:
        features = extract_features_from_sender(sender)
        sender_local_typo = features.get("sender_local_typo") == 1
        trusted_override = features.get("is_trusted_sender") == 1 and not sender_local_typo
        trusted_note = (
            " Official corporate or institutional sender domain recognized."
            if trusted_override
            else " Sender ID does not exactly match the verified recruitment mailbox."
            if sender_local_typo
            else ""
        )
        return run_evaluation(
            email_models,
            email_x_train,
            email_explainer_cache,
            features,
            EMAIL_SENDER_FEATURE_COLUMNS,
            "Sender",
            sender,
            trusted_override,
            trusted_note,
            "email",
            forced_prediction="Phishing" if sender_local_typo else None,
        )

    r = analyze_email_address(sender)
    label = r.get("label", "Invalid Email Address")
    # Map labels to previous field meanings
    prediction_map = {
        "Invalid Email Address": "Invalid",
        "Suspicious Email Address": "Suspicious",
        "Likely Legitimate Email Address": "Legit",
    }
    mapped_prediction = prediction_map.get(label, "Suspicious")
    confidence = r.get("confidence", 0.0)
    explanation = r.get("explanation", "")
    features = r.get("features", {})

    return {
        "analysis_type": "email",
        "input_label": "Sender",
        "input_value": sender,
        "features": features,
        "feature_columns": r.get("feature_columns", EMAIL_SENDER_FEATURE_COLUMNS),
        "best_model": "EmailAddressPipeline",
        "best_prediction": mapped_prediction,
        "best_confidence": confidence,
        "model_results": [],
        "best_explainer": "",
        "explainer_results": [],
        "selected_top_features": [],
        "explanation": explanation,
        "trusted_override": mapped_prediction == "Legit",
    }


def evaluate_phone(phone_number: str):
    features = extract_features_from_phone(phone_number)

    # Validate format first — treat malformed numbers as phishing heuristic
    if not is_valid_phone(phone_number):
        trusted_override = False
        return {
            "analysis_type": "phone",
            "input_label": "Phone",
            "input_value": phone_number,
            "features": features,
            "feature_columns": PHONE_FEATURE_COLUMNS,
            "best_model": "Heuristic",
            "best_prediction": "Phishing",
            "best_confidence": 0.99,
            "model_results": [],
            "best_explainer": "",
            "explainer_results": [],
            "selected_top_features": [],
            "explanation": "Input rejected: phone number format is invalid. Marked as Phishing by heuristic.",
            "trusted_override": False,
        }

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
        # For new flow we only require an email address (sender)
        sender = request.form.get("email_sender", "").strip()
        if not sender:
            return render_template(
                "index.html",
                error="Please enter the sender's email address.",
                analysis_type="email",
            ), 400
        result = evaluate_email(sender)
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
            sender = payload.get("sender", "")
            if not sender:
                return jsonify({"error": "Please provide sender email address."}), 400
            return jsonify(evaluate_email(sender))
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
            sender = request.form.get("email_sender", "")
            if not sender:
                return jsonify({"error": "Please provide sender email address."}), 400
            return jsonify(evaluate_email(sender))
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
    app.run(host="0.0.0.0",
            port=int(os.environ.get("PORT",5000))
            )
