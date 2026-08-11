const modes = ["url", "email", "phone"];
const modeButtons = [...document.querySelectorAll(".mode-btn")];
const panels = {
  url: document.getElementById("url-panel"),
  email: document.getElementById("email-panel"),
  phone: document.getElementById("phone-panel"),
};
const messageEl = document.getElementById("message");
const resultEl = document.getElementById("result");
const form = document.getElementById("analyze-form");
const fillUrlBtn = document.getElementById("fill-url-btn");
let activeMode = "url";

function setMode(mode) {
  activeMode = mode;
  modeButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  modes.forEach((m) => {
    panels[m].classList.toggle("hidden", m !== mode);
  });
  messageEl.textContent = "";
  resultEl.textContent = "";
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

fillUrlBtn.addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (tab?.url) {
      document.getElementById("url-input").value = tab.url;
      messageEl.textContent = "Current page URL inserted.";
    } else {
      messageEl.textContent = "Could not detect the current tab URL.";
    }
  } catch (err) {
    messageEl.textContent = "Unable to access current tab URL. Please try again.";
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  messageEl.textContent = "Analyzing...";
  resultEl.textContent = "";

  const payload = { analysis_type: activeMode };

  if (activeMode === "url") {
    payload.url = document.getElementById("url-input").value.trim();
    if (!payload.url) {
      messageEl.textContent = "Please enter a URL.";
      return;
    }
  } else if (activeMode === "email") {
    payload.subject = document.getElementById("email-subject").value.trim();
    payload.sender = document.getElementById("email-sender").value.trim();
    payload.body = document.getElementById("email-body").value.trim();
    if (!payload.subject && !payload.body) {
      messageEl.textContent = "Please enter an email subject or body.";
      return;
    }
  } else if (activeMode === "phone") {
    payload.phone = document.getElementById("phone-input").value.trim();
    if (!payload.phone) {
      messageEl.textContent = "Please enter a phone number.";
      return;
    }
  }

  try {
    const endpoints = [
      "http://localhost:5000/predict",
      "http://127.0.0.1:5000/predict",
    ];
    let response = null;
    let lastError = null;

    for (const endpoint of endpoints) {
      try {
        response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (response) {
          break;
        }
      } catch (error) {
        lastError = error;
      }
    }

    if (!response) {
      throw lastError || new Error("No response from backend");
    }

    if (!response.ok) {
      const error = await response.json().catch(() => null);
      messageEl.textContent = error?.error || `Server error: ${response.status}`;
      return;
    }

    const data = await response.json();
    messageEl.textContent = "Analysis complete.";
    renderResult(data);
  } catch (err) {
    messageEl.textContent = `Unable to reach Flask server. Make sure it is running on port 5000. ${err.message || err}`;
    console.error(err);
  }
});

function renderResult(data) {
  const rows = [];
  rows.push(`<div class="section"><h2>Summary</h2><p><strong>Type:</strong> ${escapeHtml(data.analysis_type.toUpperCase())}</p><p><strong>Input:</strong> ${escapeHtml(data.input_value)}</p><p><strong>Prediction:</strong> ${escapeHtml(data.best_prediction)} (${escapeHtml(data.best_confidence)})</p><p><strong>Best model:</strong> ${escapeHtml(data.best_model)}</p><p><strong>Best explainer:</strong> ${escapeHtml(data.best_explainer)}</p></div>`);

  if (Array.isArray(data.model_results)) {
    rows.push("<div class=\"section\"><h2>Model comparison</h2><ul>");
    data.model_results.forEach((item) => {
      rows.push(`<li><strong>${escapeHtml(item.name)}</strong>: ${escapeHtml(item.prediction)}, confidence ${escapeHtml(item.confidence)}, score ${escapeHtml(item.score)}</li>`);
    });
    rows.push("</ul></div>");
  }

  if (Array.isArray(data.explainer_results)) {
    rows.push("<div class=\"section\"><h2>Explainability</h2><ul>");
    data.explainer_results.forEach((item) => {
      rows.push(`<li><strong>${escapeHtml(item.name)}</strong>: score ${escapeHtml(item.score)}, robustness ${escapeHtml(item.robustness)}, complexity ${escapeHtml(item.complexity)}s</li>`);
    });
    rows.push("</ul></div>");
  }

  if (Array.isArray(data.selected_top_features)) {
    rows.push("<div class=\"section\"><h2>Top features</h2><ul>");
    data.selected_top_features.forEach((feature) => {
      rows.push(`<li>${escapeHtml(feature.feature)}: ${escapeHtml(feature.importance)}</li>`);
    });
    rows.push("</ul></div>");
  }

  if (data.features && typeof data.features === "object") {
    rows.push("<div class=\"section\"><h2>Extracted features</h2><ul>");
    Object.entries(data.features).forEach(([key, value]) => {
      rows.push(`<li>${escapeHtml(key)}: ${escapeHtml(String(value))}</li>`);
    });
    rows.push("</ul></div>");
  }

  resultEl.innerHTML = rows.join("");
  resultEl.classList.remove("hidden");
}

function escapeHtml(text) {
  const stringValue = String(text);
  return stringValue
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

setMode(activeMode);
