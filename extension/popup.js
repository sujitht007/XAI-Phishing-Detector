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

// ============================================================
// BACKEND API
// ============================================================

const API_URL =
  "https://xai-phishing-detector-3ipl.onrender.com/predict";


// ============================================================
// LISTEN FOR STORAGE CHANGES
// Content script can update the observed value
// ============================================================

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local") return;

  if (changes.observed_i) {
    const newVal = changes.observed_i.newValue || "";

    // If active mode is email, put the value into sender field
    if (activeMode === "email") {
      const el = document.getElementById("email-sender");

      if (el && !el.value) {
        el.value = newVal;
      }

    // If active mode is URL, put the value into URL input
    } else if (activeMode === "url") {
      const el = document.getElementById("url-input");

      if (el && !el.value) {
        el.value = newVal;
      }
    }
  }
});


// ============================================================
// CHANGE ANALYSIS MODE
// ============================================================

function setMode(mode) {
  activeMode = mode;

  // Update active button
  modeButtons.forEach((btn) => {
    btn.classList.toggle(
      "active",
      btn.dataset.mode === mode
    );
  });

  // Show correct panel
  modes.forEach((m) => {
    if (panels[m]) {
      panels[m].classList.toggle(
        "hidden",
        m !== mode
      );
    }
  });

  // Clear previous messages/results
  messageEl.textContent = "";
  resultEl.textContent = "";
  resultEl.classList.add("hidden");
}


// ============================================================
// MODE BUTTON EVENTS
// ============================================================

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setMode(button.dataset.mode);
  });
});


// ============================================================
// FILL CURRENT TAB URL
// ============================================================

fillUrlBtn.addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      lastFocusedWindow: true,
    });

    if (tab?.url) {
      document.getElementById("url-input").value = tab.url;

      messageEl.textContent =
        "Current page URL inserted.";
    } else {
      messageEl.textContent =
        "Could not detect the current tab URL.";
    }

  } catch (err) {
    console.error("Tab URL error:", err);

    messageEl.textContent =
      "Unable to access current tab URL. Please try again.";
  }
});


// ============================================================
// FORM SUBMISSION
// ============================================================

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  messageEl.textContent = "Analyzing...";
  resultEl.textContent = "";
  resultEl.classList.add("hidden");

  // ----------------------------------------------------------
  // CREATE PAYLOAD
  // ----------------------------------------------------------

  const payload = {
    analysis_type: activeMode,
  };


  // ----------------------------------------------------------
  // URL MODE
  // ----------------------------------------------------------

  if (activeMode === "url") {
    payload.url =
      document.getElementById("url-input").value.trim();

    if (!payload.url) {
      messageEl.textContent =
        "Please enter a URL.";
      return;
    }
  }


  // ----------------------------------------------------------
  // EMAIL MODE
  // ----------------------------------------------------------

  else if (activeMode === "email") {
    payload.sender =
      document.getElementById("email-sender").value.trim();

    if (!payload.sender) {
      messageEl.textContent =
        "Please enter the sender's email address.";
      return;
    }
  }


  // ----------------------------------------------------------
  // PHONE MODE
  // ----------------------------------------------------------

  else if (activeMode === "phone") {
    payload.phone =
      document.getElementById("phone-input").value.trim();

    if (!payload.phone) {
      messageEl.textContent =
        "Please enter a phone number.";
      return;
    }
  }


  // ----------------------------------------------------------
  // SEND REQUEST TO RENDER
  // ----------------------------------------------------------

  try {
    console.log("Sending request to:", API_URL);
    console.log("Payload:", payload);

    const response = await fetch(API_URL, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
    });


    // --------------------------------------------------------
    // CHECK HTTP RESPONSE
    // --------------------------------------------------------

    if (!response.ok) {
      let errorData = null;

      try {
        errorData = await response.json();
      } catch (jsonError) {
        console.warn(
          "Could not parse error response:",
          jsonError
        );
      }

      console.error(
        "Backend returned error:",
        response.status,
        errorData
      );

      messageEl.textContent =
        errorData?.error ||
        `Server error: ${response.status}`;

      return;
    }


    // --------------------------------------------------------
    // GET JSON RESPONSE
    // --------------------------------------------------------

    const data = await response.json();

    console.log("Backend response:", data);


    // --------------------------------------------------------
    // DISPLAY RESULT
    // --------------------------------------------------------

    messageEl.textContent =
      "Analysis complete.";

    renderResult(data);

  } catch (err) {
    console.error(
      "API connection error:",
      err
    );

    messageEl.textContent =
      `Unable to reach the phishing detection server. ${
        err.message || err
      }`;
  }
});


// ============================================================
// RENDER RESULT
// ============================================================

function renderResult(data) {
  const rows = [];


  // ----------------------------------------------------------
  // SUMMARY
  // ----------------------------------------------------------

  rows.push(`
    <div class="section">
      <h2>Summary</h2>

      <p>
        <strong>Type:</strong>
        ${escapeHtml(
          String(data.analysis_type || "").toUpperCase()
        )}
      </p>

      <p>
        <strong>Input:</strong>
        ${escapeHtml(data.input_value || "")}
      </p>

      <p>
        <strong>Prediction:</strong>
        ${escapeHtml(data.best_prediction || "")}
        (${escapeHtml(data.best_confidence || "")})
      </p>

      <p>
        <strong>Best model:</strong>
        ${escapeHtml(data.best_model || "")}
      </p>

      <p>
        <strong>Best explainer:</strong>
        ${escapeHtml(data.best_explainer || "")}
      </p>
    </div>
  `);


  // ----------------------------------------------------------
  // MODEL COMPARISON
  // ----------------------------------------------------------

  if (Array.isArray(data.model_results)) {
    rows.push(`
      <div class="section">
        <h2>Model comparison</h2>
        <ul>
    `);

    data.model_results.forEach((item) => {
      rows.push(`
        <li>
          <strong>
            ${escapeHtml(item.name || "")}
          </strong>:

          ${escapeHtml(item.prediction || "")},

          confidence
          ${escapeHtml(item.confidence || "")},

          score
          ${escapeHtml(item.score || "")}
        </li>
      `);
    });

    rows.push(`
        </ul>
      </div>
    `);
  }


  // ----------------------------------------------------------
  // EXPLAINABILITY
  // ----------------------------------------------------------

  if (Array.isArray(data.explainer_results)) {
    rows.push(`
      <div class="section">
        <h2>Explainability</h2>
        <ul>
    `);

    data.explainer_results.forEach((item) => {
      rows.push(`
        <li>
          <strong>
            ${escapeHtml(item.name || "")}
          </strong>:

          score
          ${escapeHtml(item.score || "")},

          robustness
          ${escapeHtml(item.robustness || "")},

          complexity
          ${escapeHtml(item.complexity || "")}s
        </li>
      `);
    });

    rows.push(`
        </ul>
      </div>
    `);
  }


  // ----------------------------------------------------------
  // TOP FEATURES
  // ----------------------------------------------------------

  if (Array.isArray(data.selected_top_features)) {
    rows.push(`
      <div class="section">
        <h2>Top features</h2>
        <ul>
    `);

    data.selected_top_features.forEach((feature) => {
      rows.push(`
        <li>
          ${escapeHtml(feature.feature || "")}:

          ${escapeHtml(
            String(feature.importance ?? "")
          )}
        </li>
      `);
    });

    rows.push(`
        </ul>
      </div>
    `);
  }


  // ----------------------------------------------------------
  // EXTRACTED FEATURES
  // ----------------------------------------------------------

  if (
    data.features &&
    typeof data.features === "object"
  ) {
    rows.push(`
      <div class="section">
        <h2>Extracted features</h2>
        <ul>
    `);

    Object.entries(data.features).forEach(
      ([key, value]) => {
        rows.push(`
          <li>
            ${escapeHtml(key)}:
            ${escapeHtml(String(value))}
          </li>
        `);
      }
    );

    rows.push(`
        </ul>
      </div>
    `);
  }


  // ----------------------------------------------------------
  // DISPLAY HTML
  // ----------------------------------------------------------

  resultEl.innerHTML = rows.join("");

  resultEl.classList.remove("hidden");
}


// ============================================================
// HTML ESCAPING
// Prevents HTML/script injection in API response
// ============================================================

function escapeHtml(text) {
  const stringValue = String(text);

  return stringValue
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}


// ============================================================
// INITIAL MODE
// ============================================================

setMode(activeMode);