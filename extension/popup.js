const modes = ["url", "email", "phone"];

const modeButtons = [
  ...document.querySelectorAll(".mode-btn")
];

const panels = {
  url: document.getElementById("url-panel"),
  email: document.getElementById("email-panel"),
  phone: document.getElementById("phone-panel"),
};

const messageEl =
  document.getElementById("message");

const resultEl =
  document.getElementById("result");

const form =
  document.getElementById("analyze-form");

const fillUrlBtn =
  document.getElementById("fill-url-btn");

const submitBtn =
  document.getElementById("submit-btn");

let activeMode = "url";


// ============================================================
// BACKEND API
// ============================================================

const API_URL =
  "https://xai-phishing-detector-3ipl.onrender.com/predict";


// ============================================================
// LISTEN FOR STORAGE CHANGES
// ============================================================

chrome.storage.onChanged.addListener(
  (changes, area) => {

    if (area !== "local") return;

    if (changes.observed_i) {

      const newVal =
        changes.observed_i.newValue || "";


      // EMAIL MODE
      if (activeMode === "email") {

        const el =
          document.getElementById(
            "email-sender"
          );

        if (el && !el.value) {
          el.value = newVal;
        }

      }


      // URL MODE
      else if (activeMode === "url") {

        const el =
          document.getElementById(
            "url-input"
          );

        if (el && !el.value) {
          el.value = newVal;
        }
      }
    }
  }
);


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


  // Clear previous result
  messageEl.textContent = "";

  resultEl.innerHTML = "";

  resultEl.classList.add("hidden");
}


// ============================================================
// MODE BUTTON EVENTS
// ============================================================

modeButtons.forEach((button) => {

  button.addEventListener(
    "click",
    () => {

      setMode(
        button.dataset.mode
      );

    }
  );

});


// ============================================================
// FILL CURRENT TAB URL
// ============================================================

fillUrlBtn.addEventListener(
  "click",
  async () => {

    try {

      const [tab] =
        await chrome.tabs.query({
          active: true,
          lastFocusedWindow: true,
        });


      if (tab?.url) {

        document.getElementById(
          "url-input"
        ).value = tab.url;

        messageEl.textContent =
          "Current page URL inserted.";

      }

      else {

        messageEl.textContent =
          "Could not detect the current tab URL.";

      }

    }

    catch (err) {

      console.error(
        "Tab URL error:",
        err
      );

      messageEl.textContent =
        "Unable to access current tab URL. Please try again.";

    }

  }
);


// ============================================================
// FORM SUBMISSION
// ============================================================

form.addEventListener(
  "submit",
  async (event) => {

    event.preventDefault();


    // Clear previous result
    messageEl.textContent =
      "Analyzing...";

    resultEl.innerHTML = "";

    resultEl.classList.add(
      "hidden"
    );


    // Disable button while analyzing
    submitBtn.disabled = true;

    submitBtn.textContent =
      "Analyzing...";


    // ========================================================
    // CREATE PAYLOAD
    // ========================================================

    const payload = {
      analysis_type: activeMode,
    };


    // ========================================================
    // URL MODE
    // ========================================================

    if (activeMode === "url") {

      payload.url =
        document
          .getElementById("url-input")
          .value
          .trim();


      if (!payload.url) {

        messageEl.textContent =
          "Please enter a URL.";

        resetSubmitButton();

        return;
      }
    }


    // ========================================================
    // EMAIL MODE
    // ========================================================

    else if (activeMode === "email") {

      payload.sender =
        document
          .getElementById("email-sender")
          .value
          .trim();


      if (!payload.sender) {

        messageEl.textContent =
          "Please enter the sender's email address.";

        resetSubmitButton();

        return;
      }
    }


    // ========================================================
    // PHONE MODE
    // ========================================================

    else if (activeMode === "phone") {

      payload.phone =
        document
          .getElementById("phone-input")
          .value
          .trim();


      if (!payload.phone) {

        messageEl.textContent =
          "Please enter a phone number.";

        resetSubmitButton();

        return;
      }
    }


    // ========================================================
    // SEND REQUEST TO RENDER
    // ========================================================

    try {

      console.log(
        "Sending request to:",
        API_URL
      );

      console.log(
        "Payload:",
        payload
      );


      const response =
        await fetch(
          API_URL,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body:
              JSON.stringify(payload),
          }
        );


      // ======================================================
      // CHECK HTTP RESPONSE
      // ======================================================

      if (!response.ok) {

        let errorData = null;

        try {

          errorData =
            await response.json();

        }

        catch (jsonError) {

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

        resetSubmitButton();

        return;
      }


      // ======================================================
      // GET JSON RESPONSE
      // ======================================================

      const data =
        await response.json();


      console.log(
        "Backend response:",
        data
      );


      // ======================================================
      // DISPLAY RESULT
      // ======================================================

      messageEl.textContent =
        "Analysis complete.";

      renderResult(data);

    }


    catch (err) {

      console.error(
        "API connection error:",
        err
      );


      messageEl.textContent =
        `Unable to reach the phishing detection server. ${
          err.message || err
        }`;

    }


    finally {

      resetSubmitButton();

    }

  }
);


// ============================================================
// RESET SUBMIT BUTTON
// ============================================================

function resetSubmitButton() {

  submitBtn.disabled = false;

  submitBtn.textContent =
    "Analyze";
}


// ============================================================
// RENDER RESULT
// ============================================================

function renderResult(data) {

  const rows = [];


  // ==========================================================
  // SUMMARY
  // ==========================================================

  const prediction =
    String(
      data.best_prediction || ""
    );


  const predictionClass =
    prediction
      .toLowerCase()
      .includes("phishing")
        ? "phishing"
        : "legitimate";


  rows.push(`

    <div class="section">

      <h2>Summary</h2>

      <div class="summary">


        <div class="field">

          <strong>Type</strong>

          <span>
            ${escapeHtml(
              String(
                data.analysis_type || ""
              ).toUpperCase()
            )}
          </span>

        </div>


        <div class="field">

          <strong>Input</strong>

          <span class="input-value">

            ${escapeHtml(
              data.input_value || ""
            )}

          </span>

        </div>


        <div class="field">

          <strong>Prediction</strong>

          <span>

            <span
              class="prediction ${predictionClass}">
              ${escapeHtml(prediction)}
            </span>

            ${
              data.best_confidence !==
              undefined
                ? `
                  <span class="confidence">
                    ${escapeHtml(
                      formatNumber(
                        data.best_confidence
                      )
                    )}
                  </span>
                `
                : ""
            }

          </span>

        </div>


        <div class="field">

          <strong>Best model</strong>

          <span>
            ${escapeHtml(
              data.best_model || ""
            )}
          </span>

        </div>


        <div class="field">

          <strong>Best explainer</strong>

          <span>
            ${escapeHtml(
              data.best_explainer || ""
            )}
          </span>

        </div>


      </div>

    </div>

  `);


  // ==========================================================
  // MODEL COMPARISON
  // ==========================================================

  if (
    Array.isArray(
      data.model_results
    )
  ) {

    rows.push(`

      <div class="section">

        <h3>Model Comparison</h3>

        <div class="table-wrapper">

          <table class="result-table">

            <thead>

              <tr>

                <th>Model</th>

                <th>Prediction</th>

                <th>Confidence</th>

                <th>Score</th>

              </tr>

            </thead>


            <tbody>

    `);


    data.model_results.forEach(
      (item) => {

        const itemPrediction =
          String(
            item.prediction || ""
          );


        const itemPredictionClass =
          itemPrediction
            .toLowerCase()
            .includes("phishing")
              ? "phishing"
              : "legitimate";


        rows.push(`

          <tr>

            <td>
              <strong>
                ${escapeHtml(
                  item.name || ""
                )}
              </strong>
            </td>


            <td>

              <span
                class="prediction ${itemPredictionClass}">

                ${escapeHtml(
                  itemPrediction
                )}

              </span>

            </td>


            <td>
              ${escapeHtml(
                formatNumber(
                  item.confidence
                )
              )}
            </td>


            <td>
              ${escapeHtml(
                formatNumber(
                  item.score
                )
              )}
            </td>

          </tr>

        `);

      }
    );


    rows.push(`

            </tbody>

          </table>

        </div>

      </div>

    `);

  }


  // ==========================================================
  // EXPLAINABILITY
  // ==========================================================

  if (
    Array.isArray(
      data.explainer_results
    )
  ) {

    rows.push(`

      <div class="section">

        <h3>Explainability</h3>

        <div class="table-wrapper">

          <table class="result-table">

            <thead>

              <tr>

                <th>Explainer</th>

                <th>Score</th>

                <th>Robustness</th>

                <th>Time</th>

              </tr>

            </thead>


            <tbody>

    `);


    data.explainer_results.forEach(
      (item) => {

        rows.push(`

          <tr>

            <td>

              <strong>
                ${escapeHtml(
                  item.name || ""
                )}
              </strong>

            </td>


            <td>
              ${escapeHtml(
                formatNumber(
                  item.score
                )
              )}
            </td>


            <td>
              ${escapeHtml(
                formatNumber(
                  item.robustness
                )
              )}
            </td>


            <td>
              ${escapeHtml(
                formatNumber(
                  item.complexity
                )
              )}s
            </td>

          </tr>

        `);

      }
    );


    rows.push(`

            </tbody>

          </table>

        </div>

      </div>

    `);

  }


  // ==========================================================
  // TOP FEATURES
  // ==========================================================

  if (
    Array.isArray(
      data.selected_top_features
    )
  ) {

    rows.push(`

      <div class="section">

        <h3>Top Features</h3>

        <div class="feature-list">

    `);


    data.selected_top_features.forEach(
      (feature) => {

        rows.push(`

          <div class="feature-item">

            <span class="feature-name">

              ${escapeHtml(
                feature.feature || ""
              )}

            </span>


            <span class="feature-value">

              ${escapeHtml(
                formatNumber(
                  feature.importance
                )
              )}

            </span>

          </div>

        `);

      }
    );


    rows.push(`

        </div>

      </div>

    `);

  }


  // ==========================================================
  // EXTRACTED FEATURES
  // ==========================================================

  if (
    data.features &&
    typeof data.features === "object"
  ) {

    rows.push(`

      <div class="section">

        <h3>Extracted Features</h3>

        <div class="feature-list">

    `);


    Object.entries(
      data.features
    ).forEach(
      ([key, value]) => {

        rows.push(`

          <div class="feature-item">

            <span class="feature-name">

              ${escapeHtml(key)}

            </span>


            <span class="feature-value">

              ${escapeHtml(
                String(value)
              )}

            </span>

          </div>

        `);

      }
    );


    rows.push(`

        </div>

      </div>

    `);

  }


  // ==========================================================
  // DISPLAY HTML
  // ==========================================================

  resultEl.innerHTML =
    rows.join("");


  resultEl.classList.remove(
    "hidden"
  );

}


// ============================================================
// FORMAT NUMBER
// ============================================================

function formatNumber(value) {

  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {

    return "-";

  }


  const number =
    Number(value);


  if (
    Number.isNaN(number)
  ) {

    return String(value);

  }


  if (
    Number.isInteger(number)
  ) {

    return String(number);

  }


  return number.toFixed(3);

}


// ============================================================
// HTML ESCAPING
// ============================================================

function escapeHtml(text) {

  const stringValue =
    String(text);


  return stringValue

    .replace(
      /&/g,
      "&amp;"
    )

    .replace(
      /</g,
      "&lt;"
    )

    .replace(
      />/g,
      "&gt;"
    )

    .replace(
      /"/g,
      "&quot;"
    )

    .replace(
      /'/g,
      "&#039;"
    );
}


// ============================================================
// INITIAL MODE
// ============================================================

setMode(activeMode);

