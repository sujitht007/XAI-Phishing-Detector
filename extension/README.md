XAI Phishing Detector - Chrome Extension

This extension integrates with the local XAI Phishing Detector Flask backend (http://localhost:5000) to analyze URLs, phone numbers, and email addresses.

Key points:
- Email mode analyzes ONLY the sender email address (no subject/body required).
- The content script watches for a page element with id `i` and stores its text under `observed_i` in `chrome.storage.local`.
- The popup will prefill the URL or email sender input when the content script detects `observed_i`.

Load extension (developer mode):
1. Open Chrome and go to chrome://extensions
2. Enable "Developer mode" (top-right)
3. Click "Load unpacked" and select the `extension/` folder in this repository
4. Reload any pages you want the content script to run on.

Debugging tips:
- Inspect popup: open chrome://extensions, find the extension, click the "service worker" or "Inspect views: popup" link.
- Page console: right-click the page, Inspect -> Console. Content script console.logs will appear there.
- If changes don't appear, click "Reload" on the extension in chrome://extensions and refresh the target page.

Files:
- manifest.json: extension manifest (Manifest V3)
- popup.html, popup.js, styles.css: popup UI
- content_script.js: observes page element with id `i` and syncs to storage

For API backend, ensure Flask app is running at http://localhost:5000
