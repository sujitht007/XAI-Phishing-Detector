// Content script: observe an element with id "i" (as user described) and sync its text to chrome.storage.local
(function () {
  function readAndStore() {
    try {
      const el = document.getElementById('i');
      if (!el) return;
      const value = el.innerText || el.textContent || el.value || '';
      chrome.storage.local.set({ observed_i: value });
    } catch (e) {
      // ignore errors in page context
    }
  }

  // Initial read
  readAndStore();

  // Observe DOM changes for that element if present
  const observer = new MutationObserver((mutations) => {
    let changed = false;
    for (const m of mutations) {
      if (m.type === 'characterData' || m.type === 'childList' || m.type === 'attributes') {
        changed = true;
        break;
      }
    }
    if (changed) readAndStore();
  });

  function attachObserverToEl() {
    const el = document.getElementById('i');
    if (!el) return;
    observer.observe(el, { childList: true, subtree: true, characterData: true, attributes: true });
  }

  // If element added later, watch for it
  const bodyObserver = new MutationObserver(() => {
    if (document.getElementById('i')) {
      attachObserverToEl();
      bodyObserver.disconnect();
    }
  });
  bodyObserver.observe(document.documentElement || document.body, { childList: true, subtree: true });

  // Also write value periodically as a fallback
  setInterval(readAndStore, 2000);
})();