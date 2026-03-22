// content.js
// Listens for link hovers AND clicks.
// Uses mouseover to pre-fetch results so they are ready before navigation.

if (!window.location.protocol.startsWith('http')) {
  // Do nothing on non-http pages
} else {

  var lastHoveredUrl  = null;
  var lastCheckedUrl  = null;
  var lastCheckedTime = 0;
  var DEBOUNCE_MS     = 2000;

  console.log('[URL Shield] content.js loaded on', window.location.href);

  // ── HOVER: pre-fetch result before the user even clicks ──────────────────
  // This gives the API time to respond before navigation happens.
  document.addEventListener('mouseover', function(e) {
    var target = e.target;
    while (target && target.tagName !== 'A') {
      target = target.parentElement;
    }
    if (!target || !target.href) return;

    var url = target.href;
    if (url.indexOf('http://') !== 0 && url.indexOf('https://') !== 0) return;
    if (url === lastHoveredUrl) return;

    lastHoveredUrl = url;

    console.log('[URL Shield] Hovering over:', url);

    try {
      chrome.runtime.sendMessage({
        type:    'CHECK_URL',
        url:     url,
        pageUrl: window.location.href
      }, function(response) {
        // Callback optional - background handles storage
        if (chrome.runtime.lastError) {
          // Suppress "no receiver" error if background not ready
        }
      });
    } catch(err) {
      console.log('[URL Shield] sendMessage error:', err.message);
    }

  }, true);


  // ── CLICK: send again on click to ensure latest result is stored ──────────
  document.addEventListener('click', function(e) {
    var target = e.target;
    while (target && target.tagName !== 'A') {
      target = target.parentElement;
    }
    if (!target || !target.href) return;

    var url = target.href;
    if (url.indexOf('http://') !== 0 && url.indexOf('https://') !== 0) return;

    var now = Date.now();
    if (url === lastCheckedUrl && (now - lastCheckedTime) < DEBOUNCE_MS) return;

    lastCheckedUrl  = url;
    lastCheckedTime = now;

    console.log('[URL Shield] Clicked:', url);

    try {
      chrome.runtime.sendMessage({
        type:    'CHECK_URL',
        url:     url,
        pageUrl: window.location.href
      }, function(response) {
        if (chrome.runtime.lastError) {
          // Suppress error
        }
      });
    } catch(err) {
      console.log('[URL Shield] sendMessage error:', err.message);
    }

  }, true);

}