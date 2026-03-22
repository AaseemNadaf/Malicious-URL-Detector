// background.js
// Service worker. Receives URL from content.js,
// calls Flask API, stores result by tab ID and URL.

var API_URL = 'http://localhost:5000/predict';

// Prevent duplicate in-flight requests for the same URL
var inFlight = {};

chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  if (message.type === 'CHECK_URL') {
    var url   = message.url;
    var tabId = sender.tab ? sender.tab.id : null;

    console.log('[URL Shield BG] Received CHECK_URL:', url, 'tabId:', tabId);

    // Skip if already fetching this URL
    if (inFlight[url]) {
      console.log('[URL Shield BG] Already in-flight, skipping:', url);
      return;
    }

    checkUrl(url, tabId);
  }
  // Return true to keep message channel open for async response
  return true;
});


async function checkUrl(url, tabId) {
  inFlight[url] = true;

  try {
    console.log('[URL Shield BG] Calling API for:', url);

    var response = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ url: url })
    });

    if (!response.ok) {
      throw new Error('API returned status ' + response.status);
    }

    var result = await response.json();
    console.log('[URL Shield BG] Result:', result.label, result.confidence);

    // Store by tab ID (for popup to retrieve)
    var storageData = {};

    if (tabId) {
      storageData['result_' + tabId] = {
        url:          result.url,
        label:        result.label,
        confidence:   result.confidence,
        risk_score:   result.risk_score,
        explanation:  result.explanation,
        top_segments: result.top_segments,
        whitelisted:  result.whitelisted,
        checkedAt:    Date.now()
      };
    }

    // Also store by URL so popup can find it even after navigation
    var urlKey = 'result_url_' + btoa(url).slice(0, 50);
    storageData[urlKey] = storageData['result_' + tabId] || {
      url:          result.url,
      label:        result.label,
      confidence:   result.confidence,
      risk_score:   result.risk_score,
      explanation:  result.explanation,
      top_segments: result.top_segments,
      whitelisted:  result.whitelisted,
      checkedAt:    Date.now()
    };

    await chrome.storage.local.set(storageData);
    console.log('[URL Shield BG] Stored result for tabId:', tabId);

    // Update badge
    updateBadge(tabId, result.label);

  } catch(err) {
    console.error('[URL Shield BG] Error:', err.message);

    if (tabId) {
      var errData = {};
      errData['result_' + tabId] = {
        error:     true,
        message:   'Could not reach API. Is Flask running on localhost:5000?',
        checkedAt: Date.now()
      };
      await chrome.storage.local.set(errData);

      chrome.action.setBadgeText({ text: '?', tabId: tabId });
      chrome.action.setBadgeBackgroundColor({ color: '#888888', tabId: tabId });
    }

  } finally {
    delete inFlight[url];
  }
}


function updateBadge(tabId, label) {
  if (!tabId) return;

  if (label === 'malicious') {
    chrome.action.setBadgeText({ text: '!', tabId: tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#E53935', tabId: tabId });
  } else {
    chrome.action.setBadgeText({ text: 'OK', tabId: tabId });
    chrome.action.setBadgeBackgroundColor({ color: '#43A047', tabId: tabId });
  }
}