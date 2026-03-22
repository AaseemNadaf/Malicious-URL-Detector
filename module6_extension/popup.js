// popup.js
// Reads stored result for current tab and renders it.

var API_HEALTH_URL = 'http://localhost:5000/health';

document.addEventListener('DOMContentLoaded', async function() {
  checkApiHealth();

  var tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  var tab  = tabs[0];
  if (!tab) return;

  var key  = 'result_' + tab.id;
  var data = await chrome.storage.local.get(null);

  var result = data[key];

  // If no tab-specific result, find the most recent one
  if (!result) {
    var mostRecent     = null;
    var mostRecentTime = 0;
    for (var k in data) {
      if (k.startsWith('result_') && data[k].checkedAt) {
        if (data[k].checkedAt > mostRecentTime) {
          mostRecentTime = data[k].checkedAt;
          mostRecent     = data[k];
        }
      }
    }
    result = mostRecent;
  }

  if (!result) return;

  renderResult(result);
});


async function checkApiHealth() {
  var dot    = document.getElementById('api-dot');
  var status = document.getElementById('api-status');
  try {
    var res = await fetch(API_HEALTH_URL);
    if (res.ok) {
      dot.classList.remove('offline');
      status.textContent = 'API connected';
    } else {
      throw new Error('not ok');
    }
  } catch(e) {
    dot.classList.add('offline');
    status.textContent = 'API offline';
  }
}


function renderResult(result) {
  var container = document.getElementById('main-state');

  if (result.error) {
    container.innerHTML =
      '<div class="result-card error">'
      + '<div class="verdict-row">'
      + '<span class="verdict-icon">⚠️</span>'
      + '<div>'
      + '<div class="verdict-label" style="color:#a8a29e">Error</div>'
      + '<div class="verdict-sub" style="color:#78716c">' + escapeHtml(result.message) + '</div>'
      + '</div></div></div>';
    return;
  }

  var isMalicious   = result.label === 'malicious';
  var cardClass     = result.whitelisted ? 'whitelisted' : result.label;
  var icon          = isMalicious ? '🚨' : '✅';
  var verdictText   = isMalicious ? 'Malicious' : 'Safe';
  var confidencePct = Math.round((result.confidence || 0) * 100);
  var checkedTime   = result.checkedAt
    ? new Date(result.checkedAt).toLocaleTimeString() : '';

  var shortUrl = (result.url || '').length > 52
    ? result.url.slice(0, 49) + '...'
    : (result.url || '');

  var riskDisplay = result.whitelisted
    ? 'On trusted whitelist'
    : 'Risk score: ' + ((result.risk_score || 0) * 100).toFixed(1) + '%';

  var whitelistBadge = result.whitelisted
    ? '<span class="whitelist-badge">Trusted domain</span>' : '';

  // ── Build LIME top_tokens section ──────────────────────────
  var tokensHtml = '';
  var topTokens  = result.top_tokens || [];

  if (isMalicious && !result.whitelisted && topTokens.length > 0) {
    var maxImp = Math.max.apply(null,
      topTokens.map(function(t) { return Math.abs(t.importance || 0); })
    );
    if (maxImp === 0) maxImp = 1;

    var rows = topTokens.slice(0, 4).map(function(t) {
      var pct    = Math.round((Math.abs(t.importance || 0) / maxImp) * 100);
      var reason = t.reason ? escapeHtml(t.reason) : 'suspicious pattern';
      return '<div class="token-row">'
        + '<div class="token-top">'
        + '<span class="token-text">' + escapeHtml(t.token) + '</span>'
        + '<div class="token-bar-track">'
        + '<div class="token-bar-fill" style="width:' + pct + '%"></div>'
        + '</div>'
        + '<span class="token-score">' + (t.importance || 0).toFixed(2) + '</span>'
        + '</div>'
        + '<div class="token-reason">' + reason + '</div>'
        + '</div>';
    }).join('');

    if (rows) {
      tokensHtml =
        '<div class="segments-title">Suspicious tokens</div>'
        + rows;
    }
  }

  // ── Reasons pills ───────────────────────────────────────────
  var reasonsHtml = '';
  var reasons     = result.reasons || [];
  if (isMalicious && !result.whitelisted && reasons.length > 0) {
    var pills = reasons.slice(0, 3).map(function(r) {
      return '<span class="reason-pill">' + escapeHtml(r) + '</span>';
    }).join('');
    reasonsHtml = '<div class="reasons-row">' + pills + '</div>';
  }

  container.innerHTML =
    '<div class="result-card ' + cardClass + '">'

    + '<div class="verdict-row">'
    + '<span class="verdict-icon">' + icon + '</span>'
    + '<div>'
    + '<div class="verdict-label ' + result.label + '">' + verdictText + '</div>'
    + '<div class="verdict-sub ' + result.label + '">' + riskDisplay + '</div>'
    + whitelistBadge
    + '</div></div>'

    + '<div class="url-box">' + escapeHtml(shortUrl) + '</div>'

    + '<div class="conf-row">'
    + '<span class="conf-label">Confidence</span>'
    + '<div class="conf-track">'
    + '<div class="conf-fill ' + result.label
    + '" style="width:' + confidencePct + '%"></div>'
    + '</div>'
    + '<span class="conf-pct">' + confidencePct + '%</span>'
    + '</div>'

    + '<div class="explanation-title">Explanation</div>'
    + '<div class="explanation-text">'
    + escapeHtml(result.explanation || '') + '</div>'

    + reasonsHtml
    + tokensHtml
    + '</div>';

  document.getElementById('checked-at').textContent =
    checkedTime ? 'Last checked ' + checkedTime : '';
}


function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}