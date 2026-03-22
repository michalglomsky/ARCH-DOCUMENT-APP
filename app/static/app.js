/* ============================================================
   ARCH Document Extractor — Frontend Logic
   ============================================================ */

'use strict';

// ---- PDF.js setup ----
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// ---- State ----
let currentDoc   = null;   // { name, stem }
let pdfDoc       = null;   // PDF.js document
let currentPage  = 1;
let lastPrediction = null; // last VLM extraction result
let renderTask   = null;

// ---- DOM refs ----
const docSelect    = document.getElementById('doc-select');
const prevBtn      = document.getElementById('prev-page');
const nextBtn      = document.getElementById('next-page');
const pageInfo     = document.getElementById('page-info');
const pdfCanvas    = document.getElementById('pdf-canvas');
const vlmStatus    = document.getElementById('vlm-status');
const maxPagesInput = document.getElementById('max-pages');
const spinner      = document.getElementById('spinner');
const spinnerMsg   = document.getElementById('spinner-msg');

const btnExtract   = document.getElementById('btn-extract');
const btnCompare   = document.getElementById('btn-compare');
const btnSave      = document.getElementById('btn-save');
const btnReextract = document.getElementById('btn-reextract');
const extractPanel = document.getElementById('extract-panel');

const comparePanel = document.getElementById('compare-panel');
const accuracyBarContainer = document.getElementById('accuracy-bar-container');
const accuracyFill = document.getElementById('accuracy-fill');
const accuracyLabel = document.getElementById('accuracy-label');
const accuracyPct  = document.getElementById('accuracy-pct');

const chatMessages = document.getElementById('chat-messages');
const chatInput    = document.getElementById('chat-input');
const btnSend      = document.getElementById('btn-send');

// ============================================================
// Spinner helpers
// ============================================================
function showSpinner(msg = 'Working…') {
  spinnerMsg.textContent = msg;
  spinner.classList.add('active');
}
function hideSpinner() {
  spinner.classList.remove('active');
}

// ============================================================
// API helpers
// ============================================================
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

// ============================================================
// VLM health check
// ============================================================
async function checkVlmHealth() {
  try {
    const data = await api('GET', '/api/vlm/health');
    if (data.status === 'ok') {
      const adapter = data.adapter && data.adapter !== 'none (zero-shot)'
        ? ' + LoRA' : ' (zero-shot)';
      vlmStatus.textContent = `VLM: ✓ ${data.device}${adapter}`;
      vlmStatus.className = 'ok';
    } else {
      vlmStatus.textContent = `VLM: ✗ offline`;
      vlmStatus.className = 'error';
    }
  } catch {
    vlmStatus.textContent = 'VLM: ✗ offline';
    vlmStatus.className = 'error';
  }
}

// ============================================================
// Document list
// ============================================================
async function loadDocuments() {
  try {
    const docs = await api('GET', '/api/documents');
    docSelect.innerHTML = '<option value="">— select document —</option>';
    docs.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.name;
      opt.textContent = d.name;
      docSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Failed to load documents:', e);
  }
}

// ============================================================
// PDF viewer
// ============================================================
async function loadPdf(filename) {
  if (!filename) return;
  showSpinner('Loading PDF…');
  try {
    const url = `/api/pdf/${encodeURIComponent(filename)}`;
    pdfDoc = await pdfjsLib.getDocument(url).promise;
    currentPage = 1;
    await renderPage(currentPage);
  } catch (e) {
    alert('Failed to load PDF: ' + e.message);
  } finally {
    hideSpinner();
  }
}

async function renderPage(pageNum) {
  if (!pdfDoc) return;
  if (renderTask) { renderTask.cancel(); renderTask = null; }

  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale: 1.4 });
  const ctx = pdfCanvas.getContext('2d');
  pdfCanvas.width  = viewport.width;
  pdfCanvas.height = viewport.height;

  renderTask = page.render({ canvasContext: ctx, viewport });
  await renderTask.promise.catch(e => {
    if (e.name !== 'RenderingCancelledException') throw e;
  });
  renderTask = null;

  pageInfo.textContent = `${pageNum} / ${pdfDoc.numPages}`;
  prevBtn.disabled = pageNum <= 1;
  nextBtn.disabled = pageNum >= pdfDoc.numPages;
}

prevBtn.addEventListener('click', () => {
  if (currentPage > 1) { currentPage--; renderPage(currentPage); }
});
nextBtn.addEventListener('click', () => {
  if (pdfDoc && currentPage < pdfDoc.numPages) { currentPage++; renderPage(currentPage); }
});

docSelect.addEventListener('change', () => {
  const name = docSelect.value;
  if (!name) return;
  currentDoc = { name, stem: name.replace(/\.pdf$/i, '') };
  lastPrediction = null;
  setExtractionButtons(false);
  extractPanel.innerHTML = '<p class="text-muted" style="padding:16px">Document loaded. Click Extract.</p>';
  clearCompare();
  loadPdf(name);
});

// ============================================================
// Extraction
// ============================================================
function setExtractionButtons(hasResult) {
  btnCompare.disabled   = !hasResult;
  btnSave.disabled      = !hasResult;
  btnReextract.disabled = !hasResult;
}

btnExtract.addEventListener('click', runExtract);
btnReextract.addEventListener('click', runExtract);

async function runExtract() {
  if (!currentDoc) { alert('Select a document first.'); return; }
  showSpinner('Extracting…');
  try {
    const result = await api('POST', '/api/extract', {
      pdf_name: currentDoc.name,
      max_pages: parseInt(maxPagesInput.value) || 6,
    });
    lastPrediction = result;
    renderExtraction(result);
    setExtractionButtons(true);
    clearCompare();
  } catch (e) {
    alert('Extraction failed: ' + e.message);
  } finally {
    hideSpinner();
  }
}

function renderExtraction(pred) {
  const flat = [
    { label: 'Nr wniosku',            key: 'nr_wniosku' },
    { label: 'Sposób wypełnienia',    key: 'sposob_wypelnienia' },
    { label: 'Flaga 7.9',             key: 'flaga_7_9' },
    { label: 'Nazwa inwestycji',       key: 'nazwa_inwestycji' },
    { label: 'Adres',                  key: 'adres' },
    { label: 'Teren inwestycji',       key: 'teren_inwestycji' },
    { label: 'Pow. zabudowy (całość)', key: 'pow_zabudowy_calosc' },
  ];

  let html = '';

  if (pred.needs_review) {
    html += '<div class="needs-review" style="padding:4px 8px;margin-bottom:8px">⚠ needs_review = true</div>';
  }
  if (pred._parse_error) {
    html += `<div class="needs-review" style="padding:4px 8px;margin-bottom:8px">⚠ parse error: ${esc(pred._parse_error)}</div>`;
  }

  flat.forEach(f => {
    html += fieldHtml(f.label, pred[f.key]);
  });

  // Buildings
  const buildings = pred.budynki || [];
  if (buildings.length) {
    html += '<div class="section-title">Budynki</div>';
    buildings.forEach((b, i) => {
      html += `<div class="building-block"><div class="building-title">${esc(b.oznaczenie || `Budynek ${i+1}`)}</div>`;
      [
        ['Szerokość elewacji',     'szerokosc_elewacji'],
        ['Pow. nadziemne',         'suma_pow_nadziemnych'],
        ['Pow. podziemne',         'suma_pow_podziemnych'],
        ['Wys. górnej krawędzi',   'wys_gornej_krawedzi'],
        ['Wysokość zabudowy',      'wysokosc_zabudowy'],
        ['Kond. nadziemne',        'ilosc_kond_nadziemnych'],
        ['Kond. podziemne',        'ilosc_kond_podziemnych'],
        ['Geometria dachu',        'geometria_dachu'],
      ].forEach(([label, key]) => {
        html += fieldHtml(label, b[key]);
      });
      html += '</div>';
    });
  }

  // Media
  const media = pred.media || [];
  if (media.length) {
    html += '<div class="section-title">Media</div>';
    media.forEach(m => { html += fieldHtml('', m); });
  }

  extractPanel.innerHTML = html;
}

function fieldHtml(label, value, cssClass = '') {
  const v = value !== undefined && value !== null ? String(value) : '';
  return `<div class="field-group">
    ${label ? `<div class="field-label">${esc(label)}</div>` : ''}
    <div class="field-value ${cssClass}">${esc(v) || '<span class="text-muted">—</span>'}</div>
  </div>`;
}

function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ============================================================
// Compare
// ============================================================
btnCompare.addEventListener('click', runCompare);

async function runCompare() {
  if (!lastPrediction) return;
  const nr = lastPrediction.nr_wniosku;
  if (!nr) { alert('Extraction result has no nr_wniosku — cannot compare.'); return; }

  showSpinner('Comparing…');
  try {
    const cmp = await api('POST', '/api/compare', {
      nr_wniosku: nr,
      prediction: lastPrediction,
    });
    renderComparison(cmp);
    // Switch to compare tab
    activateTab('compare');
  } catch (e) {
    alert('Compare failed: ' + e.message);
  } finally {
    hideSpinner();
  }
}

function renderComparison(cmp) {
  const pct = Math.round((cmp.overall_accuracy || 0) * 100);
  accuracyBarContainer.style.display = 'flex';
  accuracyFill.style.width = pct + '%';
  accuracyFill.style.background = pct >= 80 ? '#2d9a5f' : pct >= 50 ? '#f9a825' : '#e94560';
  accuracyLabel.textContent = 'Accuracy:';
  accuracyPct.textContent = pct + '%';

  let html = '';
  const flat = cmp.flat || {};
  Object.entries(flat).forEach(([key, v]) => {
    const cls = v.match ? 'match' : 'mismatch';
    html += `<div class="field-group">
      <div class="field-label">${esc(key)}</div>
      <div class="field-value ${cls}">
        <span style="color:#888;font-size:10px">PRED: </span>${esc(v.pred)}<br>
        <span style="color:#888;font-size:10px">GOLD: </span>${esc(v.gold)}
      </div>
    </div>`;
  });

  const budynki = cmp.budynki || [];
  budynki.forEach((bldCmp, i) => {
    html += `<div class="section-title">Budynek ${i + 1}</div>`;
    Object.entries(bldCmp).forEach(([key, v]) => {
      const cls = v.match ? 'match' : 'mismatch';
      html += `<div class="field-group">
        <div class="field-label">${esc(key)}</div>
        <div class="field-value ${cls}">
          <span style="color:#888;font-size:10px">PRED: </span>${esc(v.pred)}<br>
          <span style="color:#888;font-size:10px">GOLD: </span>${esc(v.gold)}
        </div>
      </div>`;
    });
  });

  const mediaCmp = cmp.media;
  if (mediaCmp) {
    const cls = mediaCmp.match ? 'match' : 'mismatch';
    html += `<div class="section-title">Media</div>
      <div class="field-group">
        <div class="field-label">media</div>
        <div class="field-value ${cls}">
          <span style="color:#888;font-size:10px">PRED: </span>${esc((mediaCmp.pred || []).join(', '))}<br>
          <span style="color:#888;font-size:10px">GOLD: </span>${esc((mediaCmp.gold || []).join(', '))}
        </div>
      </div>`;
  }

  comparePanel.innerHTML = html;
}

function clearCompare() {
  comparePanel.innerHTML = '<p class="text-muted" style="padding:8px">Run Extraction first, then click Compare.</p>';
  accuracyBarContainer.style.display = 'none';
}

// ============================================================
// Save
// ============================================================
btnSave.addEventListener('click', async () => {
  if (!lastPrediction) return;
  showSpinner('Saving…');
  try {
    const res = await api('POST', '/api/save', { prediction: lastPrediction });
    alert('Saved to: ' + res.path);
  } catch (e) {
    alert('Save failed: ' + e.message);
  } finally {
    hideSpinner();
  }
});

// ============================================================
// Chat
// ============================================================
btnSend.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  if (!currentDoc) { alert('Select a document first.'); return; }

  appendChat('user', msg);
  chatInput.value = '';

  showSpinner('Asking VLM…');
  try {
    const res = await api('POST', '/api/chat', {
      pdf_name: currentDoc.name,
      message: msg,
      max_pages: parseInt(maxPagesInput.value) || 6,
    });
    appendChat('assistant', res.response || JSON.stringify(res));
  } catch (e) {
    appendChat('error', 'Error: ' + e.message);
  } finally {
    hideSpinner();
  }
}

function appendChat(role, text) {
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => activateTab(tab.dataset.tab));
});

function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `tab-${name}`);
  });
}

// ============================================================
// Init
// ============================================================
(async () => {
  await Promise.all([loadDocuments(), checkVlmHealth()]);
  // Refresh VLM status every 30s
  setInterval(checkVlmHealth, 30_000);
})();
