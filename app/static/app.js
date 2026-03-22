/* ============================================================
   ARCH Document Extractor — Frontend Logic
   ============================================================ */

'use strict';

pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// ---- State ----
let currentDoc     = null;
let pdfDoc         = null;
let currentPage    = 1;
let renderTask     = null;
let lastPrediction = null;   // live prediction object (mutated on cell edits)
let lastComparison = null;   // last compare result from server
let schema         = null;   // {flat: [...], building: [...]}
let editedKeys     = new Set();  // tracks which keys were manually edited

// ---- DOM ----
const $  = id => document.getElementById(id);
const docSelect    = $('doc-select');
const prevBtn      = $('prev-page');
const nextBtn      = $('next-page');
const pageInfo     = $('page-info');
const pdfCanvas    = $('pdf-canvas');
const vlmStatus    = $('vlm-status');
const maxPages     = $('max-pages');
const spinner      = $('spinner');
const spinnerMsg   = $('spinner-msg');
const btnExtract   = $('btn-extract');
const btnCompare   = $('btn-compare');
const btnSave      = $('btn-save');
const btnReextract = $('btn-reextract');
const gridContainer = $('grid-container');
const accuracyBar  = $('accuracy-bar');
const accFill      = $('acc-fill');
const accPct       = $('acc-pct');
const accCounts    = $('acc-counts');
const compareModeLabel = $('compare-mode-label');
const chkCompareMode   = $('chk-compare-mode');

// ============================================================
// Spinner
// ============================================================
const showSpinner = (msg = 'Working…') => { spinnerMsg.textContent = msg; spinner.classList.add('active'); };
const hideSpinner = () => spinner.classList.remove('active');

// ============================================================
// API
// ============================================================
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(e.detail || r.statusText);
  }
  return r.json();
}

// ============================================================
// Settings drawer
// ============================================================
$('btn-settings').addEventListener('click', async () => {
  const drawer = $('settings-drawer');
  if (drawer.classList.toggle('hidden')) return;
  // Populate with current config
  const cfg = await api('GET', '/api/config').catch(() => ({}));
  $('cfg-pdf-dir').value    = cfg.pdf_dir    || '';
  $('cfg-labels-xlsx').value = cfg.labels_xlsx || '';
  $('cfg-vlm-url').value    = cfg.vlm_url    || '';
});

$('btn-save-config').addEventListener('click', async () => {
  const body = {
    pdf_dir:     $('cfg-pdf-dir').value.trim(),
    labels_xlsx: $('cfg-labels-xlsx').value.trim(),
    vlm_url:     $('cfg-vlm-url').value.trim(),
  };
  try {
    await api('POST', '/api/config', body);
    $('cfg-status').textContent = '✓ Applied';
    $('cfg-status').style.color = '#74c69d';
    // Reload document list with new dir
    await loadDocuments();
    setTimeout(() => { $('cfg-status').textContent = ''; }, 3000);
  } catch (e) {
    $('cfg-status').textContent = '✗ ' + e.message;
    $('cfg-status').style.color = '#f4a3a3';
  }
});

// ============================================================
// VLM health
// ============================================================
async function checkVlmHealth() {
  try {
    const d = await api('GET', '/api/vlm/health');
    if (d.status === 'ok') {
      const adapter = d.adapter && d.adapter !== 'none (zero-shot)' ? ' +LoRA' : '';
      vlmStatus.textContent = `VLM ✓ ${d.device}${adapter}`;
      vlmStatus.className = 'ok';
    } else throw new Error();
  } catch {
    vlmStatus.textContent = 'VLM ✗ offline';
    vlmStatus.className = 'error';
  }
}

// ============================================================
// Documents & PDF
// ============================================================
async function loadDocuments() {
  const docs = await api('GET', '/api/documents').catch(() => []);
  const prev = docSelect.value;
  docSelect.innerHTML = '<option value="">— select document —</option>';
  (Array.isArray(docs) ? docs : []).forEach(d => {
    const o = document.createElement('option');
    o.value = d.name; o.textContent = d.name;
    if (d.name === prev) o.selected = true;
    docSelect.appendChild(o);
  });
}

docSelect.addEventListener('change', () => {
  const name = docSelect.value;
  if (!name) return;
  currentDoc = { name, stem: name.replace(/\.pdf$/i, '') };
  lastPrediction = null; lastComparison = null; editedKeys.clear();
  setButtons(false);
  compareModeLabel.style.display = 'none';
  accuracyBar.classList.add('hidden');
  gridContainer.innerHTML = '<p class="text-muted">Document loaded. Click ⚡ Extract.</p>';
  loadPdf(name);
});

async function loadPdf(filename) {
  showSpinner('Loading PDF…');
  try {
    pdfDoc = await pdfjsLib.getDocument(`/api/pdf/${encodeURIComponent(filename)}`).promise;
    currentPage = 1;
    await renderPage(1);
  } catch (e) { alert('PDF load failed: ' + e.message); }
  finally { hideSpinner(); }
}

async function renderPage(n) {
  if (!pdfDoc) return;
  if (renderTask) { renderTask.cancel(); renderTask = null; }
  const page = await pdfDoc.getPage(n);
  const vp   = page.getViewport({ scale: 1.35 });
  pdfCanvas.width  = vp.width;
  pdfCanvas.height = vp.height;
  renderTask = page.render({ canvasContext: pdfCanvas.getContext('2d'), viewport: vp });
  await renderTask.promise.catch(e => { if (e.name !== 'RenderingCancelledException') throw e; });
  renderTask = null;
  pageInfo.textContent = `${n} / ${pdfDoc.numPages}`;
  prevBtn.disabled = n <= 1;
  nextBtn.disabled = n >= pdfDoc.numPages;
}

prevBtn.addEventListener('click', () => { if (currentPage > 1) renderPage(--currentPage); });
nextBtn.addEventListener('click', () => { if (pdfDoc && currentPage < pdfDoc.numPages) renderPage(++currentPage); });

// ============================================================
// Schema
// ============================================================
async function loadSchema() {
  schema = await api('GET', '/api/schema').catch(() => null);
}

// ============================================================
// Extraction
// ============================================================
function setButtons(has) {
  btnCompare.disabled   = !has;
  btnSave.disabled      = !has;
  btnReextract.disabled = !has;
}

btnExtract.addEventListener('click', runExtract);
btnReextract.addEventListener('click', runExtract);

async function runExtract() {
  if (!currentDoc) { alert('Select a document first.'); return; }
  showSpinner('Extracting…');
  try {
    const pred = await api('POST', '/api/extract', {
      pdf_name: currentDoc.name,
      max_pages: parseInt(maxPages.value) || 6,
    });
    lastPrediction = pred;
    lastComparison = null;
    editedKeys.clear();
    setButtons(true);
    accuracyBar.classList.add('hidden');
    compareModeLabel.style.display = 'none';
    renderGrid();
  } catch (e) { alert('Extraction failed: ' + e.message); }
  finally { hideSpinner(); }
}

// ============================================================
// Compare
// ============================================================
btnCompare.addEventListener('click', runCompare);

async function runCompare() {
  if (!lastPrediction) return;
  const nr = lastPrediction.nr_wniosku;
  if (!nr) { alert('No nr_wniosku in extraction — cannot compare.'); return; }
  showSpinner('Comparing…');
  try {
    lastComparison = await api('POST', '/api/compare', {
      nr_wniosku: nr,
      prediction: lastPrediction,
    });
    renderGrid();
    updateAccuracyBar(lastComparison);
    compareModeLabel.style.display = '';
  } catch (e) { alert('Compare failed: ' + e.message); }
  finally { hideSpinner(); }
}

chkCompareMode.addEventListener('change', renderGrid);

function updateAccuracyBar(cmp) {
  const pct = Math.round((cmp.overall_accuracy || 0) * 100);
  // count correct/total
  let correct = 0, total = 0;
  Object.values(cmp.flat || {}).forEach(v => { correct += v.match ? 1 : 0; total++; });
  (cmp.budynki || []).forEach(b => Object.values(b).forEach(v => { correct += v.match ? 1 : 0; total++; }));
  if (cmp.media) { correct += cmp.media.match ? 1 : 0; total++; }

  accFill.style.width = pct + '%';
  accFill.style.background = pct >= 80 ? '#2d9a5f' : pct >= 50 ? '#f9a825' : '#e94560';
  accPct.textContent = pct + '%';
  accCounts.textContent = `(${correct}/${total} fields)`;
  accuracyBar.classList.remove('hidden');
}

// ============================================================
// Grid rendering
// ============================================================
function renderGrid() {
  if (!lastPrediction) return;
  if (!schema) { gridContainer.innerHTML = '<p class="text-muted">Schema not loaded.</p>'; return; }

  const showGold = lastComparison && chkCompareMode.checked;
  const cmpFlat  = lastComparison?.flat  || {};
  const cmpBlds  = lastComparison?.budynki || [];
  const cmpMedia = lastComparison?.media;

  // Needs-review banner
  let banner = '';
  if (lastPrediction.needs_review) {
    banner = '<div class="needs-review-banner">⚠ Model flagged needs_review = true — verify all fields</div>';
  }

  // Build table
  const colGroup = showGold
    ? `<colgroup><col style="width:130px"><col><col></colgroup>`
    : `<colgroup><col style="width:130px"><col></colgroup>`;

  const headers = showGold
    ? `<tr><th>Field</th><th>VLM Prediction (editable)</th><th>Ground Truth</th></tr>`
    : `<tr><th>Field</th><th>VLM Prediction (editable)</th></tr>`;

  let rows = '';

  // --- Flat fields ---
  rows += sectionRow('General', showGold);
  schema.flat.forEach(f => {
    const predVal = lastPrediction[f.key] ?? '';
    const cmp     = cmpFlat[f.key];
    const goldVal = cmp ? cmp.gold : null;
    rows += fieldRow(f.label, f.key, null, predVal, goldVal, cmp?.match, showGold);
  });

  // --- Buildings ---
  const buildings = lastPrediction.budynki || [];
  buildings.forEach((bld, bi) => {
    rows += sectionRow(`Building ${bi + 1} — ${esc(bld.oznaczenie || '')}`, showGold);
    schema.building.forEach(f => {
      const predVal = bld[f.key] ?? '';
      const cmpB    = cmpBlds[bi];
      const cmp     = cmpB?.[f.key];
      const goldVal = cmp ? cmp.gold : null;
      rows += fieldRow(f.label, f.key, bi, predVal, goldVal, cmp?.match, showGold);
    });
  });
  // Button to add a building
  rows += `<tr><td colspan="${showGold ? 3 : 2}" style="padding:4px 8px">
    <button class="btn" style="font-size:11px" onclick="addBuilding()">+ Add building</button>
  </td></tr>`;

  // --- Media ---
  rows += sectionRow('Media', showGold);
  const media = lastPrediction.media || [];
  media.forEach((m, mi) => {
    const goldVal = cmpMedia ? (cmpMedia.gold[mi] ?? null) : null;
    rows += fieldRow(`Media ${mi + 1}`, 'media', mi, m, goldVal, null, showGold);
  });
  rows += `<tr><td colspan="${showGold ? 3 : 2}" style="padding:4px 8px">
    <button class="btn" style="font-size:11px" onclick="addMedia()">+ Add media entry</button>
  </td></tr>`;

  gridContainer.innerHTML = banner +
    `<table class="data-grid">${colGroup}<thead>${headers}</thead><tbody>${rows}</tbody></table>`;

  // Attach change listeners to all textareas
  gridContainer.querySelectorAll('textarea.cell-edit').forEach(el => {
    el.addEventListener('input', onCellEdit);
    el.addEventListener('keydown', autoResize);
    autoResize.call(el);
  });
}

function sectionRow(title, showGold) {
  const span = showGold ? 3 : 2;
  return `<tr class="section-row"><td colspan="${span}">${esc(title)}</td></tr>`;
}

function fieldRow(label, key, buildingIdx, predVal, goldVal, match, showGold) {
  const dataKey  = buildingIdx !== null
    ? (key === 'media' ? `media.${buildingIdx}` : `budynki.${buildingIdx}.${key}`)
    : key;
  const isEdited = editedKeys.has(dataKey);
  const matchCls = isEdited ? 'edited' : (match === true ? 'match' : match === false ? 'mismatch' : '');

  let goldCell = '';
  if (showGold) {
    const gv   = goldVal !== null && goldVal !== undefined ? String(goldVal) : '';
    const gcls = gv ? '' : ' empty';
    goldCell = `<td class="gold-cell"><span class="cell-gold${gcls}">${esc(gv) || '—'}</span></td>`;
  }

  return `<tr>
    <td class="field-label-cell" title="${esc(label)}">${esc(label)}</td>
    <td class="pred-cell"><textarea class="cell-edit ${matchCls}" data-key="${esc(dataKey)}" rows="1">${esc(String(predVal))}</textarea></td>
    ${goldCell}
  </tr>`;
}

function autoResize() {
  this.style.height = 'auto';
  this.style.height = this.scrollHeight + 'px';
}

function onCellEdit(e) {
  const el  = e.target;
  const key = el.dataset.key;
  const val = el.value;
  editedKeys.add(key);
  el.classList.remove('match', 'mismatch');
  el.classList.add('edited');

  // Write back into lastPrediction
  const parts = key.split('.');
  if (parts.length === 1) {
    lastPrediction[parts[0]] = val;
  } else if (parts[0] === 'budynki') {
    const bi = parseInt(parts[1]);
    if (!lastPrediction.budynki) lastPrediction.budynki = [];
    while (lastPrediction.budynki.length <= bi) lastPrediction.budynki.push({});
    lastPrediction.budynki[bi][parts[2]] = val;
  } else if (parts[0] === 'media') {
    const mi = parseInt(parts[1]);
    if (!lastPrediction.media) lastPrediction.media = [];
    while (lastPrediction.media.length <= mi) lastPrediction.media.push('');
    lastPrediction.media[mi] = val;
  }
}

// ============================================================
// Add rows
// ============================================================
function addBuilding() {
  if (!lastPrediction) return;
  if (!lastPrediction.budynki) lastPrediction.budynki = [];
  lastPrediction.budynki.push({ oznaczenie: `${lastPrediction.budynki.length + 1}. Nowy` });
  renderGrid();
}

function addMedia() {
  if (!lastPrediction) return;
  if (!lastPrediction.media) lastPrediction.media = [];
  lastPrediction.media.push('');
  renderGrid();
}

// ============================================================
// Save
// ============================================================
btnSave.addEventListener('click', async () => {
  if (!lastPrediction) return;
  showSpinner('Saving…');
  try {
    const res = await api('POST', '/api/save', { prediction: lastPrediction });
    alert('Saved to:\n' + res.path);
  } catch (e) { alert('Save failed: ' + e.message); }
  finally { hideSpinner(); }
});

// ============================================================
// Chat
// ============================================================
$('btn-send').addEventListener('click', sendChat);
$('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

async function sendChat() {
  const input = $('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  if (!currentDoc) { alert('Select a document first.'); return; }
  appendChat('user', msg);
  input.value = '';
  showSpinner('Asking VLM…');
  try {
    const res = await api('POST', '/api/chat', {
      pdf_name: currentDoc.name,
      message: msg,
      max_pages: parseInt(maxPages.value) || 6,
    });
    appendChat('assistant', res.response || JSON.stringify(res));
  } catch (e) { appendChat('error', 'Error: ' + e.message); }
  finally { hideSpinner(); }
}

function appendChat(role, text) {
  const msgs = $('chat-messages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

// ============================================================
// Tabs
// ============================================================
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
    document.querySelectorAll('.tab-panel').forEach(p =>
      p.classList.toggle('active', p.id === `tab-${tab.dataset.tab}`));
  });
});

// ============================================================
// Helpers
// ============================================================
function esc(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ============================================================
// Init
// ============================================================
(async () => {
  await Promise.all([loadDocuments(), checkVlmHealth(), loadSchema()]);
  setInterval(checkVlmHealth, 30_000);
})();
