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
let lastPrediction = null;
let lastComparison = null;
let schema         = null;
let editedKeys     = new Set();

// Batch state
let allDocuments   = [];       // [{name, stem}, ...]
let batchResults   = {};       // {filename: {status, prediction, error}}
let batchRunning   = false;

// ---- DOM helpers ----
const $  = id => document.getElementById(id);

// ============================================================
// Spinner
// ============================================================
const showSpinner = (msg = 'Working…') => { $('spinner-msg').textContent = msg; $('spinner').classList.add('active'); };
const hideSpinner = () => $('spinner').classList.remove('active');

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
    await loadDocuments();
    setTimeout(() => { $('cfg-status').textContent = ''; }, 3000);
  } catch (e) {
    $('cfg-status').textContent = '✗ ' + e.message;
    $('cfg-status').style.color = '#f4a3a3';
  }
});

// "Browse" button in settings opens directory modal targeting pdf_dir field
$('btn-browse-pdf').addEventListener('click', () => {
  openDirModal($('cfg-pdf-dir').value || '/', (chosenPath) => {
    $('cfg-pdf-dir').value = chosenPath;
  });
});

// ============================================================
// Directory Browser Modal
// ============================================================
let _modalCallback = null;
let _modalCurrentPath = '/';

function openDirModal(startPath, onSelect) {
  _modalCallback = onSelect;
  $('dir-modal').classList.remove('hidden');
  browseTo(startPath || '/');
}

function closeDirModal() {
  $('dir-modal').classList.add('hidden');
  _modalCallback = null;
}

$('btn-modal-close').addEventListener('click', closeDirModal);
$('btn-modal-cancel').addEventListener('click', closeDirModal);

$('btn-modal-select').addEventListener('click', () => {
  if (_modalCallback) _modalCallback(_modalCurrentPath);
  closeDirModal();
});

// Click outside modal to close
$('dir-modal').addEventListener('click', e => {
  if (e.target === $('dir-modal')) closeDirModal();
});

async function browseTo(path) {
  const body = $('modal-body');
  body.innerHTML = '<p class="text-muted">Loading…</p>';
  try {
    const data = await api('GET', `/api/browse?path=${encodeURIComponent(path)}`);
    _modalCurrentPath = data.path;

    // Breadcrumb
    const parts = data.path.split('/').filter(Boolean);
    let crumbHtml = '<span style="color:#e94560">/ </span>';
    let built = '';
    parts.forEach((p, i) => {
      built += '/' + p;
      const isLast = i === parts.length - 1;
      crumbHtml += isLast
        ? `<span>${esc(p)}</span>`
        : `<span class="crumb-link" data-path="${esc(built)}" style="cursor:pointer;color:#e94560">${esc(p)}</span> / `;
    });
    $('modal-breadcrumb').innerHTML = crumbHtml;
    $('modal-breadcrumb').querySelectorAll('.crumb-link').forEach(el => {
      el.addEventListener('click', () => browseTo(el.dataset.path));
    });

    // PDF count
    $('modal-pdf-count').textContent = data.pdf_count
      ? `${data.pdf_count} PDF${data.pdf_count !== 1 ? 's' : ''} in this folder`
      : 'No PDFs in this folder';

    // Directory list
    let html = '';
    if (data.parent) {
      html += `<div class="dir-item up" data-path="${esc(data.parent)}">
        <span class="dir-icon">⬆</span>
        <span class="dir-name">.. (up)</span>
      </div>`;
    }
    if (data.dirs.length === 0 && !data.parent) {
      html += '<p class="text-muted">No subdirectories found.</p>';
    }
    data.dirs.forEach(d => {
      html += `<div class="dir-item" data-path="${esc(d.path)}">
        <span class="dir-icon">📁</span>
        <span class="dir-name">${esc(d.name)}</span>
      </div>`;
    });
    body.innerHTML = html;
    body.querySelectorAll('.dir-item').forEach(el => {
      el.addEventListener('click', () => browseTo(el.dataset.path));
    });
  } catch (e) {
    body.innerHTML = `<p class="text-muted">Error: ${esc(e.message)}</p>`;
  }
}

// ============================================================
// VLM health
// ============================================================
async function checkVlmHealth() {
  try {
    const d = await api('GET', '/api/vlm/health');
    if (d.status === 'ok') {
      const adapter = d.adapter && d.adapter !== 'none (zero-shot)' ? ' +LoRA' : '';
      $('vlm-status').textContent = `VLM ✓ ${d.device}${adapter}`;
      $('vlm-status').className = 'ok';
    } else throw new Error();
  } catch {
    $('vlm-status').textContent = 'VLM ✗ offline';
    $('vlm-status').className = 'error';
  }
}

// ============================================================
// Documents
// ============================================================
async function loadDocuments() {
  const docs = await api('GET', '/api/documents').catch(() => []);
  allDocuments = Array.isArray(docs) ? docs : [];

  // Update dropdown
  const prev = $('doc-select').value;
  $('doc-select').innerHTML = '<option value="">— select document —</option>';
  allDocuments.forEach(d => {
    const o = document.createElement('option');
    o.value = d.name; o.textContent = d.name;
    if (d.name === prev) o.selected = true;
    $('doc-select').appendChild(o);
  });

  // Update batch list
  renderBatchList();
}

$('doc-select').addEventListener('change', () => {
  const name = $('doc-select').value;
  if (!name) return;
  currentDoc = { name, stem: name.replace(/\.pdf$/i, '') };
  lastPrediction = null; lastComparison = null; editedKeys.clear();
  setButtons(false);
  $('compare-mode-label').style.display = 'none';
  $('accuracy-bar').classList.add('hidden');
  $('grid-container').innerHTML = '<p class="text-muted">Document loaded. Click ⚡ Extract.</p>';
  loadPdf(name);
});

// ============================================================
// PDF viewer
// ============================================================
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
  const canvas = $('pdf-canvas');
  canvas.width = vp.width; canvas.height = vp.height;
  renderTask = page.render({ canvasContext: canvas.getContext('2d'), viewport: vp });
  await renderTask.promise.catch(e => { if (e.name !== 'RenderingCancelledException') throw e; });
  renderTask = null;
  $('page-info').textContent = `${n} / ${pdfDoc.numPages}`;
  $('prev-page').disabled = n <= 1;
  $('next-page').disabled = n >= pdfDoc.numPages;
}

$('prev-page').addEventListener('click', () => { if (currentPage > 1) renderPage(--currentPage); });
$('next-page').addEventListener('click', () => { if (pdfDoc && currentPage < pdfDoc.numPages) renderPage(++currentPage); });

// ============================================================
// Schema
// ============================================================
async function loadSchema() {
  schema = await api('GET', '/api/schema').catch(() => null);
}

// ============================================================
// Single extraction
// ============================================================
function setButtons(has) {
  $('btn-compare').disabled   = !has;
  $('btn-save').disabled      = !has;
  $('btn-reextract').disabled = !has;
}

$('btn-extract').addEventListener('click', runExtract);
$('btn-reextract').addEventListener('click', runExtract);

async function runExtract() {
  if (!currentDoc) { alert('Select a document first.'); return; }
  showSpinner('Extracting…');
  try {
    const pred = await api('POST', '/api/extract', {
      pdf_name: currentDoc.name,
      max_pages: parseInt($('max-pages').value) || 6,
    });
    lastPrediction = pred;
    lastComparison = null;
    editedKeys.clear();
    setButtons(true);
    $('accuracy-bar').classList.add('hidden');
    $('compare-mode-label').style.display = 'none';
    renderGrid();
  } catch (e) { alert('Extraction failed: ' + e.message); }
  finally { hideSpinner(); }
}

// ============================================================
// Compare
// ============================================================
$('btn-compare').addEventListener('click', runCompare);

async function runCompare() {
  if (!lastPrediction) return;
  const nr = lastPrediction.nr_wniosku;
  if (!nr) { alert('No nr_wniosku in extraction — cannot compare.'); return; }
  showSpinner('Comparing…');
  try {
    lastComparison = await api('POST', '/api/compare', { nr_wniosku: nr, prediction: lastPrediction });
    renderGrid();
    updateAccuracyBar(lastComparison);
    $('compare-mode-label').style.display = '';
  } catch (e) { alert('Compare failed: ' + e.message); }
  finally { hideSpinner(); }
}

$('chk-compare-mode').addEventListener('change', renderGrid);

function updateAccuracyBar(cmp) {
  const pct = Math.round((cmp.overall_accuracy || 0) * 100);
  let correct = 0, total = 0;
  Object.values(cmp.flat || {}).forEach(v => { correct += v.match ? 1 : 0; total++; });
  (cmp.budynki || []).forEach(b => Object.values(b).forEach(v => { correct += v.match ? 1 : 0; total++; }));
  if (cmp.media) { correct += cmp.media.match ? 1 : 0; total++; }
  $('acc-fill').style.width = pct + '%';
  $('acc-fill').style.background = pct >= 80 ? '#2d9a5f' : pct >= 50 ? '#f9a825' : '#e94560';
  $('acc-pct').textContent = pct + '%';
  $('acc-counts').textContent = `(${correct}/${total} fields)`;
  $('accuracy-bar').classList.remove('hidden');
}

// ============================================================
// Grid rendering
// ============================================================
function renderGrid() {
  if (!lastPrediction || !schema) return;
  const showGold = lastComparison && $('chk-compare-mode').checked;
  const cmpFlat  = lastComparison?.flat   || {};
  const cmpBlds  = lastComparison?.budynki || [];
  const cmpMedia = lastComparison?.media;

  let banner = '';
  if (lastPrediction.needs_review) {
    banner = '<div class="needs-review-banner">⚠ Model flagged needs_review = true — verify all fields</div>';
  }

  const colGroup = showGold
    ? `<colgroup><col style="width:130px"><col><col></colgroup>`
    : `<colgroup><col style="width:130px"><col></colgroup>`;
  const headers = showGold
    ? `<tr><th>Field</th><th>VLM Prediction (editable)</th><th>Ground Truth</th></tr>`
    : `<tr><th>Field</th><th>VLM Prediction (editable)</th></tr>`;

  let rows = '';
  rows += sectionRow('General', showGold);
  schema.flat.forEach(f => {
    const predVal = lastPrediction[f.key] ?? '';
    const cmp = cmpFlat[f.key];
    rows += fieldRow(f.label, f.key, null, predVal, cmp?.gold ?? null, cmp?.match, showGold);
  });

  const buildings = lastPrediction.budynki || [];
  buildings.forEach((bld, bi) => {
    rows += sectionRow(`Building ${bi + 1} — ${esc(bld.oznaczenie || '')}`, showGold);
    schema.building.forEach(f => {
      const predVal = bld[f.key] ?? '';
      const cmp = cmpBlds[bi]?.[f.key];
      rows += fieldRow(f.label, f.key, bi, predVal, cmp?.gold ?? null, cmp?.match, showGold);
    });
  });
  rows += `<tr><td colspan="${showGold ? 3 : 2}" style="padding:4px 8px">
    <button class="btn" style="font-size:11px" onclick="addBuilding()">+ Add building</button>
  </td></tr>`;

  rows += sectionRow('Media', showGold);
  (lastPrediction.media || []).forEach((m, mi) => {
    const goldVal = cmpMedia ? (cmpMedia.gold[mi] ?? null) : null;
    rows += fieldRow(`Media ${mi + 1}`, 'media', mi, m, goldVal, null, showGold);
  });
  rows += `<tr><td colspan="${showGold ? 3 : 2}" style="padding:4px 8px">
    <button class="btn" style="font-size:11px" onclick="addMedia()">+ Add media entry</button>
  </td></tr>`;

  $('grid-container').innerHTML = banner +
    `<table class="data-grid">${colGroup}<thead>${headers}</thead><tbody>${rows}</tbody></table>`;

  $('grid-container').querySelectorAll('textarea.cell-edit').forEach(el => {
    el.addEventListener('input', onCellEdit);
    el.addEventListener('input', autoResize);
    autoResize.call(el);
  });
}

function sectionRow(title, showGold) {
  return `<tr class="section-row"><td colspan="${showGold ? 3 : 2}">${esc(title)}</td></tr>`;
}

function fieldRow(label, key, buildingIdx, predVal, goldVal, match, showGold) {
  const dataKey = buildingIdx !== null
    ? (key === 'media' ? `media.${buildingIdx}` : `budynki.${buildingIdx}.${key}`)
    : key;
  const isEdited = editedKeys.has(dataKey);
  const matchCls = isEdited ? 'edited' : (match === true ? 'match' : match === false ? 'mismatch' : '');

  let goldCell = '';
  if (showGold) {
    const gv = goldVal !== null && goldVal !== undefined ? String(goldVal) : '';
    goldCell = `<td class="gold-cell"><span class="cell-gold${gv ? '' : ' empty'}">${esc(gv) || '—'}</span></td>`;
  }
  return `<tr>
    <td class="field-label-cell" title="${esc(label)}">${esc(label)}</td>
    <td class="pred-cell"><textarea class="cell-edit ${matchCls}" data-key="${esc(dataKey)}" rows="1">${esc(String(predVal))}</textarea></td>
    ${goldCell}
  </tr>`;
}

function autoResize() { this.style.height = 'auto'; this.style.height = this.scrollHeight + 'px'; }

function onCellEdit(e) {
  const el  = e.target;
  const key = el.dataset.key;
  editedKeys.add(key);
  el.classList.remove('match', 'mismatch');
  el.classList.add('edited');
  const parts = key.split('.');
  if (parts.length === 1) {
    lastPrediction[parts[0]] = el.value;
  } else if (parts[0] === 'budynki') {
    const bi = parseInt(parts[1]);
    if (!lastPrediction.budynki) lastPrediction.budynki = [];
    while (lastPrediction.budynki.length <= bi) lastPrediction.budynki.push({});
    lastPrediction.budynki[bi][parts[2]] = el.value;
  } else if (parts[0] === 'media') {
    const mi = parseInt(parts[1]);
    if (!lastPrediction.media) lastPrediction.media = [];
    while (lastPrediction.media.length <= mi) lastPrediction.media.push('');
    lastPrediction.media[mi] = el.value;
  }
}

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
// Save (single)
// ============================================================
$('btn-save').addEventListener('click', async () => {
  if (!lastPrediction) return;
  showSpinner('Saving…');
  try {
    const res = await api('POST', '/api/save', { prediction: lastPrediction });
    alert('Saved to:\n' + res.path);
  } catch (e) { alert('Save failed: ' + e.message); }
  finally { hideSpinner(); }
});

// ============================================================
// Batch tab
// ============================================================
function renderBatchList() {
  const container = $('batch-list-container');
  if (allDocuments.length === 0) {
    container.innerHTML = '<p class="text-muted">No PDFs found in the current directory.</p>';
    $('batch-sel-count').textContent = '';
    return;
  }

  let html = '';
  allDocuments.forEach(doc => {
    const result = batchResults[doc.name];
    const status = result ? result.status : 'pending';
    const statusText = { pending: '', running: '⏳ extracting…', done: '✓ done', error: '✗ error' }[status] || '';
    html += `<div class="batch-item" data-name="${esc(doc.name)}">
      <input type="checkbox" class="batch-chk" data-name="${esc(doc.name)}" ${status === 'done' ? '' : 'checked'}>
      <span class="batch-item-name" title="${esc(doc.name)}">${esc(doc.name)}</span>
      <span class="batch-item-status ${status}">${statusText}</span>
    </div>`;
  });
  container.innerHTML = html;

  // Row click toggles checkbox
  container.querySelectorAll('.batch-item').forEach(row => {
    row.addEventListener('click', e => {
      if (e.target.type === 'checkbox') return;
      const chk = row.querySelector('.batch-chk');
      chk.checked = !chk.checked;
      updateBatchSelCount();
    });
  });
  container.querySelectorAll('.batch-chk').forEach(chk => {
    chk.addEventListener('change', updateBatchSelCount);
  });
  updateBatchSelCount();
}

function updateBatchSelCount() {
  const total    = document.querySelectorAll('.batch-chk').length;
  const selected = document.querySelectorAll('.batch-chk:checked').length;
  $('batch-sel-count').textContent = `${selected} / ${total} selected`;
}

$('btn-select-all').addEventListener('click', () => {
  document.querySelectorAll('.batch-chk').forEach(c => c.checked = true);
  updateBatchSelCount();
});
$('btn-select-none').addEventListener('click', () => {
  document.querySelectorAll('.batch-chk').forEach(c => c.checked = false);
  updateBatchSelCount();
});

$('btn-batch-extract').addEventListener('click', runBatchExtract);

async function runBatchExtract() {
  if (batchRunning) return;
  const checked = [...document.querySelectorAll('.batch-chk:checked')].map(c => c.dataset.name);
  if (checked.length === 0) { alert('Select at least one document.'); return; }

  batchRunning = true;
  $('btn-batch-extract').disabled = true;
  $('btn-batch-save-all').disabled = true;

  const progress = $('batch-progress');
  progress.classList.remove('hidden');

  let done = 0;
  const mp = parseInt($('max-pages').value) || 6;

  for (const name of checked) {
    // Update status in list
    setBatchItemStatus(name, 'running', '⏳ extracting…');
    $('bp-label').textContent = name;
    $('bp-fill').style.width  = Math.round((done / checked.length) * 100) + '%';
    $('bp-pct').textContent   = `${done}/${checked.length}`;

    try {
      const pred = await api('POST', '/api/extract', { pdf_name: name, max_pages: mp });
      batchResults[name] = { status: 'done', prediction: pred };
      setBatchItemStatus(name, 'done', '✓ done');
    } catch (e) {
      batchResults[name] = { status: 'error', error: e.message };
      setBatchItemStatus(name, 'error', '✗ error');
    }
    done++;
  }

  $('bp-fill').style.width = '100%';
  $('bp-pct').textContent  = `${done}/${checked.length}`;
  $('bp-label').textContent = `Done — ${done} extracted`;

  batchRunning = false;
  $('btn-batch-extract').disabled = false;

  const anyDone = Object.values(batchResults).some(r => r.status === 'done');
  $('btn-batch-save-all').disabled = !anyDone;
}

function setBatchItemStatus(name, statusCls, text) {
  const row = $('batch-list-container').querySelector(`.batch-item[data-name="${CSS.escape(name)}"]`);
  if (!row) return;
  const span = row.querySelector('.batch-item-status');
  span.className = `batch-item-status ${statusCls}`;
  span.textContent = text;
}

$('btn-batch-save-all').addEventListener('click', async () => {
  const doneResults = Object.values(batchResults).filter(r => r.status === 'done');
  if (doneResults.length === 0) return;
  showSpinner(`Saving ${doneResults.length} results…`);
  let saved = 0, failed = 0;
  for (const r of doneResults) {
    try {
      await api('POST', '/api/save', { prediction: r.prediction });
      saved++;
    } catch { failed++; }
  }
  hideSpinner();
  alert(`Saved ${saved} records to extracted_results.xlsx${failed ? `\n${failed} failed` : ''}.`);
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
      max_pages: parseInt($('max-pages').value) || 6,
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
