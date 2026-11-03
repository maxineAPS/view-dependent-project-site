import * as core from './vd-core.js?v=7071d1c4';
import { SliceFigure, SlicingFigure, HeavisidePlot, renderSlice } from './widgets.js?v=7071d1c4';
import { paint, solid, noiseImage, stripesBW } from './targets.js?v=7071d1c4';
import { JsCell, round } from './jscell.js?v=7071d1c4';
import { OptimizerPanel, SCENARIOS, workerUrl } from './panel.js?v=7071d1c4';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ---------------------------------------------------------------------------------------------
// Code walkthrough: functions extracted verbatim from the repository files
// ---------------------------------------------------------------------------------------------
const srcCache = {};
async function source(file) { return (srcCache[file] ??= fetch(`py/${file}?v=7071d1c4`).then((r) => r.text())); }
function extract(src, name) {
  const lines = src.split('\n');
  const i = lines.findIndex((l) => new RegExp(`^(\\s*)def ${name}\\(`).test(l));
  if (i < 0) return `# ${name} not found`;
  const indent = lines[i].match(/^\s*/)[0].length;
  let start = i;
  // include the '''docstring''' block written just above the def
  let j = i - 1; while (j >= 0 && !lines[j].trim()) j--;
  if (j >= 0 && lines[j].trim().endsWith("'''")) { let k = j - 1; while (k >= 0 && !lines[k].trim().startsWith("'''")) k--; if (k >= 0) start = k; }
  let end = i + 1;
  while (end < lines.length) {
    const l = lines[end];
    if (l.trim() && l.match(/^\s*/)[0].length <= indent && !l.trim().startsWith('#')) break;
    end++;
  }
  while (end > i && !lines[end - 1].trim()) end--;
  // drop a trailing docstring that belongs to the next function
  let body = lines.slice(start, end);
  const lastQuote = body.map((l) => l.trim()).lastIndexOf("'''");
  if (lastQuote > i - start) body = body.slice(0, body.slice(0, lastQuote).findLastIndex((l) => l.trim()) + 1);
  return body.join('\n');
}
function extractRange(src, from, to) {
  const lines = src.split('\n');
  const a = lines.findIndex((l) => l.includes(from)), b = to ? lines.findIndex((l, i) => i > a && l.includes(to)) : a;
  return lines.slice(a, (b < 0 ? a : b) + 1).join('\n');
}
const KW = /\b(def|return|for|in|if|elif|else|import|from|as|class|with|not|and|or|is|None|True|False|while|range|len|print|lambda|try|except|raise|global)\b/g;
function highlight(code) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const out = [];
  const re = /('''[\s\S]*?'''|"""[\s\S]*?"""|'[^'\n]*'|"[^"\n]*"|#[^\n]*)/g;
  let last = 0, m;
  while ((m = re.exec(code))) {
    out.push(esc(code.slice(last, m.index)).replace(KW, '<span class="kw">$1</span>').replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="num">$1</span>'));
    out.push(`<span class="${m[0].startsWith('#') ? 'cm' : m[0].startsWith("'''") || m[0].startsWith('"""') ? 'doc' : 'str'}">${esc(m[0])}</span>`);
    last = m.index + m[0].length;
  }
  out.push(esc(code.slice(last)).replace(KW, '<span class="kw">$1</span>').replace(/\b(\d+\.?\d*(e-?\d+)?)\b/g, '<span class="num">$1</span>'));
  return out.join('');
}
async function fillCode() {
  for (const pre of $$('pre.pysnip')) pre.innerHTML = `<code>${highlight(pre.textContent)}</code>`;
  for (const pre of $$('pre[data-src]')) {
    const src = await source(pre.dataset.src);
    let code;
    if (pre.dataset.fn) code = pre.dataset.fn.split(',').map((n) => extract(src, n.trim())).join('\n\n');
    else code = extractRange(src, pre.dataset.from, pre.dataset.to);
    pre.innerHTML = `<code>${highlight(code)}</code>`;
    const cap = pre.previousElementSibling;
    if (cap && cap.classList.contains('code-file')) cap.innerHTML = `<span>src/${pre.dataset.src}</span><span class="muted">verbatim from the repository</span>`;
  }
}

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
function initFigures() {
  const slicing = new SlicingFigure($('#fig-slicing'));
  const slice = new SliceFigure($('#fig-slice'));
  const heav = new HeavisidePlot($('#fig-heaviside'));
  [slicing, slice, heav].forEach((f) => f.draw());
  return { slicing, slice, heav };
}

// The code cells: the repository's Python (read-only) next to the same steps in the JavaScript port,
// which the visitor can edit and run. Cells that use the slicing figure rerun when it changes.
function initCells(figs) {
  let st = null;
  const scope = (uses) => () => {
    const pixel = [st.pixel[0], st.pixel[1], 1.0], cam = Array.from(st.cam);
    const H = Array.from({ length: 8 }, (_, a) => Array.from(st.H.slice(a * 8, a * 8 + 8)));
    const COL = Array.from({ length: 8 }, (_, a) => st.COL.slice(a * 8, a * 8 + 8).map((c) => c.slice()));
    const vars = { core, renderSlice, round, pixel, cam, H, COL };
    if (uses.includes('ray')) vars.ray = core.rayInfo(pixel.map((p, i) => p - 20 * cam[i]), cam, 8);
    return vars;
  };
  const cells = $$('.pyjs').map((el) => new JsCell($('.jscell', el), scope((el.dataset.uses || '').split(','))));
  let queued = false;
  figs.slicing.onChange = (s) => {
    st = s;
    if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; cells.forEach((c) => c.run()); }); }
  };
  figs.slicing.draw();
  return cells;
}

function initTable2() {
  const btn = $('#t2-run');
  if (!btn) return;
  const rows = $$('#table2 tbody tr');
  let worker = null, id = 0;
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  btn.addEventListener('click', () => {
    const cases = [[solid([0, 0, 0]), solid([1, 1, 1])], [noiseImage(11), noiseImage(12)], [solid([0, 0, 0]), noiseImage(13)], [solid([0, 0, 0]), stripesBW()], [noiseImage(14), stripesBW()]].map((images) => ({ images }));
    if (!worker) worker = new Worker(workerUrl, { type: 'module' });
    id += 1; const my = id;
    rows.forEach((r) => { r.querySelector('.live').textContent = '…'; });
    btn.disabled = true; btn.textContent = 'Running 15 optimizations…';
    const t0 = performance.now();
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.id !== my) return;
      if (m.done) { btn.disabled = false; btn.textContent = 'Run again'; $('#t2-time').textContent = `15 optimizations of 100 steps took ${((performance.now() - t0) / 1000).toFixed(1)} s in your browser.`; return; }
      const sd = Math.sqrt(mean(m.vals.map((v) => (v - mean(m.vals)) ** 2)));
      rows[m.row].querySelector('.live').innerHTML = `<b>${mean(m.vals).toFixed(3)}</b> <span class="muted">± ${sd.toFixed(3)} (${m.vals.length} run${m.vals.length > 1 ? 's' : ''})</span>`;
    };
    worker.postMessage({ type: 'table2', id: my, cases, seeds: [1, 2, 3] });
  });
}

function initNav() {
  const links = $$('.toc a');
  const secs = links.map((a) => $(a.getAttribute('href'))).filter(Boolean);
  const io = new IntersectionObserver((ents) => {
    for (const e of ents) if (e.isIntersecting) links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === `#${e.target.id}`));
  }, { rootMargin: '-40% 0px -55% 0px' });
  secs.forEach((s) => io.observe(s));
  $('#bibtex-copy')?.addEventListener('click', async (e) => {
    try { await navigator.clipboard.writeText($('#bibtex').textContent.trim()); e.target.textContent = 'Copied'; } catch { e.target.textContent = 'Select and copy'; }
    setTimeout(() => { e.target.textContent = 'Copy'; }, 1600);
  });
}

function main() {
  initNav();
  fillCode();
  const hero = new OptimizerPanel($('#hero-demo'), { views: SCENARIOS.fig1.views, pace: 25, autostart: true });
  const figs = initFigures();
  initCells(figs);
  const pg = new OptimizerPanel($('#playground'), { views: SCENARIOS.sunmoon.views, editable: true });
  window.__app = { hero, pg, figs, core };
}
main();
