// Editable JavaScript cells, shown next to the repository's Python (read-only). The cells call the
// JavaScript port (vd-core.js) that the optimizer on this page uses; the Python itself is not run here.
import { paint } from './targets.js?v=adfa0743';

// print() formatting: numbers as written, arrays (also typed and nested) as [a, b, c]
export function fmt(v) {
  if (typeof v === 'number') return Object.is(v, -0) ? '0' : String(v);
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return `[${Array.from(v, fmt).join(', ')}]`;
  if (v && typeof v === 'object') { try { return JSON.stringify(v); } catch { return String(v); } }
  return String(v);
}
// round(x, digits) for a number or an array of numbers (like np.round)
export function round(v, d = 0) {
  if (typeof v === 'number') { const p = 10 ** d; return Math.round(v * p) / p; }
  return Array.from(v, (x) => round(x, d));
}

// A cell: <div class="jscell"><textarea>code</textarea><button data-r=run><pre class="out"></div>
// scope() returns the variables the code can use (the figure's pixel, camera, heights, ...).
export class JsCell {
  constructor(root, scope) {
    this.root = root; this.scope = scope;
    this.code = root.querySelector('textarea'); this.out = root.querySelector('.out');
    this.img = root.querySelector('[data-r=img]');
    this.code.value = this.code.value.replace(/^\n/, '');
    this.original = this.code.value;
    root.querySelector('[data-r=run]').addEventListener('click', () => this.run());
    root.querySelector('[data-r=reset]').addEventListener('click', () => { this.code.value = this.original; autosize(this.code); this.run(); });
    this.code.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey || e.shiftKey) && e.key === 'Enter') { e.preventDefault(); this.run(); }
      if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); this.code.setRangeText('  ', this.code.selectionStart, this.code.selectionEnd, 'end'); }
    });
    this.code.addEventListener('input', () => autosize(this.code));
    autosize(this.code);
    new ResizeObserver(() => autosize(this.code)).observe(this.code);
  }
  run() {
    const vars = this.scope(), names = Object.keys(vars), lines = [];
    const print = (...a) => { lines.push(a.map(fmt).join(' ')); };
    const show = (image, n) => { if (this.img) { paint(this.img, image, n); this.img.parentElement.hidden = false; } };
    this.out.classList.remove('err');
    try {
      new Function(...names, 'print', 'show', `"use strict";\n${this.code.value}`)(...names.map((n) => vars[n]), print, show);
      this.out.textContent = lines.join('\n') || '(no output)';
    } catch (err) {
      this.out.textContent = [...lines, `${err.name}: ${err.message}`].join('\n');
      this.out.classList.add('err');
    }
  }
}

function autosize(t) { t.style.height = 'auto'; t.style.height = `${t.scrollHeight + 2}px`; }
