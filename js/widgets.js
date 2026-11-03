// Interactive figures: the 2D slice renderer (Sec. 3.3), heightfield slicing (Sec. 3.2), Heaviside
// approximations (Sec. 4.1) and a live loss chart. All values are computed with the same formulas as
// rendering.py (via vd-core.js where possible).
import * as core from './vd-core.js?v=6a93b9b1';

const f32 = Math.fround;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const rgbCss = (c) => `rgb(${c.map((v) => Math.round(Math.max(0, Math.min(1, v)) * 255)).join(',')})`;
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');

// Redraw when an element's width changes. Deferred to the next frame: the redraw changes the figure's own
// height, and doing that inside the observer callback makes WebKit report a ResizeObserver loop.
function onWidth(el, fn) {
  let w = -1, queued = false;
  new ResizeObserver((ents) => {
    const nw = ents[ents.length - 1].contentRect.width;
    if (nw === w) return;
    w = nw;
    if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; fn(); }); }
  }).observe(el);
}

function setupCanvas(cv, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = `${w}px`; cv.style.height = `${h}px`;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ---------------------------------------------------------------------------------------------
// The 1D slice renderer (rendering.py: h_to_boundary, monotonize_boundary_points,
// render_ray_given_boundaries), used by both slice figures.
// ---------------------------------------------------------------------------------------------
export function renderSlice(heights, colors, widthSums, slope, oy, hv = 'tan', k = 0.1) {
  // slope = d.y / d.x of the 2D ray (negative: the ray goes down as x grows)
  const n = heights.length;
  const y = heights.map((h, i) => f32(f32(h) - f32(slope * widthSums[i])));
  const mono = [], arg = [];
  let best = -Infinity, bi = 0;
  y.forEach((v, i) => { if (v >= best) { best = v; bi = i; } mono.push(best); arg.push(bi); });
  const g = mono.map((v) => (oy - v > 0 ? 1 : 0));
  let col = colors[0].slice();
  for (let i = 0; i < n - 1; i++) if (g[i]) col = col.map((c, ch) => c + (colors[i + 1][ch] - colors[i][ch]));
  let hit = n - 1;
  for (let i = 0; i < n - 1; i++) if (!g[i]) { hit = i; break; }
  const dsur = mono.map((v) => (core.HEAVISIDE[hv] || core.HEAVISIDE.tan).bwd(oy - v, k));
  return { y, mono, arg, g, color: col, hit, dsur };
}

// Draws a slice (bars, back-traced boundaries, the ray) into a canvas region.
function drawSlice(ctx, box, s, opts = {}) {
  const { heights, colors, widthSums, slope, oy, res } = s;
  const n = heights.length, W = widthSums[n - 1];
  const yMax = Math.max(opts.yMax || 0, ...res.y, oy, ...heights) * 1.08 + 0.05;
  const pad = { l: 46, r: 12, t: 12, b: 22 };
  const X = (x) => box.x + pad.l + (x / W) * (box.w - pad.l - pad.r);
  const Y = (y) => box.y + box.h - pad.b - (y / yMax) * (box.h - pad.t - pad.b);
  const ink = css('--ink') || '#222', muted = css('--muted') || '#888';
  // ground + axis (x = 0 is where the projected ray enters the heightfield)
  ctx.strokeStyle = ink; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(X(0) - 6, Y(0)); ctx.lineTo(X(W) + 6, Y(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(0), Y(yMax)); ctx.stroke();
  ctx.fillStyle = muted; ctx.font = '11px "Noto Sans", sans-serif'; ctx.textAlign = 'right';
  ctx.fillText('x = 0', X(0) - 5, Y(0) + 15); ctx.textAlign = 'center';
  // bars
  const hidden = res.y.map((v, i) => i > 0 && v < res.mono[i] && res.arg[i] !== i);
  for (let i = 0; i < n; i++) {
    const x0 = i ? widthSums[i - 1] : 0, x1 = widthSums[i];
    ctx.globalAlpha = hidden[i] ? 0.35 : 1;
    ctx.fillStyle = rgbCss(colors[i]);
    ctx.fillRect(X(x0), Y(heights[i]), X(x1) - X(x0), Y(0) - Y(heights[i]));
    ctx.globalAlpha = 1;
    ctx.strokeStyle = opts.selected === i ? '#000' : 'rgba(0,0,0,0.55)'; ctx.lineWidth = opts.selected === i ? 2 : 1;
    ctx.strokeRect(X(x0), Y(heights[i]), X(x1) - X(x0), Y(0) - Y(heights[i]));
    if (opts.labels !== false) { ctx.fillStyle = muted; ctx.font = '10px "Noto Sans", sans-serif'; ctx.fillText(String(i), (X(x0) + X(x1)) / 2, Y(0) + 15); }
  }
  // back-traced boundaries y_i (Eq. 1): from the far top corner of each bar back to x = 0
  if (opts.boundaries !== false) {
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      ctx.strokeStyle = hidden[i] ? 'rgba(232,89,12,0.35)' : 'rgba(232,89,12,0.9)';
      ctx.beginPath(); ctx.moveTo(X(widthSums[i]), Y(heights[i])); ctx.lineTo(X(0), Y(res.y[i])); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.textAlign = 'right'; ctx.font = '10px "Noto Sans", sans-serif';
    let lastLabelY = Infinity;
    const order = res.y.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    for (const [v, i] of order) {
      const py = Y(v);
      ctx.fillStyle = hidden[i] ? 'rgba(232,89,12,0.45)' : '#e8590c';
      ctx.beginPath(); ctx.arc(X(0), py, 2.5, 0, Math.PI * 2); ctx.fill();
      if (Math.abs(py - lastLabelY) > 11) { ctx.fillText(`y${sub(i)}`, X(0) - 6, py + 3); lastLabelY = py; }
    }
  }
  // the ray(s)
  const rays = opts.rays || [oy];
  for (const o of rays) {
    const r = o === oy ? res : renderSlice(heights, colors, widthSums, slope, o, opts.hv, opts.k);
    const k = r.hit;
    const xL = k ? widthSums[k - 1] : 0;
    let xe, ye;
    const zL = o + slope * xL;
    if (zL <= heights[k]) { xe = xL; ye = zL; } else { xe = (heights[k] - o) / slope; ye = heights[k]; if (xe > widthSums[k]) { xe = widthSums[k]; ye = o + slope * xe; } }
    const main = o === oy;
    ctx.strokeStyle = main ? '#111' : rgbCss(r.color); ctx.lineWidth = main ? 2.2 : 1.4;
    ctx.globalAlpha = main ? 1 : 0.9;
    ctx.beginPath(); ctx.moveTo(X(0), Y(o)); ctx.lineTo(X(xe), Y(ye)); ctx.stroke();
    ctx.globalAlpha = 1;
    if (main) {
      // arrow head + hit marker
      const ang = Math.atan2(Y(ye) - Y(o), X(xe) - X(0));
      ctx.fillStyle = '#111'; ctx.beginPath();
      ctx.moveTo(X(xe), Y(ye)); ctx.lineTo(X(xe) - 9 * Math.cos(ang - 0.4), Y(ye) - 9 * Math.sin(ang - 0.4)); ctx.lineTo(X(xe) - 9 * Math.cos(ang + 0.4), Y(ye) - 9 * Math.sin(ang + 0.4)); ctx.fill();
      ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(X(0), Y(o), 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(0), Y(o), 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = ink; ctx.textAlign = 'left'; ctx.font = 'bold 11px "Noto Sans", sans-serif';
      ctx.fillText('o_y', X(0) + 7, Y(o) - 6);
    }
  }
  // gradient arrows (direction the height would move to bring the pixel closer to the target)
  if (opts.grads) {
    const m = Math.max(1e-12, ...opts.grads.map(Math.abs));
    opts.grads.forEach((gv, i) => {
      if (Math.abs(gv) < 1e-9 * m) return;
      const x0 = i ? widthSums[i - 1] : 0, xc = X((x0 + widthSums[i]) / 2), y0 = Y(heights[i]) - 6;
      const L = 8 + 30 * Math.abs(gv) / m, dir = gv > 0 ? -1 : 1; // move against the gradient
      ctx.strokeStyle = '#7048e8'; ctx.fillStyle = '#7048e8'; ctx.lineWidth = 2.2;
      const yEnd = dir > 0 ? y0 - L : y0 - 2;
      const yStart = dir > 0 ? y0 : y0 - L;
      ctx.beginPath(); ctx.moveTo(xc, yStart); ctx.lineTo(xc, yEnd); ctx.stroke();
      const tip = dir > 0 ? yEnd : yEnd;
      ctx.beginPath(); ctx.moveTo(xc, tip + (dir > 0 ? -5 : 5)); ctx.lineTo(xc - 5, tip + (dir > 0 ? 3 : -3)); ctx.lineTo(xc + 5, tip + (dir > 0 ? 3 : -3)); ctx.fill();
    });
  }
  return { X, Y, yMax, pad };
}
const SUBS = '₀₁₂₃₄₅₆₇₈₉';
const sub = (i) => String(i).split('').map((d) => SUBS[+d]).join('');

// ---------------------------------------------------------------------------------------------
// Figure: rendering a single pixel (drag the bars, the ray origin; flip the viewing side)
// ---------------------------------------------------------------------------------------------
const PALETTE = [[0.92, 0.33, 0.27], [0.98, 0.78, 0.25], [0.22, 0.62, 0.36], [0.24, 0.45, 0.85], [0.55, 0.36, 0.76], [0.95, 0.95, 0.95], [0.12, 0.12, 0.14]];

export class SliceFigure {
  constructor(root) {
    this.root = root;
    this.cv = root.querySelector('canvas');
    this.heights = [0.6, 1.15, 0.55, 1.7, 1.25, 2.2];
    this.colors = [[0.73, 0.79, 0.9], [0.95, 0.8, 0.68], [0.31, 0.68, 0.36], [0.4, 0.21, 0.6], [0.98, 0.78, 0.25], [0.92, 0.33, 0.27]];
    this.elev = 33; this.oy = 2.4; this.flip = false; this.many = false; this.showGrad = false; this.target = 2;
    this.hv = 'tan'; this.k = 0.1;
    const q = (s) => root.querySelector(s);
    this.ui = { elev: q('[data-r=elev]'), elevOut: q('[data-r=elev-out]'), flip: q('[data-r=flip]'), many: q('[data-r=many]'), grad: q('[data-r=grad]'), step: q('[data-r=step]'),
      table: q('[data-r=table]'), swatch: q('[data-r=swatch]'), strip: q('[data-r=strip]'), stripB: q('[data-r=strip-b]'), targets: q('[data-r=targets]'), formula: q('[data-r=formula]') };
    this.ui.elev.addEventListener('input', () => { this.elev = +this.ui.elev.value; this.draw(); });
    this.ui.flip.addEventListener('click', () => { this.flip = !this.flip; this.ui.flip.classList.toggle('on', this.flip); this.draw(); });
    this.ui.many.addEventListener('click', () => { this.many = !this.many; this.ui.many.classList.toggle('on', this.many); this.draw(); });
    this.ui.grad.addEventListener('click', () => { this.showGrad = !this.showGrad; this.ui.grad.classList.toggle('on', this.showGrad); this.ui.step.disabled = !this.showGrad; this.ui.targets.hidden = !this.showGrad; this.draw(); });
    this.ui.step.addEventListener('click', () => this.gradStep());
    this.ui.targets.innerHTML = PALETTE.map((c, i) => `<button class="swatch-btn" data-i="${i}" style="background:${rgbCss(c)}" aria-label="target colour ${i}"></button>`).join('');
    this.ui.targets.addEventListener('click', (e) => { const b = e.target.closest('[data-i]'); if (!b) return; this.target = +b.dataset.i; this.draw(); });
    this.bindPointer();
    onWidth(root.querySelector('.fig-canvas-wrap'), () => this.draw());
  }
  state() {
    const order = this.flip ? [...this.heights.keys()].reverse() : [...this.heights.keys()];
    const heights = order.map((i) => this.heights[i]), colors = order.map((i) => this.colors[i]);
    const widthSums = heights.map((_, i) => i + 1);
    const slope = -Math.tan((this.elev * Math.PI) / 180);
    return { order, heights, colors, widthSums, slope, oy: this.oy, res: renderSlice(heights, colors, widthSums, slope, this.oy, this.hv, this.k) };
  }
  grads(s) {
    // L = |color - target|^2 ; dL/dh via the surrogate derivative of the step (backward of happrox.HeaviTanh)
    const t = PALETTE[this.target];
    const dL = s.res.color.map((c, i) => 2 * (c - t[i]));
    const g = new Array(s.heights.length).fill(0);
    for (let i = 0; i < s.heights.length - 1; i++) {
      const dc = s.colors[i + 1].reduce((acc, c, ch) => acc + dL[ch] * (c - s.colors[i][ch]), 0);
      g[s.res.arg[i]] += -dc * s.res.dsur[i]; // d(color)/d(y'_i) = -g'(o - y'_i) (c_{i+1} - c_i); dy/dh = 1
    }
    return g;
  }
  gradStep() {
    const s = this.state();
    const g = this.grads(s);
    const m = Math.max(1e-12, ...g.map(Math.abs));
    g.forEach((gv, j) => { const i = s.order[j]; this.heights[i] = Math.max(0.15, Math.min(3.2, this.heights[i] - 0.25 * gv / m)); });
    this.draw();
  }
  bindPointer() {
    let drag = null;
    const pick = (e) => {
      const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const s = this.state(), m = this.metrics;
      if (!m) return null;
      if (Math.abs(x - m.X(0)) < 14 && Math.abs(y - m.Y(s.oy)) < 14) return { kind: 'oy' };
      for (let j = 0; j < s.heights.length; j++) {
        const x0 = j ? s.widthSums[j - 1] : 0;
        if (x > m.X(x0) && x < m.X(s.widthSums[j])) {
          if (Math.abs(y - m.Y(s.heights[j])) < 16 || (y > m.Y(s.heights[j]) && y < m.Y(0))) return { kind: 'bar', j, i: s.order[j], top: Math.abs(y - m.Y(s.heights[j])) < 16 };
        }
      }
      if (Math.abs(x - m.X(0)) < 26) return { kind: 'oy' };
      return null;
    };
    const toWorldY = (e) => { const r = this.cv.getBoundingClientRect(), m = this.metrics; return ((m.Y(0) - (e.clientY - r.top)) / (m.Y(0) - m.Y(1))); };
    this.cv.addEventListener('pointerdown', (e) => {
      const p = pick(e);
      if (!p) return;
      e.preventDefault();
      if (p.kind === 'bar' && !p.top) { this.colors[p.i] = PALETTE[(PALETTE.findIndex((c) => c.join() === this.colors[p.i].join()) + 1) % PALETTE.length]; this.draw(); return; }
      drag = p; this.cv.setPointerCapture(e.pointerId);
      this.frozenMax = this.metrics.yMax;
    });
    this.cv.addEventListener('pointermove', (e) => {
      if (!drag) { const p = pick(e); this.cv.style.cursor = p ? (p.kind === 'oy' || p.top ? 'ns-resize' : 'pointer') : 'default'; return; }
      const v = toWorldY(e);
      if (drag.kind === 'oy') this.oy = Math.max(0.05, Math.min(this.frozenMax * 0.97, v));
      else this.heights[drag.i] = Math.max(0.15, Math.min(3.2, v));
      this.draw();
    });
    const end = () => { drag = null; this.frozenMax = 0; };
    this.cv.addEventListener('pointerup', end); this.cv.addEventListener('pointercancel', end);
  }
  draw() {
    const wrap = this.root.querySelector('.fig-canvas-wrap');
    const w = wrap.clientWidth, h = Math.max(260, Math.min(380, w * 0.55));
    const ctx = setupCanvas(this.cv, w, h);
    ctx.clearRect(0, 0, w, h);
    const s = this.state();
    this.ui.elevOut.textContent = `${this.elev}°`;
    const maxY = Math.max(...this.heights) + Math.tan((this.elev * Math.PI) / 180) * this.heights.length;
    const rays = this.many ? Array.from({ length: 28 }, (_, i) => 0.1 + (i + 0.5) * (maxY / 28)) : null;
    this.metrics = drawSlice(ctx, { x: 0, y: 0, w, h }, s, { yMax: this.frozenMax || maxY * 0.92, rays: rays ? [...rays, s.oy] : null, grads: this.showGrad ? this.grads(s) : null, hv: this.hv, k: this.k });
    // table (Eq. 1, monotonic, Eq. 6)
    const r = s.res;
    const rows = s.heights.map((hh, j) => `<tr class="${j === r.hit ? 'hit' : ''}${r.y[j] < r.mono[j] && r.arg[j] !== j ? ' hidden-row' : ''}">
      <td>${j}</td><td><span class="chip" style="background:${rgbCss(s.colors[j])}"></span></td><td>${fmt(hh)}</td><td>${fmt(s.widthSums[j], 0)}</td><td>${fmt(r.y[j])}</td><td>${fmt(r.mono[j])}</td><td>${j < s.heights.length - 1 ? r.g[j] : '–'}</td></tr>`).join('');
    this.ui.table.innerHTML = `<thead><tr><th>i</th><th>c<sub>i</sub></th><th>h<sub>i</sub></th><th>W<sub>i</sub></th><th>y<sub>i</sub> = h<sub>i</sub> + W<sub>i</sub>·tan&thinsp;θ</th><th>max(y<sub>0..i</sub>)</th><th>g(o<sub>y</sub> − y'<sub>i</sub>)</th></tr></thead><tbody>${rows}</tbody>`;
    this.ui.swatch.style.background = rgbCss(r.color);
    this.ui.formula.innerHTML = `o<sub>y</sub> = ${fmt(s.oy)} &nbsp;→&nbsp; color = c<sub>0</sub> + Σ g(o<sub>y</sub> − y'<sub>i</sub>)(c<sub>i+1</sub> − c<sub>i</sub>) = colour of bar <b>${r.hit}</b>`;
    // the 1D image of this slice seen from this side and from the other side
    this.paintStrip(this.ui.strip, s);
    const o = this.flip; this.flip = !o; const s2 = this.state(); this.flip = o;
    this.paintStrip(this.ui.stripB, s2);
    [...this.ui.targets.children].forEach((b, i) => b.classList.toggle('on', i === this.target));
  }
  paintStrip(cv, s) {
    const n = 48, ctx = setupCanvas(cv, 22, 150);
    const maxY = Math.max(...this.heights) + Math.tan((this.elev * Math.PI) / 180) * this.heights.length;
    for (let i = 0; i < n; i++) {
      const o = maxY * (1 - (i + 0.5) / n);
      ctx.fillStyle = rgbCss(renderSlice(s.heights, s.colors, s.widthSums, s.slope, o).color);
      ctx.fillRect(0, (i * 150) / n, 22, 150 / n + 0.5);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Figure: slicing a 3D heightfield with a projected ray (find_intersections / get_2D_ray_info)
// ---------------------------------------------------------------------------------------------
export class SlicingFigure {
  constructor(root) {
    this.root = root;
    this.top = root.querySelector('[data-r=top]');
    this.side = root.querySelector('[data-r=side]');
    this.S = 8;
    const rnd = core.mulberry32(5);
    this.H = Array.from({ length: 64 }, () => f32(0.25 + 1.1 * rnd()));
    const pal = [[0.93, 0.42, 0.36], [0.98, 0.8, 0.35], [0.4, 0.7, 0.45], [0.36, 0.55, 0.88], [0.62, 0.45, 0.8], [0.9, 0.9, 0.9], [0.3, 0.3, 0.33]];
    this.COL = Array.from({ length: 64 }, () => pal[Math.floor(rnd() * pal.length)]);
    this.pixel = [0.42, 0.57]; this.az = 18; this.elev = 40;
    this.ui = { elev: root.querySelector('[data-r=elev]'), elevOut: root.querySelector('[data-r=elev-out]'), info: root.querySelector('[data-r=info]'), swatch: root.querySelector('[data-r=swatch]') };
    this.ui.elev.addEventListener('input', () => { this.elev = +this.ui.elev.value; this.draw(); });
    this.bind();
    onWidth(root, () => this.draw());
  }
  cam() { return core.camFromAzEl(this.az, this.elev).map((v, i) => v + [0.0013, 0.0021, 0][i]); }
  compute() {
    const d = this.cam();
    const origin = [this.pixel[0] - 20 * d[0], this.pixel[1] - 20 * d[1], 1 - 20 * d[2]];
    const info = core.rayInfo(origin, d, this.S);
    const heights = info.ids.map(([a, b]) => this.H[a * this.S + b]);
    const colors = info.ids.map(([a, b]) => this.COL[a * this.S + b]);
    const slope = info.d2y / info.d2x;
    const res = renderSlice(heights, colors, info.widthSums, slope, f32(info.oy));
    return { d, origin, info, heights, colors, slope, res };
  }
  bind() {
    let drag = null;
    const toUnit = (e) => { const r = this.top.getBoundingClientRect(), m = this.tm; return [((e.clientY - r.top) - m.oy) / m.size, ((e.clientX - r.left) - m.ox) / m.size]; };
    this.top.addEventListener('pointerdown', (e) => {
      const [x, y] = toUnit(e);
      const knob = this.knobPos();
      const r = this.top.getBoundingClientRect();
      if (Math.hypot(e.clientX - r.left - knob[0], e.clientY - r.top - knob[1]) < 18) drag = 'az';
      else if (x >= -0.05 && x <= 1.05 && y >= -0.05 && y <= 1.05) { drag = 'px'; this.pixel = [Math.min(0.995, Math.max(0.005, x)), Math.min(0.995, Math.max(0.005, y))]; this.draw(); }
      else drag = 'az';
      if (drag) { this.top.setPointerCapture(e.pointerId); e.preventDefault(); }
    });
    this.top.addEventListener('pointermove', (e) => {
      if (!drag) return;
      if (drag === 'px') { const [x, y] = toUnit(e); this.pixel = [Math.min(0.995, Math.max(0.005, x)), Math.min(0.995, Math.max(0.005, y))]; }
      else { const r = this.top.getBoundingClientRect(), m = this.tm; const cx = m.ox + m.size / 2, cy = m.oy + m.size / 2; const vx = e.clientX - r.left - cx, vy = e.clientY - r.top - cy; this.az = (Math.atan2(vx, vy) * 180) / Math.PI + 180; }
      this.draw();
    });
    const end = () => { drag = null; };
    this.top.addEventListener('pointerup', end); this.top.addEventListener('pointercancel', end);
  }
  knobPos() { // camera handle on a circle around the square: it sits where the camera is (opposite to the ray direction)
    const m = this.tm, d = this.cam(), L = Math.hypot(d[0], d[1]);
    const cx = m.ox + m.size / 2, cy = m.oy + m.size / 2, R = m.size * 0.5 + 15;
    return [cx - (d[1] / L) * R, cy - (d[0] / L) * R];
  }
  draw() {
    const wTot = this.root.querySelector('.slicing-grid').clientWidth;
    const narrow = wTot < 640;
    const topW = narrow ? wTot : Math.min(360, wTot * 0.42);
    const ctx = setupCanvas(this.top, topW, topW);
    const size = topW - 60;
    this.tm = { ox: 30, oy: 30, size };
    const { info, res, heights, colors, slope, d } = this.compute();
    this.ui.elevOut.textContent = `${this.elev}°`;
    ctx.clearRect(0, 0, topW, topW);
    const S = this.S, c = size / S;
    // top view: x runs down (rows), y runs right (columns), like the image/optimizer layout
    for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
      ctx.fillStyle = rgbCss(this.COL[a * S + b]); ctx.fillRect(30 + b * c, 30 + a * c, c, c);
      ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.font = `${Math.max(8, c * 0.26)}px "Noto Sans", sans-serif`; ctx.textAlign = 'center';
      ctx.fillText(this.H[a * S + b].toFixed(1), 30 + (b + 0.5) * c, 30 + (a + 0.62) * c);
    }
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1;
    for (let i = 0; i <= S; i++) { ctx.beginPath(); ctx.moveTo(30 + i * c, 30); ctx.lineTo(30 + i * c, 30 + size); ctx.stroke(); ctx.beginPath(); ctx.moveTo(30, 30 + i * c); ctx.lineTo(30 + size, 30 + i * c); ctx.stroke(); }
    // crossed cells, in order
    info.ids.forEach(([a, b], i) => {
      ctx.strokeStyle = i === res.hit ? '#000' : '#e8590c'; ctx.lineWidth = i === res.hit ? 3 : 2;
      ctx.strokeRect(30 + b * c + 1.5, 30 + a * c + 1.5, c - 3, c - 3);
      ctx.fillStyle = '#e8590c'; ctx.font = `bold ${Math.max(8, c * 0.24)}px "Noto Sans", sans-serif`; ctx.textAlign = 'left';
      ctx.fillText(String(i), 30 + b * c + 3, 30 + a * c + Math.max(9, c * 0.27));
    });
    // projected ray line through the pixel, intersections
    const P = (x, y) => [30 + y * size, 30 + x * size];
    const pts = info.pts;
    ctx.strokeStyle = '#111'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(...P(pts[0][0], pts[0][1])); ctx.lineTo(...P(pts[pts.length - 1][0], pts[pts.length - 1][1])); ctx.stroke();
    ctx.fillStyle = '#111';
    for (const p of pts) { ctx.beginPath(); ctx.arc(...P(p[0], p[1]), 2.6, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = '#fff'; ctx.strokeStyle = '#111'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(...P(this.pixel[0], this.pixel[1]), 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // camera knob
    const k = this.knobPos();
    ctx.fillStyle = '#e8590c'; ctx.beginPath(); ctx.arc(k[0], k[1], 9, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = 'bold 10px "Noto Sans", sans-serif'; ctx.textAlign = 'center'; ctx.fillText('cam', k[0], k[1] + 3.5);
    ctx.fillStyle = css('--muted') || '#888'; ctx.font = '11px "Noto Sans", sans-serif'; ctx.textAlign = 'left';
    ctx.fillText('x ↓  y →', 2, 14);
    // side view of the slice
    const sideW = narrow ? wTot : wTot - topW - 24, sideH = narrow ? 240 : topW;
    const sctx = setupCanvas(this.side, sideW, sideH);
    sctx.clearRect(0, 0, sideW, sideH);
    drawSlice(sctx, { x: 0, y: 0, w: sideW, h: sideH }, { heights, colors, widthSums: info.widthSums, slope, oy: f32(info.oy), res }, { labels: true });
    this.ui.swatch.style.background = rgbCss(res.color);
    this.ui.info.innerHTML = `<span>camera direction (${d.map((v) => v.toFixed(2)).join(', ')}) · pixel at (${this.pixel.map((v) => v.toFixed(2)).join(', ')}, 1) · ${info.ids.length} strips crossed · widths ${info.widths.map((w) => w.toFixed(3)).join(', ')} · o<sub>y</sub> = ${info.oy.toFixed(3)} · hits strip <b>${res.hit}</b></span>`;
    if (this.onChange) this.onChange({ pixel: this.pixel, cam: d, S: this.S, H: this.H, COL: this.COL });
  }
}

// ---------------------------------------------------------------------------------------------
// Figure: Heaviside approximations (happrox.py)
// ---------------------------------------------------------------------------------------------
export class HeavisidePlot {
  constructor(root) {
    this.root = root; this.cv = root.querySelector('canvas');
    this.k = 0.1; this.range = 10; this.sel = 'tan';
    this.ui = { k: root.querySelector('[data-r=k]'), kOut: root.querySelector('[data-r=k-out]'), range: root.querySelector('[data-r=range]'), fns: root.querySelector('[data-r=fns]') };
    this.ui.k.addEventListener('input', () => { this.k = Math.pow(10, +this.ui.k.value); this.draw(); });
    this.ui.range.addEventListener('change', () => { this.range = +this.ui.range.value; this.draw(); });
    this.ui.fns.addEventListener('click', (e) => { const b = e.target.closest('[data-f]'); if (!b) return; this.sel = b.dataset.f; this.draw(); });
    onWidth(root, () => this.draw());
  }
  draw() {
    const w = this.cv.parentElement.clientWidth, h = Math.max(220, Math.min(300, w * 0.5));
    const ctx = setupCanvas(this.cv, w, h);
    ctx.clearRect(0, 0, w, h);
    this.ui.kOut.textContent = this.k < 1 ? this.k.toFixed(2) : this.k.toFixed(1);
    const pad = { l: 34, r: 10, t: 10, b: 24 }, R = this.range;
    const X = (x) => pad.l + ((x + R) / (2 * R)) * (w - pad.l - pad.r), Y = (y) => h - pad.b - y * (h - pad.t - pad.b);
    const muted = css('--muted') || '#888', ink = css('--ink') || '#222';
    ctx.strokeStyle = 'rgba(128,128,128,0.25)'; ctx.lineWidth = 1;
    for (const y of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(pad.l, Y(y)); ctx.lineTo(w - pad.r, Y(y)); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(0), Y(1)); ctx.stroke();
    ctx.fillStyle = muted; ctx.font = '11px "Noto Sans", sans-serif'; ctx.textAlign = 'right';
    for (const y of [0, 0.5, 1]) ctx.fillText(String(y), pad.l - 6, Y(y) + 4);
    ctx.textAlign = 'center';
    for (const x of [-R, -R / 2, 0, R / 2, R]) ctx.fillText(String(x), X(x), h - 6);
    // exact step (what the forward pass computes)
    ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(-R), Y(0)); ctx.lineTo(X(0), Y(0)); ctx.lineTo(X(0), Y(1)); ctx.lineTo(X(R), Y(1)); ctx.stroke(); ctx.setLineDash([]);
    const COLORS = { tan: '#e8590c', erfc: '#1c7ed6', circ: '#2b8a3e', log: '#ae3ec9', circdist: '#868e96' };
    const fns = ['circ', 'circdist', 'erfc', 'log', 'tan'];
    for (const f of fns) {
      const on = f === this.sel;
      ctx.strokeStyle = COLORS[f]; ctx.lineWidth = on ? 3 : 1.3; ctx.globalAlpha = on ? 1 : 0.55;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) { const x = -R + (2 * R * i) / 200, y = core.HEAVISIDE[f].smooth(x, this.k); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // surrogate derivative used in backward() for the selected function (scaled to max 1)
    const der = Array.from({ length: 201 }, (_, i) => core.HEAVISIDE[this.sel].bwd(-R + (2 * R * i) / 200, this.k));
    const m = Math.max(...der);
    ctx.strokeStyle = COLORS[this.sel]; ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]);
    ctx.beginPath(); der.forEach((v, i) => { const x = -R + (2 * R * i) / 200; i ? ctx.lineTo(X(x), Y(v / m)) : ctx.moveTo(X(x), Y(v / m)); }); ctx.stroke(); ctx.setLineDash([]);
    [...this.ui.fns.querySelectorAll('[data-f]')].forEach((b) => b.classList.toggle('on', b.dataset.f === this.sel));
  }
}

// ---------------------------------------------------------------------------------------------
// Live loss chart
// ---------------------------------------------------------------------------------------------
export class LossChart {
  constructor(cv, { height = 150 } = {}) { this.cv = cv; this.h = height; this.reset(); onWidth(cv.parentElement, () => this.draw()); }
  reset(maxStep = 295) { this.pts = []; this.events = []; this.maxStep = maxStep; this.draw(); }
  add(hist, events) { for (const p of hist) this.pts.push(p); for (const e of events || []) this.events.push(e); this.draw(); }
  draw() {
    const w = this.cv.parentElement.clientWidth || 300, h = this.h;
    const ctx = setupCanvas(this.cv, w, h);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 40, r: 8, t: 8, b: 20 };
    const vals = this.pts.map((p) => p[1]).filter((v) => v > 0);
    const yMax = Math.max(0.05, ...vals.slice(Math.min(3, vals.length)), ...(vals.length ? [Math.min(vals[0], 0.5)] : [0.3]));
    const X = (s) => pad.l + (s / this.maxStep) * (w - pad.l - pad.r), Y = (v) => h - pad.b - Math.min(1, v / yMax) * (h - pad.t - pad.b);
    const muted = css('--muted') || '#888';
    // phase shading: colour steps vs height steps
    for (const p of this.pts) if (p[2]) { ctx.fillStyle = 'rgba(112,72,232,0.08)'; ctx.fillRect(X(p[0] - 0.5), pad.t, X(1) - X(0) + 0.5, h - pad.t - pad.b); }
    ctx.strokeStyle = 'rgba(128,128,128,0.3)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.stroke();
    ctx.fillStyle = muted; ctx.font = '10px "Noto Sans", sans-serif'; ctx.textAlign = 'right';
    for (const f of [0, 0.5, 1]) ctx.fillText((yMax * f).toFixed(2), pad.l - 4, Y(yMax * f) + 3);
    ctx.textAlign = 'center';
    for (let s = 0; s <= this.maxStep; s += 50) ctx.fillText(String(s), X(s), h - 6);
    for (const e of this.events) {
      if (e.type === 'subdivide' || e.type === 'rays') {
        ctx.strokeStyle = e.type === 'subdivide' ? 'rgba(232,89,12,0.7)' : 'rgba(28,126,214,0.6)'; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(e.step), pad.t); ctx.lineTo(X(e.step), h - pad.b); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = ctx.strokeStyle; ctx.textAlign = 'left';
        ctx.textAlign = e.type === 'subdivide' ? 'left' : 'right';
        ctx.fillText(e.type === 'subdivide' ? `${e.strips}×${e.strips} bars` : `${e.numRays}² rays`, X(e.step) + (e.type === 'subdivide' ? 3 : -3), e.type === 'subdivide' ? pad.t + 9 : h - pad.b - 4);
      }
    }
    ctx.strokeStyle = '#e8590c'; ctx.lineWidth = 1.8; ctx.beginPath();
    this.pts.forEach((p, i) => { i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])); });
    ctx.stroke();
  }
}

export { rgbCss, setupCanvas };
