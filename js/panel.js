// The live optimizer panel (used by the walkthrough page and the standalone UI page).
import * as core from './vd-core.js?v=adfa0743';
import { HeightfieldViewer, CAM_COLORS } from './viewer3d.js?v=adfa0743';
import { LossChart } from './widgets.js?v=adfa0743';
import { PRESETS, makeTarget, fromFile, paint, solid, noiseImage, stripesBW, orient, display } from './targets.js?v=adfa0743';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const workerUrl = new URL('./worker.js?v=adfa0743', import.meta.url);

// ---------------------------------------------------------------------------------------------
// Live optimizer panel (hero demo + playground)
// ---------------------------------------------------------------------------------------------
export class OptimizerPanel {
  constructor(root, { views, editable = false, pace = 0, autostart = false, maxViews = 4 }) {
    this.root = root; this.editable = editable; this.pace = pace; this.maxViews = maxViews;
    this.views = views.map((v) => ({ ...v, image: v.image || makeTarget(v.target) }));
    this.worker = null; this.runId = 0; this.running = false; this.last = null;
    this.viewer = new HeightfieldViewer($('[data-r=viewer]', root));
    this.viewer.onViewChange = (q) => $$('[data-view]', root).forEach((b) => b.classList.toggle('on', +b.dataset.view === q));
    this.chart = new LossChart($('[data-r=chart]', root), { height: editable ? 150 : 110 });
    this.status = $('[data-r=status]', root);
    $('[data-r=run]', root)?.addEventListener('click', () => this.run());
    $('[data-r=stop]', root)?.addEventListener('click', () => this.stop());
    $('[data-r=project]', root)?.addEventListener('change', (e) => this.viewer.setProjection(e.target.checked));
    $('[data-r=step2]', root)?.addEventListener('click', () => this.bakeForView());
    $('[data-r=shade]', root)?.addEventListener('change', (e) => this.viewer.setShading(e.target.checked));
    $('[data-r=arrows]', root)?.addEventListener('change', (e) => { this.viewer.arrowsOn = e.target.checked; this.viewer.setArrows(e.target.checked); });
    $('[data-r=add-view]', root)?.addEventListener('click', () => this.addView());
    $('[data-r=download]', root)?.addEventListener('click', () => this.download());
    $('[data-r=preset]', root)?.addEventListener('change', (e) => this.applyScenario(e.target.value));
    $('[data-r=speed]', root)?.addEventListener('change', (e) => this.worker?.postMessage({ type: 'pace', pace: +e.target.value }));
    this.renderViews();
    this.preview();
    if (autostart) {
      const io = new IntersectionObserver((ents) => { if (ents.some((e) => e.isIntersecting)) { io.disconnect(); this.run(); } }, { threshold: 0.15 });
      io.observe(root);
    }
  }

  cams() { return this.views.map((v) => core.camFromAzEl(v.az, v.el)); }

  // pictures rotated to face their cameras (what the optimizer receives), and high-res versions for the projection
  inputs() { return this.views.map((v) => orient({ width: 64, height: 64, data: core.desiredFromImage(v.image, 64) }, v.az, 64)); }
  textures() { return this.views.map((v) => orient(v.image, v.az, 256)); }

  options() {
    const o = {};
    const get = (k) => $(`[data-opt=${k}]`, this.root);
    if (!this.editable) return { seed: Math.floor(Math.random() * 1e9), ...V2 };
    o.epochs = Math.max(2, Math.min(600, +get('epochs').value || 295));
    o.simAnnealing = get('anneal').checked;
    o.barrier = get('barrier').checked;
    o.neighbor = get('neighbor').checked;
    o.heaviside = get('heaviside').value;
    o.k = +get('k').value || 0.1;
    o.optim = get('optim').value;
    o.seed = get('seed').value === '' ? Math.floor(Math.random() * 1e9) : +get('seed').value;
    if (!get('v10').checked) Object.assign(o, V2);
    return o;
  }

  // a uniform heightfield before the run starts (what grad_descent starts from)
  preview() {
    this.viewer.setSurface(new Float32Array(16).fill(0.5), 4, 2, new Float32Array(64 * 3).fill(1), 8, 1);
    this.viewer.setCameras(this.cams(), this.textures());
    this.views.forEach((v, q) => { const c = $(`[data-r=target-${q}]`, this.root); if (c) paint(c, core.desiredFromImage(v.image, 64), 64); const r = $(`[data-r=render-${q}]`, this.root); if (r) paint(r, new Float32Array(64 * 64 * 3).fill(0.86), 64); });
  }

  renderViews() {
    const box = $('[data-r=views]', this.root);
    box.innerHTML = this.views.map((v, q) => `
      <div class="view-card" data-q="${q}">
        <div class="view-head"><span class="dot" style="background:${CAM_COLORS[q]}">${q + 1}</span>
          ${this.editable ? `<select data-r="target-sel" aria-label="desired appearance for view ${q + 1}">${Object.entries(PRESETS).map(([k, p]) => `<option value="${k}" ${k === v.target ? 'selected' : ''}>${p.label}</option>`).join('')}<option value="__upload">Upload an image…</option>${v.target === '__file' ? '<option value="__file" selected>Your image</option>' : ''}</select>
          <input type="file" accept="image/*" hidden data-r="file">` : `<span class="view-label">${PRESETS[v.target]?.label || ''}</span>`}
          ${this.editable && this.views.length > 2 ? `<button class="icon-btn" data-r="remove" aria-label="remove view ${q + 1}">✕</button>` : ''}
        </div>
        <div class="view-imgs">
          <figure><canvas data-r="target-${q}" width="64" height="64"></canvas><figcaption>desired</figcaption></figure>
          <figure><canvas data-r="render-${q}" width="64" height="64"></canvas><figcaption>rendered</figcaption></figure>
        </div>
        ${this.editable ? `<div class="view-angles">
          <label>azimuth <input type="range" min="0" max="359" step="1" value="${v.az}" data-r="az"><output>${v.az}°</output></label>
          <label>elevation <input type="range" min="15" max="85" step="1" value="${v.el}" data-r="el"><output>${v.el}°</output></label></div>` : `<div class="view-angles static">azimuth ${v.az}°, elevation ${v.el}°</div>`}
        <button class="ghost-btn" data-view="${q}">Look from view ${q + 1}</button>
      </div>`).join('');
    $$('.view-card', box).forEach((card) => {
      const q = +card.dataset.q;
      $('[data-view]', card).addEventListener('click', () => (this.viewer.view === q ? this.viewer.freeView(true) : this.viewer.setView(q)));
      if (!this.editable) return;
      const sel = $('[data-r=target-sel]', card), file = $('[data-r=file]', card);
      sel.addEventListener('change', () => {
        if (sel.value === '__upload') { file.click(); sel.value = this.views[q].target; return; }
        if (sel.value === '__file') return;
        this.views[q].target = sel.value; this.views[q].image = makeTarget(sel.value); this.reset();
      });
      file.addEventListener('change', async () => {
        if (!file.files[0]) return;
        try { this.views[q].image = await fromFile(file.files[0]); this.views[q].target = '__file'; this.renderViews(); this.reset(); }
        catch (e) { this.setStatus(`Could not read that image: ${e.message}`); }
      });
      for (const key of ['az', 'el']) {
        const inp = $(`[data-r=${key}]`, card);
        inp.addEventListener('input', () => { this.views[q][key] = +inp.value; inp.nextElementSibling.textContent = `${inp.value}°`; if (this.last || this.running) this.reset(); else this.viewer.setCameras(this.cams(), this.textures()); this.warnAngles(); });
      }
      $('[data-r=remove]', card)?.addEventListener('click', () => { this.views.splice(q, 1); this.renderViews(); this.reset(); });
    });
    const add = $('[data-r=add-view]', this.root);
    if (add) add.disabled = this.views.length >= this.maxViews;
    this.warnAngles();
  }

  addView() {
    if (this.views.length >= this.maxViews) return;
    const used = this.views.map((v) => v.az);
    const az = [0, 90, 180, 270, 45, 135, 225, 315].find((a) => !used.includes(a)) ?? 0;
    this.views.push({ az, el: 45, target: 'heart', image: makeTarget('heart') });
    this.renderViews(); this.reset();
  }

  applyScenario(key) {
    const S = SCENARIOS[key];
    if (!S) return;
    this.views = S.views.map((v) => ({ ...v, image: makeTarget(v.target) }));
    this.renderViews(); this.reset();
  }

  // the views changed: stop any run, drop the old result and go back to the flat starting surface
  reset() {
    this.stop(true);
    this.runId += 1; this.last = null; this.baked = null;
    this.viewer.clearTextured();
    this.setSteps({ step2: false, download: false, project: false });
    this.chart.reset(this.editable ? this.options().epochs : 295);
    if (this.editable) this.setStatus('Pick views and pictures, then press Optimize.');
    this.preview();
  }

  setSteps({ step2, download, project, label = 'Project &amp; make printable' }) {
    const s2 = $('[data-r=step2]', this.root); if (s2) { s2.disabled = !step2; s2.innerHTML = `<span class="step-n">2</span> ${label}`; }
    const dl = $('[data-r=download]', this.root); if (dl) dl.disabled = !download;
    const pr = $('[data-r=project]', this.root); if (pr) { pr.disabled = !project; pr.checked = project; }
  }

  warnAngles() {
    const w = $('[data-r=angle-warn]', this.root);
    if (!w) return;
    const cams = this.cams();
    let minAng = 180;
    for (let i = 0; i < cams.length; i++) for (let j = i + 1; j < cams.length; j++) {
      const d = Math.min(1, Math.max(-1, cams[i].reduce((s, v, k) => s + v * cams[j][k], 0)));
      minAng = Math.min(minAng, (Math.acos(d) * 180) / Math.PI);
    }
    w.hidden = minAng >= 40;
    w.textContent = `Two views are only ${minAng.toFixed(0)}° apart. Nearby views compete for the same bars, so expect a higher loss (the paper recommends at least 60° between views).`;
  }

  setRunning(on) {
    this.running = on; this.root.classList.toggle('is-running', on);
    const b = $('[data-r=stop]', this.root); if (b) b.disabled = !on;
  }

  setStatus(html) { if (this.status) this.status.innerHTML = html; }

  run() {
    this.stop(true);
    if (!this.worker) { this.worker = new Worker(workerUrl, { type: 'module' }); this.worker.onmessage = (e) => this.onMessage(e.data); this.worker.onerror = (e) => this.setStatus(`Worker error: ${e.message}`); }
    const opts = this.options();
    const cfg = { ...opts };
    if (cfg.optim === 'SGD') cfg.sgd = true;
    this.cfg = cfg;
    this.runId += 1;
    this.setRunning(true);
    this.viewer.clearTextured(); this.baked = null;
    this.setSteps({ step2: false, download: false, project: false });
    this.chart.reset(cfg.epochs || 295);
    this.viewer.setCameras(this.cams(), this.textures());
    this.t0 = performance.now();
    this.setStatus('Starting… simulated annealing on the initial flat surface');
    this.runViews = this.views.map((v) => ({ ...v }));
    const speed = $('[data-r=speed]', this.root);
    this.worker.postMessage({ type: 'run', id: this.runId, cfg, cams: this.cams(), images: this.inputs(), pace: speed ? +speed.value : this.pace, frameMs: 40 });
  }

  stop(silent = false) {
    const was = this.running;
    if (this.worker && this.running) this.worker.postMessage({ type: 'stop' });
    this.setRunning(false);
    if (!silent && was) {
      this.setStatus(this.last ? `Stopped at step ${this.last.step} · loss per view <b>${this.last.perView.toFixed(3)}</b> · you can still project this result (step 2)` : 'Stopped.');
      if (this.last) this.setSteps({ step2: true, download: false, project: false });
    }
  }

  onMessage(m) {
    if (m.type === 'baked') {
      if (m.id !== this.runId) return;
      this.baked = m;
      this.viewer.setTextured(m);
      this.setSteps({ step2: true, download: true, project: true, label: 'Projected ✓' });
      this.viewer.setProjection(true);
      this.setStatus(`<b>Ready to print</b> · the images are projected onto the surface (per segment, blended by how directly each camera sees it) and the surface is sliced thin · ${m.faces} faces · ${(m.ms / 1000).toFixed(1)} s`);
      return;
    }
    if (m.id !== this.runId) return;
    if (m.type === 'error') { this.setStatus(`<span class="err">${m.message}</span>`); this.setRunning(false); return; }
    if (m.type === 'stopped') return;
    this.last = m;
    this.viewer.setSurface(m.H, m.u, m.reps, m.COL, m.S, 1);
    this.viewer.setCameraPlane(m.planeZ ?? 1);
    if (m.events.some((e) => e.type === 'anneal')) this.viewer.setCameras(m.cams, this.textures());
    if (this.viewer.view >= 0) this.viewer.setView(this.viewer.view);
    const azs = (this.runViews || this.views).map((v) => v.az);
    m.desired.forEach((d, q) => { const c = $(`[data-r=target-${q}]`, this.root); if (c) paint(c, display(d, m.numRays, azs[q]), m.numRays); });
    m.renders.forEach((r, q) => { const c = $(`[data-r=render-${q}]`, this.root); if (c) paint(c, display(r, m.numRays, azs[q]), m.numRays); });
    this.chart.add(m.hist, m.events);
    const phase = m.phase === 'H' ? 'updating heights' : 'updating colours';
    const secs = (m.ms / 1000).toFixed(1);
    if (m.type === 'done') {
      this.setRunning(false);
      this.setStatus(`<b>Done</b> · ${m.step} steps in ${secs} s, in your browser · ${m.S}×${m.S} bars, ${m.numRays}×${m.numRays} rays per view · loss per view <b>${m.perView.toFixed(3)}</b>`);
      this.setSteps({ step2: true, download: false, project: false });
      if (this.onDone) this.onDone(m);
    } else {
      this.setStatus(`step <b>${m.step}</b> · ${phase} · ${m.S}×${m.S} bars · ${m.numRays}×${m.numRays} rays · loss per view <b>${m.perView.toFixed(3)}</b> · ${secs} s`);
    }
  }

  // bake the projected mesh in the worker and show it in the viewer
  bakeForView() {
    const m = this.last;
    if (!m || !this.worker) return;
    const views = this.runViews || this.views;
    const tex = views.map((v) => orient(v.image, v.az, 256));
    this.setSteps({ step2: false, download: false, project: false, label: 'Projecting…' });
    this.viewer.clearTextured(); this.baked = null;
    this.setStatus('Slicing the surface thin and projecting the images onto it…');
    this.worker.postMessage({ type: 'bake', purpose: 'view', id: this.runId, H: m.H, u: m.u, S: m.S, COL: m.COL, cams: m.cams, planeZ: m.planeZ ?? 1, images: tex.map((t) => ({ n: t.width, data: t.data })) });
  }

  async download() {
    const m = this.last, baked = this.baked;
    if (!m || !baked) { this.setStatus('Run step 1 (optimize) and step 2 (project) first.'); return; }
    const S = m.S, Hrep = new Float32Array(S * S);
    for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) Hrep[a * S + b] = m.H[(a % m.u) * m.u + (b % m.u)];
    const views = this.runViews || this.views;
    const tex = views.map((v) => orient(v.image, v.az, 256));
    const png = async (data, n) => { const cv = document.createElement('canvas'); paint(cv, data, n); return new Uint8Array(await (await new Promise((r) => cv.toBlob(r, 'image/png'))).arrayBuffer()); };
    const atlasPng = async ({ width, height, data }) => { const cv = document.createElement('canvas'); cv.width = width; cv.height = height; cv.getContext('2d').putImageData(new ImageData(data, width, height), 0, 0); return new Uint8Array(await (await new Promise((r) => cv.toBlob(r, 'image/png'))).arrayBuffer()); };
    const files = [];
    files.push(['surface.obj', new TextEncoder().encode(baked.obj)]);
    files.push(['surface.mtl', new TextEncoder().encode(baked.mtl)]);
    files.push(['surface.png', await atlasPng(baked.atlas)]);
    files.push(['result_heights.npy', npy(Hrep, [S, S])]);
    files.push(['result_colors.npy', npy(m.COL, [S, S, 3])]);
    for (let q = 0; q < views.length; q++) files.push([`view${q + 1}.png`, await png(tex[q].data, tex[q].width)]);
    files.push(['result_cameras.json', new TextEncoder().encode(JSON.stringify({ camera_directions: m.cams, image_paths: views.map((_, q) => `view${q + 1}.png`), camera_z: m.planeZ ?? 1 }, null, 1))]);
    files.push(['README.txt', new TextEncoder().encode(`View-dependent surface optimized in the browser (port of the code release, src/grad_descent.py).

surface.obj / .mtl / .png   printable closed mesh with the desired images projected onto it (as written by
                            src/reproject.py). Import into your slicer for full-colour printing. The surface
                            is 1 x 1 with heights in the same units; scale it to the print size you want.
result_heights.npy          ${S}x${S} bar heights        result_colors.npy   ${S}x${S}x3 bar colours
result_cameras.json         camera directions, images and camera plane used by the optimizer
view*.png                   desired images, as given to the optimizer (rotated so each faces its camera)

To redo the projection with the Python code (src/ folder of the code release):
    python reproject.py path/to/result_heights.npy

Settings: ${JSON.stringify({ ...this.cfg, views: views.map((v) => ({ azimuth: v.az, elevation: v.el, target: v.target })) })}
Final loss per view: ${m.perView.toFixed(4)}
`)]);
    const blob = new Blob([zip(files)], { type: 'application/zip' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'view_dependent_surface.zip';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    this.setStatus(`Downloaded: printable mesh with projected images (${baked.faces} faces, baked in ${(baked.ms / 1000).toFixed(1)} s) and the optimizer output.`);
  }
}

// Code release v2.0 (default on this page): camera rays start at the current mean surface height
// (re-centred every 10 steps) and the neighbour loss is averaged over the bars. v1.0 = fixed plane z = 1, summed.
export const V2 = { adaptivePlane: 'mean', planeEvery: 10, neighborNorm: 'wrap', neighborWeight: 1.0, barrierMax: 1.0 };

export const SCENARIOS = {
  fig1: { views: [{ az: 0, el: 45, target: 'stripes-rg' }, { az: 90, el: 45, target: 'stripes-bw' }, { az: 180, el: 45, target: 'stripes-yp' }] },
  fig10: { views: [{ az: 0, el: 60, target: 'diag-on' }, { az: 90, el: 60, target: 'checker-bw' }, { az: 180, el: 60, target: 'checker-rg' }, { az: 270, el: 60, target: 'dots-yb' }] },
  sunmoon: { views: [{ az: 0, el: 40, target: 'sun' }, { az: 180, el: 40, target: 'moon' }] },
  ab: { views: [{ az: 0, el: 40, target: 'A' }, { az: 180, el: 40, target: 'B' }] },
  cmyk: { views: [{ az: 0, el: 30, target: 'cyan' }, { az: 90, el: 30, target: 'magenta' }, { az: 180, el: 30, target: 'yellow' }, { az: 270, el: 30, target: 'black' }] },
  heartstar: { views: [{ az: 0, el: 45, target: 'heart' }, { az: 180, el: 45, target: 'star' }] },
};

// ---------------------------------------------------------------------------------------------
// Files: .npy, OBJ/MTL, zip (store)
// ---------------------------------------------------------------------------------------------
export function npy(data, shape) {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  const total = 10 + header.length + 1;
  header += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const out = new Uint8Array(10 + header.length + data.length * 4);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  out.set(new TextEncoder().encode(header), 10);
  out.set(new Uint8Array(Float32Array.from(data).buffer), 10 + header.length);
  return out;
}
function toOBJ(H, COL, S) {
  const v = [], f = [], m = [];
  let n = 1;
  for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
    const h = H[a * S + b], x0 = a / S, x1 = (a + 1) / S, y0 = b / S, y1 = (b + 1) / S;
    const c = [0, 1, 2].map((k) => Math.max(0, Math.min(1, COL[(a * S + b) * 3 + k])));
    m.push(`newmtl bar_${a}_${b}\nKd ${c.map((x) => x.toFixed(4)).join(' ')}\n`);
    for (const [x, y, z] of [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]]) v.push(`v ${x.toFixed(5)} ${y.toFixed(5)} ${z.toFixed(5)}`);
    const q = (i, j, k, l) => `f ${n + i} ${n + j} ${n + k} ${n + l}`;
    f.push(`usemtl bar_${a}_${b}`, q(0, 3, 2, 1), q(4, 5, 6, 7), q(0, 1, 5, 4), q(1, 2, 6, 5), q(2, 3, 7, 6), q(3, 0, 4, 7));
    n += 8;
  }
  return { obj: `# view-dependent heightfield (${S}x${S} bars)\nmtllib surface.mtl\n${v.join('\n')}\n${f.join('\n')}\n`, mtl: m.join('\n') };
}
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
function crc32(d) { let c = 0xffffffff; for (let i = 0; i < d.length; i++) c = CRC[(c ^ d[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
export function zip(files) {
  const parts = [], central = []; let off = 0;
  const u16 = (x) => [x & 255, (x >>> 8) & 255], u32 = (x) => [x & 255, (x >>> 8) & 255, (x >>> 16) & 255, (x >>> 24) & 255];
  for (const [name, data] of files) {
    const nm = new TextEncoder().encode(name), crc = crc32(data);
    const local = new Uint8Array([...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nm.length), ...u16(0)]);
    parts.push(local, nm, data);
    central.push(new Uint8Array([...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21), ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nm.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0), ...u32(off)]), nm);
    off += local.length + nm.length + data.length;
  }
  const cdSize = central.reduce((s, p) => s + p.length, 0);
  const end = new Uint8Array([...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length), ...u32(cdSize), ...u32(off), ...u16(0)]);
  return new Blob([...parts, ...central, end]);
}


export { workerUrl };
