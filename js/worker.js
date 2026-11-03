// Runs the optimizer (vd-core.js, a port of grad_descent.py) off the main thread and streams progress.
import * as core from './vd-core.js?v=7071d1c4';
import { exportSurface, heightsFull } from './reproject.js?v=7071d1c4';

let current = 0; // id of the run that is allowed to continue
let pace = 0;    // ms to wait after each step (can change during a run)

const tick = () => new Promise((r) => setTimeout(r, 0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function run({ id, cfg, cams, images, pace: p0 = 0, frameMs = 50 }) {
  current = id; pace = p0;
  const t0 = performance.now();
  const events = [];
  let it;
  try {
    it = core.gradDescent(cfg, cams, images, (ev) => {
      const e = { type: ev.type, step: ev.step };
      if (ev.strips) e.strips = ev.strips;
      if (ev.numRays) e.numRays = ev.numRays;
      if (ev.cams) e.cams = ev.cams;
      if (ev.losses) e.annealLosses = ev.losses;
      events.push(e);
    });
  } catch (err) { postMessage({ type: 'error', id, message: err.message }); return; }
  let lastPost = 0, sliceStart = performance.now(), last = null;
  const hist = [];
  try {
    for (;;) {
      if (current !== id) { postMessage({ type: 'stopped', id }); return; }
      const r = it.next();
      if (r.done) { last = r.value ? { ...last, final: true } : last; break; }
      const v = r.value;
      last = v;
      hist.push([v.step, v.perView, v.type === 'H' ? 1 : 0, v.strips, v.numRays]);
      const now = performance.now();
      if (now - lastPost > frameMs || events.length) {
        post(id, v, hist, events, false, now - t0);
        hist.length = 0; events.length = 0; lastPost = now;
      }
      if (pace) await sleep(pace);
      else if (now - sliceStart > 12) { await tick(); sliceStart = performance.now(); }
    }
  } catch (err) { postMessage({ type: 'error', id, message: err.message }); return; }
  post(id, last, hist, events, true, performance.now() - t0);
}

function post(id, v, hist, events, done, ms) {
  postMessage({
    type: done ? 'done' : 'progress', id, ms, step: v.step, perView: v.perView, phase: v.type, strips: v.S, numRays: v.numRays,
    hist: hist.slice(), events: events.slice(), H: Float32Array.from(v.H), u: v.u, reps: v.reps, COL: Float32Array.from(v.COL), S: v.S,
    renders: v.renders.map((x) => Float32Array.from(x)), desired: v.desired.map((x) => Float32Array.from(x)), cams: v.cams, planeZ: v.planeZ ?? 1,
  });
}

// Table 3 of the paper (last column): two views, 45 deg elevation on opposite sides, MSE after 100 steps.
async function table2({ id, cases, seeds }) {
  current = id;
  const cams = [[1, 0, -1], [-1, 0, -1]];
  for (let c = 0; c < cases.length; c++) {
    const vals = [];
    for (const seed of seeds) {
      if (current !== id) return;
      const it = core.gradDescent({ seed, epochs: 101 }, cams, cases[c].images);
      let r, last;
      while (!(r = it.next()).done) last = r.value;
      vals.push(last.perView);
      postMessage({ type: 'table2', id, row: c, vals: vals.slice(), done: false });
      await tick();
    }
  }
  postMessage({ type: 'table2', id, done: true });
}

onmessage = (e) => {
  const m = e.data;
  if (m.type === 'stop') { current = -1; return; }
  if (m.type === 'pace') { pace = m.pace; return; }
  if (m.type === 'run') run(m);
  if (m.type === 'table2') table2(m);
  if (m.type === 'bake') {
    const t0 = performance.now();
    const Hf = heightsFull(m.H, m.u, m.S);
    const r = exportSurface(Hf, m.S, m.COL, m.cams, m.images, m.cams.map(() => m.planeZ), 'surface');
    postMessage({ type: 'baked', id: m.id, purpose: m.purpose, ...r, ms: performance.now() - t0 }, [r.atlas.data.buffer, r.mesh.pos.buffer, r.mesh.uv.buffer, r.mesh.nor.buffer, r.mesh.idx.buffer]);
  }
};
