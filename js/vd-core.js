// vd-core.js — a line-by-line JavaScript port of the view-dependent heightfield optimizer
// (src/rendering.py, src/happrox.py, src/opt.py, src/heightfield.py, src/image.py,
//  src/simannealing.py, src/grad_descent.py).
//
// Faithfulness notes
// * Heights/colours are float32 like the PyTorch tensors; Math.fround is applied where PyTorch
//   rounds to float32 (theta, ray height, colour accumulation, AdamW updates).
// * Forward pass uses the exact Heaviside step (the torch.autograd.Functions in happrox.py return
//   heaviside(x) in forward); only the backward pass uses the smooth surrogate derivative.
// * Gradients are written by hand (no autograd) and are checked against torch.autograd in
//   verify/check_core.mjs.
// * AdamW follows torch.optim.AdamW (decoupled weight decay 0.01, betas 0.9/0.999, eps 1e-8).
//   The repo pins torch 1.9, whose zero_grad() zeroes gradients instead of freeing them; this matters
//   for the parameter that is frozen during block-coordinate descent (it still receives weight
//   decay). `torch19` (default true) reproduces that.

const f32 = Math.fround;

// ---------------------------------------------------------------------------------------------
// Random numbers (seeded, so runs are reproducible in the browser)
// ---------------------------------------------------------------------------------------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------------------------
// rendering.py — slicing the heightfield with a projected ray
// ---------------------------------------------------------------------------------------------

// 2x2 solve with partial pivoting, mirroring LAPACK gesv as used by np.linalg.solve.
function solve2(m00, m01, m10, m11, b0, b1) {
  if (Math.abs(m10) > Math.abs(m00)) {
    let t = m00; m00 = m10; m10 = t; t = m01; m01 = m11; m11 = t; t = b0; b0 = b1; b1 = t;
  }
  const l = m10 * (1 / m00);
  const u11 = m11 - l * m01;
  const y1 = b1 - l * b0;
  const x1 = y1 / u11;
  const x0 = (b0 - x1 * m01) / m00;
  return [x0, x1];
}

// find_intersections(projected_origin, projected_direction, num_strips)
export function findIntersections(ox, oy, dx, dy, n) {
  const a1x = ox + 100 * dx, a1y = oy + 100 * dy;
  const vx = a1x - ox, vy = a1y - oy; // A[1] - A[0]
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const c = i / n; // B = [[c,0],[c,5]], B0 - B1 = (0,-5), rhs = B0 - A0
    const [t] = solve2(vx, 0, vy, -5, c - ox, 0 - oy);
    const px = (1 - t) * ox + t * a1x, py = (1 - t) * oy + t * a1y;
    if (px >= -0.0001 && px <= 1.0001 && py >= -0.0001 && py <= 1.0001) pts.push([px, py]);
  }
  for (let j = 0; j <= n; j++) {
    const c = j / n; // B = [[0,c],[5,c]], B0 - B1 = (-5,0)
    const [t] = solve2(vx, -5, vy, 0, 0 - ox, c - oy);
    const px = (1 - t) * ox + t * a1x, py = (1 - t) * oy + t * a1y;
    if (px >= -0.0001 && px <= 1.0001 && py >= -0.0001 && py <= 1.0001) pts.push([px, py]);
  }
  // utils.Sort_Tuple: stable sort on x, reversed when the ray faces -x
  if (dx >= 0) pts.sort((a, b) => a[0] - b[0]);
  else pts.sort((a, b) => b[0] - a[0]);
  return pts;
}

// find_ids_and_lengths + get_2D_ray + get_width_sums for one 3D ray.
export function rayInfo(origin, dir, n) {
  const pts = findIntersections(origin[0], origin[1], dir[0], dir[1], n);
  const ids = [], widths = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
    ids.push([Math.floor(mx * n), Math.floor(my * n)]);
    const ex = pts[i + 1][0] - pts[i][0], ey = pts[i + 1][1] - pts[i][1];
    widths.push(Math.sqrt(ex * ex + ey * ey));
  }
  const nx = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
  const nn = Math.sqrt(nx * nx + dir[2] * dir[2]);
  const d2x = nx / nn, d2y = dir[2] / nn;
  const t = (pts[0][0] - origin[0]) / dir[0];
  const oz = origin[2] + t * dir[2];
  const widthSums = [];
  let s = widths[0];
  widthSums.push(s);
  for (let i = 1; i < widths.length; i++) { s += widths[i]; widthSums.push(s); }
  return { ids, widths, widthSums, oy: oz, d2x, d2y, pts };
}

// The camera rays of grad_descent.py (pixel centres on z = 1, traced back 20 units) and
// rendering.process_rays. Returns a packed structure used by render/backward.
// Periodic slicing ("repeating units", Sec. 3): the surface repeats with period 1 in x and y, so a ray
// is followed from where it descends below zTop until it reaches the ground, through neighbouring copies.
// Bars are looked up modulo the grid. Every ray then ends on a bar (no last-bar fallback, no edge walls).
export function periodicRayInfo(origin, dir, n, zTop = 2) {
  const L = Math.hypot(dir[0], dir[1]);
  const ux = dir[0] / L, uy = dir[1] / L, s = -dir[2] / L; // horizontal unit direction, descent per unit length
  // the point where the ray crosses z = zTop, and the length until it reaches z = 0
  const tTop = (origin[2] - zTop) / s;
  const ex = origin[0] + ux * tTop, ey = origin[1] + uy * tTop;
  const len = zTop / s;
  let cx = Math.floor(ex * n), cy = Math.floor(ey * n);
  const stx = ux > 0 ? 1 : -1, sty = uy > 0 ? 1 : -1;
  const tdx = Math.abs(ux) > 1e-12 ? 1 / (Math.abs(ux) * n) : Infinity, tdy = Math.abs(uy) > 1e-12 ? 1 / (Math.abs(uy) * n) : Infinity;
  let tmx = Math.abs(ux) > 1e-12 ? ((cx + (stx > 0 ? 1 : 0)) / n - ex) / ux : Infinity;
  let tmy = Math.abs(uy) > 1e-12 ? ((cy + (sty > 0 ? 1 : 0)) / n - ey) / uy : Infinity;
  const ids = [], widths = [];
  let t = 0;
  while (t < len - 1e-12) {
    const tn = Math.min(tmx, tmy, len);
    if (tn - t > 1e-12) { ids.push([((cx % n) + n) % n, ((cy % n) + n) % n]); widths.push(tn - t); }
    t = tn;
    if (tmx < tmy) { tmx += tdx; cx += stx; } else { tmy += tdy; cy += sty; }
  }
  const widthSums = []; let acc = 0; for (const w of widths) { acc += w; widthSums.push(acc); }
  return { ids, widths, widthSums, oy: zTop, d2x: 1 / Math.hypot(1, s), d2y: -s / Math.hypot(1, s) };
}

export function processRays(cam, numRays, numStrips, pixelZ = 1, periodic = false) {
  const rw = 1 / numRays;
  const nR = numRays * numRays;
  const start = new Int32Array(nR + 1);
  const idsL = [], cterm = [];
  const oyf = new Float64Array(nR);
  let p = 0, wrapped = 0;
  for (let i = 0; i < numRays; i++) {
    for (let k = 0; k < numRays; k++) {
      const coord = [i * rw + rw / 2, k * rw + rw / 2, pixelZ]; // grad_descent.py: z = 1
      const origin = [coord[0] - 20 * cam[0], coord[1] - 20 * cam[1], coord[2] - 20 * cam[2]];
      const info = periodic ? periodicRayInfo(origin, cam, numStrips) : rayInfo(origin, cam, numStrips);
      const r = i * numRays + k;
      start[r] = p;
      const slope = info.d2y / info.d2x;
      for (let q = 0; q < info.ids.length; q++) {
        let a = info.ids[q][0], b = info.ids[q][1];
        // Python indexing: a midpoint just outside [0,1] (the code accepts intersections within 1e-4 of
        // the border) gives index -1, which Python silently wraps to the last row/column.
        if (a < 0) { a += numStrips; wrapped++; } if (b < 0) { b += numStrips; wrapped++; }
        // an index == numStrips wraps the same way (rendering.py, v2.0)
        if (a >= numStrips) { a -= numStrips; wrapped++; } if (b >= numStrips) { b -= numStrips; wrapped++; }
        idsL.push(a * numStrips + b);
        // theta_i = H_i - (d.y/d.x) * width_sums[i]  -> the constant part, rounded to float32
        cterm.push(f32(slope * info.widthSums[q]));
        p++;
      }
      oyf[r] = f32(info.oy); // torch.full(size, y) is float32
    }
  }
  start[nR] = p;
  return { numRays, numStrips, nR, start, ids: Int32Array.from(idsL), cterm: Float32Array.from(cterm), oy: oyf, maxLen: maxRun(start), wrapped };
}
function maxRun(start) { let m = 0; for (let i = 0; i + 1 < start.length; i++) m = Math.max(m, start[i + 1] - start[i]); return m; }

// ---------------------------------------------------------------------------------------------
// happrox.py — Heaviside step (forward) and surrogate derivatives (backward)
// ---------------------------------------------------------------------------------------------
export const HEAVISIDE = {
  // name: { fwd(x, k, rnd) -> {0,1}, bwd(x, k) -> d/dx used by autograd, smooth(x,k) for plots }
  tan: { label: 'tanh', fwd: (x) => (x > 0 ? 1 : 0), bwd: (x, k) => { const t = Math.tanh(x * k); return 1 - t * t; }, smooth: (x, k) => 0.5 + 0.5 * Math.tanh(k * x) },
  erfc: { label: 'erfc', fwd: (x) => (x > 0 ? 1 : 0), bwd: (x, k) => (2 * Math.exp(-((k * x) ** 2))) / Math.sqrt(Math.PI), smooth: (x, k) => 0.5 + 0.5 * erf(k * x) },
  circ: { label: 'circle', fwd: (x) => (x > 0 ? 1 : 0), bwd: circBwd, smooth: (x, a) => 0.5 + 0.5 * (x / Math.sqrt(x * x + a * a)) },
  log: { label: 'log (stochastic)', fwd: (x, k, rnd) => (rnd() < 0.5 + 0.5 * Math.tanh(k * x) ? 1 : 0), bwd: (x, k) => { const t = Math.tanh(x * k); return 1 - t * t; }, smooth: (x, k) => 0.5 + 0.5 * Math.tanh(k * x) },
  circdist: { label: 'circle distance (stochastic)', fwd: (x, a, rnd) => (rnd() < 0.5 + 0.5 * (x / Math.sqrt(x * x + a * a)) ? 1 : 0), bwd: circBwd, smooth: (x, a) => 0.5 + 0.5 * (x / Math.sqrt(x * x + a * a)) },
};
function circBwd(x, a) { const s = a * a + x * x; return (-(x * x) / (2 * Math.pow(s, 1.5)) + 1 / (2 * Math.sqrt(s))) * 2 * a; }
export function erf(x) { // Abramowitz-Stegun 7.1.26 (plots only)
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}

// ---------------------------------------------------------------------------------------------
// Rendering a camera (render_camera / render_ray), with an optional cache for backward
// ---------------------------------------------------------------------------------------------
// H: Float32Array unit heights (u*u), reps: num_repeats, COL: Float32Array (S*S*3), S = u*reps.
export function renderCamera(H, u, reps, COL, pack, hv = 'tan', k = 0.1, rnd = Math.random, cache = null) {
  const S = pack.numStrips;
  if (u * reps !== S) throw new Error(`heightfield ${u}x${reps} does not match ${S} strips`);
  const fn = HEAVISIDE[hv] || HEAVISIDE.erfc;
  const out = new Float32Array(pack.nR * 3);
  const L = pack.maxLen;
  const theta = new Float32Array(L);
  let steps = null, argm = null, xin = null;
  if (cache) { steps = cache.steps = new Uint8Array(pack.ids.length); argm = cache.argm = new Int32Array(pack.ids.length); xin = cache.x = new Float64Array(pack.ids.length); }
  const { start, ids, cterm, oy } = pack;
  for (let r = 0; r < pack.nR; r++) {
    const s0 = start[r], n = start[r + 1] - s0;
    // get_backtraced_boundaries (float32) and monotonize_boundary_points (cummax, ties -> last index)
    let best = -Infinity, bi = 0;
    const y = oy[r];
    let c0 = 0, c1 = 0, c2 = 0;
    for (let q = 0; q < n; q++) {
      const sid = ids[s0 + q];
      const a = (sid / S) | 0, b = sid - a * S;
      const h = H[(a % u) * u + (b % u)];
      const th = f32(h - cterm[s0 + q]);
      if (th >= best) { best = th; bi = q; }
      theta[q] = best;
      if (argm) argm[s0 + q] = bi;
    }
    // render_ray_given_boundaries
    let id = ids[s0] * 3;
    c0 = COL[id]; c1 = COL[id + 1]; c2 = COL[id + 2];
    for (let q = 0; q < n - 1; q++) {
      const x = y - theta[q];
      const g = fn.fwd(x, k, rnd);
      if (steps) { steps[s0 + q] = g; xin[s0 + q] = x; }
      if (g) {
        const ia = ids[s0 + q] * 3, ib = ids[s0 + q + 1] * 3;
        c0 = f32(c0 + f32(COL[ib] - COL[ia]));
        c1 = f32(c1 + f32(COL[ib + 1] - COL[ia + 1]));
        c2 = f32(c2 + f32(COL[ib + 2] - COL[ia + 2]));
      }
    }
    out[3 * r] = c0; out[3 * r + 1] = c1; out[3 * r + 2] = c2;
  }
  return out;
}

// Backward of renderCamera. gOut: dL/d(pixel colours) (nR*3). Accumulates into gH (u*u) and gCOL.
export function renderBackward(H, u, reps, COL, pack, cache, gOut, gH, gCOL, hv = 'tan', k = 0.1) {
  const S = pack.numStrips;
  const fn = HEAVISIDE[hv] || HEAVISIDE.erfc;
  const { start, ids } = pack;
  const L = pack.maxLen;
  const gth = new Float64Array(L);
  for (let r = 0; r < pack.nR; r++) {
    const s0 = start[r], n = start[r + 1] - s0;
    const g0 = gOut[3 * r], g1 = gOut[3 * r + 1], g2 = gOut[3 * r + 2];
    if (g0 === 0 && g1 === 0 && g2 === 0) continue;
    if (gCOL) {
      const i0 = ids[s0] * 3;
      gCOL[i0] += g0; gCOL[i0 + 1] += g1; gCOL[i0 + 2] += g2;
    }
    if (gH) for (let q = 0; q < n; q++) gth[q] = 0;
    for (let q = 0; q < n - 1; q++) {
      const ia = ids[s0 + q] * 3, ib = ids[s0 + q + 1] * 3;
      const st = cache.steps[s0 + q];
      if (gCOL && st) {
        gCOL[ib] += g0; gCOL[ib + 1] += g1; gCOL[ib + 2] += g2;
        gCOL[ia] -= g0; gCOL[ia + 1] -= g1; gCOL[ia + 2] -= g2;
      }
      if (gH) {
        const dh = g0 * (COL[ib] - COL[ia]) + g1 * (COL[ib + 1] - COL[ia + 1]) + g2 * (COL[ib + 2] - COL[ia + 2]);
        if (dh !== 0) gth[cache.argm[s0 + q]] -= dh * fn.bwd(cache.x[s0 + q], k);
      }
    }
    if (gH) {
      for (let q = 0; q < n; q++) {
        if (gth[q] === 0) continue;
        const sid = ids[s0 + q];
        const a = (sid / S) | 0, b = sid - a * S;
        gH[(a % u) * u + (b % u)] += gth[q];
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// opt.py — regularizers (values and gradients)
// ---------------------------------------------------------------------------------------------
export function barrier(H, rows, minV = 0.1, maxV = 1.5, grad = null, w = 1) {
  const mx = f32(maxV), mn = f32(minV);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < H.length; i++) {
    s1 += Math.log(f32(mx - H[i]));
    s2 += Math.log(f32(H[i] - mn));
    if (grad) grad[i] += w * (1 / f32(mx - H[i]) - 1 / f32(H[i] - mn)) / rows;
  }
  return (-s1 - s2) / rows;
}

// opt.neighbor_fn returns diff/len(H)*len(H), which Python evaluates as diff (the sum). norm = 'mean'
// divides by len(H)*len(H), the evidently intended average (operator precedence).
// norm = 'wrap': every bar against its 4 neighbours, wrapping around the edges of the repeating unit
// (the unit is tiled, so its border bars touch the opposite border of the next tile), averaged over bars.
function neighborWrap(H, n, grad, w) {
  let diff = 0;
  const sg = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
  const ww = w / (n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = H[i * n + j];
    for (const q of [i * n + ((j + 1) % n), ((i + 1) % n) * n + j]) { // each pair once
      const d = f32(v - H[q]);
      diff += Math.abs(d);
      if (grad) { const s2 = sg(d); grad[i * n + j] += ww * s2; grad[q] -= ww * s2; }
    }
  }
  return diff / (n * n);
}

export function neighbor(H, n, grad = null, w = 1, norm = 'sum') {
  if (norm === 'wrap') return neighborWrap(H, n, grad, w);
  if (norm === 'mean') w = w / (n * n);
  let diff = 0;
  const sg = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);
  for (let i = 1; i < n - 1; i++) {
    for (let j = 1; j < n - 1; j++) {
      const v = H[i * n + j];
      const nb = [i * n + (j - 1), i * n + (j + 1), (i - 1) * n + j, (i + 1) * n + j];
      for (const q of nb) {
        const d = f32(v - H[q]);
        diff += Math.abs(d);
        if (grad) { const s = sg(d); grad[i * n + j] += w * s; grad[q] -= w * s; }
      }
    }
  }
  return norm === 'mean' ? diff / (n * n) : diff;
}

// ---------------------------------------------------------------------------------------------
// image.py — get_tensor_from_image (cv2.INTER_NEAREST resizes)
// ---------------------------------------------------------------------------------------------
// img: { width, height, data: Float32Array RGB (h*w*3) in [0,1] }
function resizeNearest(img, W, Hh) {
  const out = new Float32Array(W * Hh * 3);
  const fx = img.width / W, fy = img.height / Hh;
  for (let y = 0; y < Hh; y++) {
    const sy = Math.min(Math.floor(y * fy), img.height - 1);
    for (let x = 0; x < W; x++) {
      const sx = Math.min(Math.floor(x * fx), img.width - 1);
      const s = (sy * img.width + sx) * 3, d = (y * W + x) * 3;
      out[d] = img.data[s]; out[d + 1] = img.data[s + 1]; out[d + 2] = img.data[s + 2];
    }
  }
  return { width: W, height: Hh, data: out };
}
export function desiredFromImage(img, numRays, dim = 64) {
  const warped = resizeNearest(img, dim, dim); // dim is square, so the crop is a no-op
  return resizeNearest(warped, numRays, numRays).data; // pixel (row i, col k) <-> ray (x = i, y = k)
}

// heightfield.double_resolution
export function doubleRes(A, n, ch = 1) {
  const m = n * 2, out = new Float32Array(m * m * ch);
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) for (let c = 0; c < ch; c++) out[(i * m + j) * ch + c] = A[(((i >> 1) * n) + (j >> 1)) * ch + c];
  return out;
}

// ---------------------------------------------------------------------------------------------
// Loss for all cameras (MSE summed over cameras) + gradients
// ---------------------------------------------------------------------------------------------
export function lossAndGrad(st, opts, needH, needC) {
  const { H, u, reps, COL, packs, desired } = st;
  const N = packs[0].nR * 3;
  let mse = 0;
  const renders = [];
  const gH = needH ? new Float64Array(H.length) : null;
  const gC = needC ? new Float64Array(COL.length) : null;
  for (let q = 0; q < packs.length; q++) {
    const cache = (needH || needC) ? {} : null;
    const img = renderCamera(H, u, reps, COL, packs[q], opts.heaviside, opts.k, opts.rnd, cache);
    renders.push(img);
    const d = desired[q];
    let s = 0;
    const gOut = (needH || needC) ? new Float64Array(N) : null;
    for (let i = 0; i < N; i++) { const e = img[i] - d[i]; s += e * e; if (gOut) gOut[i] = f32((2 * e) / N); }
    mse += s / N;
    if (gOut) renderBackward(H, u, reps, COL, packs[q], cache, gOut, gH, gC, opts.heaviside, opts.k);
  }
  let total = mse, bar = null, nb = null;
  const hMax = opts.barrierMax ?? 1.5, wNb = opts.neighborWeight ?? 0.5;
  if (opts.barrier) { bar = barrier(H, u, 0.1, hMax, gH, 0.02); total += 0.02 * bar; }
  if (opts.neighbor) { nb = neighbor(H, u, gH, wNb, opts.neighborNorm); total += wNb * nb; }
  return { mse, total, barrier: bar, neighbor: nb, renders, gH, gC };
}

// ---------------------------------------------------------------------------------------------
// torch.optim.AdamW (single-tensor implementation)
// ---------------------------------------------------------------------------------------------
class AdamW {
  constructor(params, lr) { this.params = params; this.lr = lr; this.state = new Map(); }
  step() {
    const lr = this.lr, wd = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    for (const p of this.params) {
      if (!p.grad) continue;
      let s = this.state.get(p);
      if (!s) { s = { step: 0, m: new Float32Array(p.data.length), v: new Float32Array(p.data.length) }; this.state.set(p, s); }
      s.step += 1;
      const bc1 = 1 - Math.pow(b1, s.step), bc2s = Math.sqrt(1 - Math.pow(b2, s.step));
      const stepSize = lr / bc1;
      const decay = f32(1 - lr * wd);
      const d = p.data, g = p.grad, m = s.m, v = s.v;
      for (let i = 0; i < d.length; i++) {
        d[i] = f32(d[i] * decay);
        m[i] = f32(m[i] + f32(0.1 * f32(g[i] - m[i])));      // exp_avg.lerp_(grad, 1 - beta1)
        v[i] = f32(f32(v[i] * b2) + f32(0.001 * f32(g[i] * g[i])));
        const denom = f32(f32(Math.sqrt(v[i]) / bc2s) + eps);
        d[i] = f32(d[i] - stepSize * f32(m[i] / denom));
      }
    }
  }
  zeroGrad(torch19) { for (const p of this.params) if (p.grad) { if (torch19) p.grad.fill(0); else p.grad = null; } }
}

// torch.optim.SGD(params, lr, momentum=0.9) (optim_type="SGD" in grad_descent.py)
class SGD {
  constructor(params, lr) { this.params = params; this.lr = lr; this.state = new Map(); }
  step() {
    for (const p of this.params) {
      if (!p.grad) continue;
      let buf = this.state.get(p);
      const d = p.data, g = p.grad;
      if (!buf) { buf = Float32Array.from(g); this.state.set(p, buf); }
      else for (let i = 0; i < d.length; i++) buf[i] = f32(f32(buf[i] * 0.9) + g[i]);
      for (let i = 0; i < d.length; i++) d[i] = f32(d[i] - f32(this.lr * buf[i]));
    }
  }
  zeroGrad(torch19) { for (const p of this.params) if (p.grad) { if (torch19) p.grad.fill(0); else p.grad = null; } }
}

// ---------------------------------------------------------------------------------------------
// Schedules
// ---------------------------------------------------------------------------------------------
// Release defaults = the values hard-coded in grad_descent.py / simannealing.py / optimize_surface.py.
export const RELEASE_DEFAULTS = {
  epochs: 295, unit: 4, reps: 2, numRays: 16, heaviside: 'tan', k: 0.1,
  barrier: true, neighbor: true, simAnnealing: true,
  // block coordinate descent, one period: 'C' = update colours, 'H' = update heights
  block: [['C', 10], ['H', 10], ['C', 20], ['H', 10]],
  subdivideEvery: 100, rayLead: 5, maxStrips: Infinity,
  annealEvery: 0, annealSteps: 50, annealT0: 3, annealTmin: 0, annealDecay: 0.99,
  torch19: true,
};

function blockType(block, j) {
  const period = block.reduce((s, b) => s + b[1], 0);
  let r = j % period;
  for (const [t, n] of block) { if (r < n) return t; r -= n; }
  return block[block.length - 1][0];
}

// simannealing.sim_annealing
export function simAnnealing(st, opts, rnd, steps, T0, Tmin, decay) {
  let T = T0;
  const lo = opts.barrier ? 0.1 + 1e-4 : 0.1, hi = opts.barrier ? (opts.barrierMax ?? 1.5) - 1e-4 : 3.9; // stay inside the barrier bounds
  const losses = [];
  for (let i = 0; i < steps; i++) {
    if (Tmin > 0 && T <= Tmin) break;
    const Hn = new Float32Array(st.H.length), Cn = new Float32Array(st.COL.length);
    for (let q = 0; q < Hn.length; q++) Hn[q] = Math.min(f32(hi), Math.max(f32(lo), f32(st.H[q] + f32(0.05 * f32(f32(rnd()) - 0.5)))));
    for (let q = 0; q < Cn.length; q++) Cn[q] = Math.min(1, Math.max(0, f32(st.COL[q] + f32(0.05 * f32(f32(rnd()) - 0.5)))));
    const oldL = lossAndGrad(st, { ...opts, barrier: false, neighbor: false }, false, false).mse;
    const newL = lossAndGrad({ ...st, H: Hn, COL: Cn }, { ...opts, barrier: false, neighbor: false }, false, false).mse;
    let accept = newL <= oldL;
    if (!accept) accept = rnd() <= Math.exp((oldL - newL) / T);
    if (accept) { st.H = Hn; st.COL = Cn; }
    losses.push(accept ? newL : oldL);
    T *= decay;
  }
  return losses;
}

// grad_descent.grad_descent as a generator: yields after every step with progress information.
// images: array of {width,height,data RGB float} desired appearances; cams: [[x,y,z],...]
export function* gradDescent(cfgIn, cams, images, onEvent = null) {
  const cfg = { ...RELEASE_DEFAULTS, ...cfgIn };
  const rnd = cfg.rnd || mulberry32(cfg.seed ?? 1);
  const opts = { heaviside: cfg.heaviside, k: cfg.k, rnd, barrier: cfg.barrier, neighbor: cfg.neighbor, neighborNorm: cfg.neighborNorm || 'sum', neighborWeight: cfg.neighborWeight ?? 0.5, barrierMax: cfg.barrierMax ?? 1.5 };
  // optimize_surface.py: camera jitter "prevent singular matrix"
  const C = cams.map((c) => c.map((v) => v + (cfg.jitter ? cfg.jitter : rnd() / 100)));
  if (cfg.jitterFixed) for (let q = 0; q < C.length; q++) for (let c = 0; c < 3; c++) C[q][c] = cams[q][c] + cfg.jitterFixed[q][c];
  let u = cfg.unit, reps = cfg.reps, numRays = cfg.numRays;
  let S = u * reps;
  const st = { u, reps, H: new Float32Array(u * u).fill(cfg.initHeight ?? 0.5), COL: new Float32Array(S * S * 3).fill(1) };
  if (cfg.initH) st.H = Float32Array.from(cfg.initH);
  let planeZ = cfg.pixelZ ?? 1;
  let planeZs = null; // per-camera planes ('hit' mode)
  // average height of the points where each camera's rays hit the surface (top or side of the hit bar)
  const hitHeights = () => st.packs.map((pk) => {
    let s = 0, cnt = 0;
    for (let r = 0; r < pk.nR; r++) {
      const s0 = pk.start[r], len = pk.start[r + 1] - s0; let best = -Infinity, hit = -1;
      for (let k = 0; k < len; k++) { const sid = pk.ids[s0 + k], a = (sid / S) | 0, b = sid - a * S; best = Math.max(best, f32(Hp.data[(a % u) * u + (b % u)] - pk.cterm[s0 + k])); if (!(pk.oy[r] > best)) { hit = k; break; } }
      if (hit < 0) continue;
      const sid = pk.ids[s0 + hit], a = (sid / S) | 0, b = sid - a * S, h = Hp.data[(a % u) * u + (b % u)];
      const zin = hit === 0 ? pk.oy[r] : pk.oy[r] - pk.cterm[s0 + hit - 1];
      s += Math.min(h, zin); cnt++;
    }
    return cnt ? s / cnt : null;
  });
  const surfaceHeight = (H) => {
    if (cfg.adaptivePlane === 'mean') return H.reduce((a, b) => a + b, 0) / H.length;
    const s = Float32Array.from(H).sort(); return s[Math.min(s.length - 1, Math.floor(s.length * (cfg.adaptiveQuantile ?? 0.75)))];
  };
  if (cfg.adaptivePlane && cfg.adaptivePlane !== 'hit') planeZ = surfaceHeight(st.H);
  if (cfg.adaptivePlane === 'hit') planeZ = surfaceHeight(st.H);
  const build = () => { st.packs = C.map((c, q) => processRays(c, numRays, S, planeZs ? planeZs[q] : planeZ, !!cfg.periodic)); st.desired = images.map((im) => desiredFromImage(im, numRays)); };
  build();
  const history = [];
  if (cfg.simAnnealing) {
    const l = simAnnealing(st, opts, rnd, cfg.annealSteps, cfg.annealT0, cfg.annealTmin, cfg.annealDecay);
    if (onEvent) onEvent({ type: 'anneal', step: 0, losses: l, H: Float32Array.from(st.H), COL: Float32Array.from(st.COL), cams: C });
  }
  // tensors (objects matter: grads live on objects, as in PyTorch)
  let Hp = { name: 'H', data: st.H, grad: null, rg: true };
  let Cp = { name: 'COL', data: st.COL, grad: null, rg: true };
  const Optim = cfg.sgd ? SGD : AdamW;
  let optimizer = new Optim([Hp, Cp], 0.1 * (cfg.lrScale ?? 1));
  let lr;
  let prevType = blockType(cfg.block, 0);
  for (let j = 1; j < cfg.epochs; j++) {
    let change = false;
    const canSub = S * 2 <= cfg.maxStrips;
    if (j % cfg.subdivideEvery === 0 && canSub) {
      st.H = doubleRes(Hp.data, u); st.COL = doubleRes(Cp.data, S, 3);
      u *= 2; S *= 2; st.u = u;
      Hp = { name: 'H', data: st.H, grad: null, rg: false }; Cp = { name: 'COL', data: st.COL, grad: null, rg: false };
      change = true;
      if (onEvent) onEvent({ type: 'subdivide', step: j, strips: S });
    }
    // rays double `rayLead` steps before each subdivision that will actually happen
    const rayStep = j % cfg.subdivideEvery === cfg.subdivideEvery - cfg.rayLead && S * 2 <= cfg.maxStrips;
    if (rayStep) { numRays *= 2; change = true; if (onEvent) onEvent({ type: 'rays', step: j, numRays }); }
    if (rayStep || (j % cfg.subdivideEvery === 0 && canSub)) { build(); Hp.rg = true; Cp.rg = true; change = true; }
    if (cfg.annealEvery && j % cfg.annealEvery === 0 && cfg.simAnnealing) {
      st.H = Hp.data; st.COL = Cp.data;
      const l = simAnnealing(st, opts, rnd, cfg.annealSteps, cfg.annealT0, cfg.annealTmin, cfg.annealDecay);
      Hp = { name: 'H', data: st.H, grad: null, rg: true }; Cp = { name: 'COL', data: st.COL, grad: null, rg: true };
      change = true;
      if (onEvent) onEvent({ type: 'anneal', step: j, losses: l });
    }
    if (cfg.adaptivePlane && j % (cfg.planeEvery ?? 10) === 0) {
      if (cfg.adaptivePlane === 'hit') {
        const zs = hitHeights().map((z, q) => (z == null ? (planeZs ? planeZs[q] : planeZ) : z));
        if (!planeZs || zs.some((z, q) => Math.abs(z - planeZs[q]) > 1e-4)) { planeZs = zs; build(); change = true; }
      } else {
        const z = surfaceHeight(Hp.data);
        if (Math.abs(z - planeZ) > 1e-4) { planeZ = z; build(); change = true; }
      }
    }
    const type = blockType(cfg.block, j);
    if (type === 'C') { Hp.rg = false; Cp.rg = true; lr = 0.05; }
    else {
      // grad_descent.py re-creates COL as a fresh tensor on every height step (COL.detach().numpy())
      Hp.rg = true; Cp = { name: 'COL', data: Cp.data, grad: null, rg: false };
      if (j > 49) lr = 0.03; else if (lr === undefined) lr = 0.05;
    }
    if (type !== prevType) change = true;
    prevType = type;
    if (change) optimizer = new Optim([Hp, Cp], lr * (cfg.lrScale ?? 1));
    st.H = Hp.data; st.COL = Cp.data;
    const res = lossAndGrad(st, opts, Hp.rg, Cp.rg);
    if (!Number.isFinite(res.total)) throw new Error(`loss became ${res.total} at step ${j}`);
    if (cfg.gradNoise) { const nr = cfg.gradNoiseRnd; for (const g of [res.gH, res.gC]) if (g) for (let i = 0; i < g.length; i++) g[i] *= 1 + cfg.gradNoise * (nr() - 0.5); }
    optimizer.zeroGrad(cfg.torch19);
    if (Hp.rg) { if (!Hp.grad) Hp.grad = new Float32Array(Hp.data.length); for (let i = 0; i < Hp.grad.length; i++) Hp.grad[i] = f32(Hp.grad[i] + res.gH[i]); }
    if (Cp.rg) { if (!Cp.grad) Cp.grad = new Float32Array(Cp.data.length); for (let i = 0; i < Cp.grad.length; i++) Cp.grad[i] = f32(Cp.grad[i] + res.gC[i]); }
    optimizer.step();
    // (fixed release) keep heights strictly inside the barrier bounds; see grad_descent.py
    if (cfg.barrier && cfg.keepInBounds !== false) { const hm = cfg.barrierMax ?? 1.5, d = Hp.data, hi = f32(hm), lo = f32(0.1); for (let i = 0; i < d.length; i++) { if (d[i] >= hi) d[i] = f32(hm - 1e-4); else if (d[i] <= lo) d[i] = f32(0.1 + 1e-4); } }
    const rec = { step: j, type, mse: res.mse, perView: res.mse / C.length, barrier: res.barrier, neighbor: res.neighbor, strips: S, numRays };
    history.push(rec);
    yield { ...rec, planeZ, planeZs, renders: res.renders, desired: st.desired, H: Hp.data, u, reps, COL: Cp.data, S, cams: C };
  }
  return { history, H: Hp.data, u, reps, COL: Cp.data, S, cams: C, numRays, desired: st.desired, planeZ, planeZs };
}

export function runToEnd(cfg, cams, images, onEvent) {
  const it = gradDescent(cfg, cams, images, onEvent);
  let last = null, r;
  const hist = [];
  while (!(r = it.next()).done) { last = r.value; hist.push({ step: last.step, mse: last.mse, perView: last.perView }); }
  return { final: r.value, last, hist };
}

// ---------------------------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------------------------
// Viewing direction (the direction rays travel) from azimuth/elevation in degrees. Elevation is
// measured from the surface plane, azimuth 0 = looking along +x.
export function camFromAzEl(azDeg, elDeg) {
  const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
  return [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), -Math.sin(el)];
}
