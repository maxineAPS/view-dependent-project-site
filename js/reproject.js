// Post-optimization projection (paper Sec. 5): the desired images are projected back onto the
// optimized heightfield. Each surface point P with normal n is coloured from the cameras that see it:
//   - its colour in camera q's image is where P's camera ray crosses the camera's pixel plane z = zc
//     (the same pixel mapping the optimizer uses);
//   - rule 'overwrite': the last camera that sees P wins (what the original Blender script does);
//   - rule 'best':      the camera that sees P most head-on wins;
//   - rule 'average':   cameras are blended with weights max(0, -n.d_q) (visible cameras only), which
//                       minimizes the squared colour error summed over all camera images for that point.
// Points no camera sees keep the optimized bar colour.

export function heightsFull(H, u, S) {
  const Hf = new Float32Array(S * S);
  for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) Hf[a * S + b] = H[(a % u) * u + (b % u)];
  return Hf;
}

const norm = (d) => { const l = Math.hypot(d[0], d[1], d[2]); return [d[0] / l, d[1] / l, d[2] / l]; };

// Is point P (slightly off its face) visible from viewing direction d? Walk the grid towards the camera.
export function visibleFrom(Hf, S, P, d) {
  const L = Math.hypot(d[0], d[1]);
  if (L < 1e-9) return true;
  const ux = -d[0] / L, uy = -d[1] / L, s = -d[2] / L; // horizontal direction towards the camera, rise per unit
  let cx = Math.floor(P[0] * S), cy = Math.floor(P[1] * S);
  const stx = ux > 0 ? 1 : -1, sty = uy > 0 ? 1 : -1;
  const tdx = Math.abs(ux) > 1e-12 ? 1 / (Math.abs(ux) * S) : Infinity, tdy = Math.abs(uy) > 1e-12 ? 1 / (Math.abs(uy) * S) : Infinity;
  let tmx = Math.abs(ux) > 1e-12 ? ((cx + (stx > 0 ? 1 : 0)) / S - P[0]) / ux : Infinity;
  let tmy = Math.abs(uy) > 1e-12 ? ((cy + (sty > 0 ? 1 : 0)) / S - P[1]) / uy : Infinity;
  let t = 0;
  for (let i = 0; i < 4 * S + 4; i++) {
    if (cx < 0 || cy < 0 || cx >= S || cy >= S) return true;
    const z = P[2] + s * t;
    if (z < Hf[cx * S + cy] - 1e-7) return false;
    if (tmx < tmy) { t = tmx; tmx += tdx; cx += stx; } else { t = tmy; tmy += tdy; cy += sty; }
  }
  return true;
}

// First surface hit of the ray o + t d (d pointing down). Returns {P, n, a, b} or null.
export function traceHit(Hf, S, o, d) {
  const L = Math.hypot(d[0], d[1]);
  const ux = d[0] / L, uy = d[1] / L, s = -d[2] / L;
  // start where the ray enters the bounding box [0,1]^2 x [0, zTop]
  const zTop = 4;
  let t0 = (o[2] - zTop) / s; // horizontal distance from o's xy to the zTop crossing
  let x = o[0] + ux * t0, y = o[1] + uy * t0, z = zTop;
  // clip to the square
  let tin = 0, tout = Infinity;
  for (const [p, u] of [[x, ux], [y, uy]]) {
    if (Math.abs(u) < 1e-12) { if (p < 0 || p > 1) return null; continue; }
    const ta = (0 - p) / u, tb = (1 - p) / u;
    tin = Math.max(tin, Math.min(ta, tb)); tout = Math.min(tout, Math.max(ta, tb));
  }
  if (tin > tout) return null;
  x += ux * tin; y += uy * tin; z -= s * tin;
  let cx = Math.min(S - 1, Math.max(0, Math.floor(x * S + 1e-9 * Math.sign(ux)))), cy = Math.min(S - 1, Math.max(0, Math.floor(y * S + 1e-9 * Math.sign(uy))));
  // entering face normal (if we enter from a side of the square)
  let n = [0, 0, 1];
  const onX = Math.abs(x) < 1e-9 || Math.abs(x - 1) < 1e-9, onY = Math.abs(y) < 1e-9 || Math.abs(y - 1) < 1e-9;
  if (onX) n = [-Math.sign(ux), 0, 0]; else if (onY) n = [0, -Math.sign(uy), 0];
  const stx = ux > 0 ? 1 : -1, sty = uy > 0 ? 1 : -1;
  for (let i = 0; i < 4 * S + 4; i++) {
    const h = Hf[cx * S + cy];
    if (z <= h) return { P: [x, y, z], n, a: cx, b: cy };   // entered this cell below its top: side hit
    const tx = Math.abs(ux) > 1e-12 ? ((cx + (stx > 0 ? 1 : 0)) / S - x) / ux : Infinity;
    const ty = Math.abs(uy) > 1e-12 ? ((cy + (sty > 0 ? 1 : 0)) / S - y) / uy : Infinity;
    const te = Math.min(tx, ty);
    if (z - s * te <= h) { // reaches the top inside this cell
      const tt = (z - h) / s;
      return { P: [x + ux * tt, y + uy * tt, h], n: [0, 0, 1], a: cx, b: cy };
    }
    x += ux * te; y += uy * te; z -= s * te;
    if (tx < ty) { cx += stx; n = [-stx, 0, 0]; } else { cy += sty; n = [0, -sty, 0]; }
    if (cx < 0 || cy < 0 || cx >= S || cy >= S) return null; // passes over the surface
  }
  return null;
}

// colour of surface point P (normal n, on bar (a,b)) under a projection rule
export function shade(P, n, a, b, ctx, rule) {
  const { Hf, S, cams, images, zc, COL } = ctx;
  let acc = [0, 0, 0], wsum = 0, bestW = 0, bestCol = null, lastCol = null;
  const Q = [P[0] + n[0] * 1e-6, P[1] + n[1] * 1e-6, P[2] + n[2] * 1e-6];
  for (let q = 0; q < cams.length; q++) {
    const d = cams[q];
    const w = -(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]);
    if (w <= 0 || !visibleFrom(Hf, S, Q, d)) continue;
    const t = (zc[q] - P[2]) / d[2];
    const px = P[0] + t * d[0], py = P[1] + t * d[1];
    if (px < 0 || px >= 1 || py < 0 || py >= 1) continue;
    const im = images[q];
    const r = Math.min(im.n - 1, Math.floor(px * im.n)), c = Math.min(im.n - 1, Math.floor(py * im.n));
    const i = (r * im.n + c) * 3, col = [im.data[i], im.data[i + 1], im.data[i + 2]];
    for (let k = 0; k < 3; k++) acc[k] += w * col[k];
    wsum += w; lastCol = col;
    if (w > bestW) { bestW = w; bestCol = col; }
  }
  if (wsum === 0) { const i = (a * S + b) * 3; return [COL[i], COL[i + 1], COL[i + 2]].map((v) => Math.min(1, Math.max(0, v))); }
  if (rule === 'average') return acc.map((v) => v / wsum);
  if (rule === 'best') return bestCol;
  return lastCol; // 'overwrite': last visible camera wins
}

export { norm };

// ---------------------------------------------------------------------------------------------
// Textured, closed mesh export: a port of reproject.py (build_faces + export_surface).
// ---------------------------------------------------------------------------------------------
export const SEG = 0.1; // vertical segment height (create_mesh.py: vertical_segment_height = 0.1)

// Faces of the closed mesh: bar tops, the exposed parts of the bar sides split into vertical segments at
// multiples of SEG (the post-optimization projection of the paper works per segment), and the bottom.
export function buildFaces(Hf, S) {
  const w = 1 / S, faces = [];
  for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) {
    const h = Hf[a * S + b], x0 = a * w, y0 = b * w;
    faces.push({ o: [x0, y0, h], eu: [w, 0, 0], ev: [0, w, 0], n: [0, 0, 1], ab: [a, b] });
    for (const [nx, ny, na, nb] of [[-1, 0, a - 1, b], [1, 0, a + 1, b], [0, -1, a, b - 1], [0, 1, a, b + 1]]) {
      const low = na >= 0 && nb >= 0 && na < S && nb < S ? Hf[na * S + nb] : 0;
      if (h <= low) continue;
      let z0 = low;
      while (z0 < h - 1e-9) {
        const z1 = Math.min(h, (Math.floor(z0 / SEG + 1e-9) + 1) * SEG);
        if (nx !== 0) faces.push({ o: [nx < 0 ? x0 : x0 + w, y0, z0], eu: [0, w, 0], ev: [0, 0, z1 - z0], n: [nx, 0, 0], ab: [a, b] });
        else faces.push({ o: [x0, ny < 0 ? y0 : y0 + w, z0], eu: [w, 0, 0], ev: [0, 0, z1 - z0], n: [0, ny, 0], ab: [a, b] });
        z0 = z1;
      }
    }
  }
  faces.push({ o: [0, 0, 0], eu: [1, 0, 0], ev: [0, 1, 0], n: [0, 0, -1], ab: [0, 0] });
  return faces;
}

// Camera weights of a whole segment: visible from the camera at the segment's centre (as Blender's
// "select visible faces"), times how directly the camera sees it.
export function segmentWeights(f, ctx) {
  const c = [0, 1, 2].map((k) => f.o[k] + 0.5 * f.eu[k] + 0.5 * f.ev[k] + f.n[k] * 1e-6);
  return ctx.cams.map((d) => { const w = -(f.n[0] * d[0] + f.n[1] * d[1] + f.n[2] * d[2]); return w > 0 && visibleFrom(ctx.Hf, ctx.S, c, d) ? w : 0; });
}

// colour of point P on a segment with the given camera weights (weighted average of the projected images)
export function shadeSegment(P, wts, ab, ctx) {
  const { S, cams, images, zc, COL } = ctx;
  let acc = [0, 0, 0], wsum = 0;
  for (let q = 0; q < cams.length; q++) {
    if (!wts[q]) continue;
    const d = cams[q], t = (zc[q] - P[2]) / d[2];
    const px = P[0] + t * d[0], py = P[1] + t * d[1];
    if (px < 0 || px >= 1 || py < 0 || py >= 1) continue;
    const im = images[q];
    const r = Math.min(im.n - 1, Math.floor(px * im.n)), c = Math.min(im.n - 1, Math.floor(py * im.n)), i = (r * im.n + c) * 3;
    acc[0] += wts[q] * im.data[i]; acc[1] += wts[q] * im.data[i + 1]; acc[2] += wts[q] * im.data[i + 2]; wsum += wts[q];
  }
  if (wsum === 0) { const i = (ab[0] * S + ab[1]) * 3; return [COL[i], COL[i + 1], COL[i + 2]].map((v) => Math.min(1, Math.max(0, v))); }
  return acc.map((v) => v / wsum);
}

// Slice the surface thin before projecting (same as reproject.py): lower every bar and the pixel planes so the
// shortest bar is `base` tall. This does not change which bars hide which from any camera.
export function exportSurface(Hf0, S, COL, cams, images, zc0, name = 'surface', texels = 6, base = 0.05) {
  let shift = 0;
  if (base != null) { let mn = Infinity; for (const h of Hf0) mn = Math.min(mn, h); shift = Math.max(0, mn - base); }
  const Hf = Float32Array.from(Hf0, (h) => h - shift), zc = zc0.map((z) => z - shift);
  const faces = buildFaces(Hf, S);
  const len = (v) => Math.hypot(v[0], v[1], v[2]);
  const sizes = faces.map((f) => (f.n[2] === -1 ? [1, 1] : [Math.max(1, Math.round(len(f.eu) * S * texels)), Math.max(1, Math.ceil(len(f.ev) * S * texels - 1e-9))]));
  const W = 2048, pos = [];
  let x = 0, y = 0, rowH = 0;
  for (const [tw, th] of sizes) { if (x + tw + 2 > W) { x = 0; y += rowH + 2; rowH = 0; } pos.push([x + 1, y + 1]); x += tw + 2; rowH = Math.max(rowH, th); }
  const Hh = y + rowH + 2;
  const atlas = new Uint8ClampedArray(W * Hh * 4).fill(255);
  const ctx = { Hf, S, cams: cams.map(norm), images, zc, COL };
  faces.forEach((f, i) => {
    const [tw, th] = sizes[i], [px, py] = pos[i];
    const wts = segmentWeights(f, ctx);
    for (let jv = -1; jv <= th; jv++) for (let ju = -1; ju <= tw; ju++) {
      const s = Math.min(1, Math.max(0, (ju + 0.5) / tw)), t = Math.min(1, Math.max(0, (jv + 0.5) / th));
      const P = [0, 1, 2].map((k) => f.o[k] + s * f.eu[k] + t * f.ev[k]);
      const c = shadeSegment(P, wts, f.ab, ctx);
      const o = ((py + jv) * W + (px + ju)) * 4;
      atlas[o] = Math.round(c[0] * 255); atlas[o + 1] = Math.round(c[1] * 255); atlas[o + 2] = Math.round(c[2] * 255); atlas[o + 3] = 255;
    }
  });
  const v = [], vt = [], fl = [];
  const mpos = new Float32Array(faces.length * 12), muv = new Float32Array(faces.length * 8), mnor = new Float32Array(faces.length * 12), midx = new Uint32Array(faces.length * 6);
  let k = 1;
  faces.forEach((f, i) => {
    const [tw, th] = sizes[i], [px, py] = pos[i];
    let corners = [f.o, f.o.map((c, j) => c + f.eu[j]), f.o.map((c, j) => c + f.eu[j] + f.ev[j]), f.o.map((c, j) => c + f.ev[j])];
    let uvs = [[px, py], [px + tw, py], [px + tw, py + th], [px, py + th]];
    const cr = [f.eu[1] * f.ev[2] - f.eu[2] * f.ev[1], f.eu[2] * f.ev[0] - f.eu[0] * f.ev[2], f.eu[0] * f.ev[1] - f.eu[1] * f.ev[0]];
    if (cr[0] * f.n[0] + cr[1] * f.n[1] + cr[2] * f.n[2] < 0) { corners = corners.reverse(); uvs = uvs.reverse(); }
    // OBJ convention: y up (as written by Blender)
    corners.forEach((c, j) => { v.push(`v ${c[0].toFixed(6)} ${c[2].toFixed(6)} ${(-c[1]).toFixed(6)}`); vt.push(`vt ${(uvs[j][0] / W).toFixed(6)} ${(1 - uvs[j][1] / Hh).toFixed(6)}`); });
    fl.push(`f ${k}/${k} ${k + 1}/${k + 1} ${k + 2}/${k + 2} ${k + 3}/${k + 3}`);
    corners.forEach((c, j) => { mpos.set(c, i * 12 + j * 3); mnor.set(f.n, i * 12 + j * 3); muv.set([uvs[j][0] / W, uvs[j][1] / Hh], i * 8 + j * 2); });
    const base = i * 4; midx.set([base, base + 1, base + 2, base, base + 2, base + 3], i * 6);
    k += 4;
  });
  const obj = [`# view-dependent heightfield with projected images (reproject.py)`, `mtllib ${name}.mtl`, 'usemtl surface', ...v, ...vt, ...fl].join('\n') + '\n';
  const mtl = `newmtl surface\nKa 1 1 1\nKd 1 1 1\nmap_Kd ${name}.png\n`;
  return { obj, mtl, atlas: { width: W, height: Hh, data: atlas }, faces: faces.length, mesh: { pos: mpos, uv: muv, nor: mnor, idx: midx } };
}
