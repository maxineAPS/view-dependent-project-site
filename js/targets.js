// Desired-appearance images, drawn procedurally on a canvas (nothing precomputed).
// Images are {width, height, data: Float32Array RGB in [0,1]} like plt.imread(png)[..., :3].

const N = 64;
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

function fromCanvas(cv) {
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  const px = ctx.getImageData(0, 0, cv.width, cv.height).data;
  const data = new Float32Array(cv.width * cv.height * 3);
  for (let i = 0; i < cv.width * cv.height; i++) for (let c = 0; c < 3; c++) data[3 * i + c] = Math.fround(px[4 * i + c] / 255);
  return { width: cv.width, height: cv.height, data };
}
function draw(fn, size = N) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = size;
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  fn(ctx, size);
  return fromCanvas(cv);
}
const fill = (c) => (ctx, n) => { ctx.fillStyle = c; ctx.fillRect(0, 0, n, n); };
const stripes = (a, b, k = 4, dir = 'v') => (ctx, n) => {
  ctx.fillStyle = a; ctx.fillRect(0, 0, n, n); ctx.fillStyle = b;
  const w = n / (2 * k);
  for (let i = 0; i < k; i++) dir === 'v' ? ctx.fillRect((2 * i + 1) * w, 0, w, n) : ctx.fillRect(0, (2 * i + 1) * w, n, w);
};
const diag = (a, b) => (ctx, n) => {
  ctx.fillStyle = a; ctx.fillRect(0, 0, n, n); ctx.fillStyle = b;
  for (let k = -n; k < 2 * n; k += n / 2) { ctx.beginPath(); ctx.moveTo(k, 0); ctx.lineTo(k + n / 4, 0); ctx.lineTo(k + n / 4 - n, n); ctx.lineTo(k - n, n); ctx.fill(); }
};
const checker = (a, b, k = 4) => (ctx, n) => {
  const w = n / k;
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { ctx.fillStyle = (i + j) % 2 ? b : a; ctx.fillRect(i * w, j * w, w, w); }
};
const dots = (bg, fg) => (ctx, n) => {
  ctx.fillStyle = bg; ctx.fillRect(0, 0, n, n); ctx.fillStyle = fg;
  for (const [x, y] of [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]) { ctx.beginPath(); ctx.arc(x * n, y * n, n * 0.15, 0, Math.PI * 2); ctx.fill(); }
};
const letter = (ch, bg, fg) => (ctx, n) => {
  ctx.fillStyle = bg; ctx.fillRect(0, 0, n, n); ctx.fillStyle = fg;
  ctx.font = `800 ${Math.round(n * 0.86)}px "Google Sans", "Helvetica Neue", Arial, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(ch, n / 2, n * 0.54);
};
const heart = (bg, fg) => (ctx, n) => {
  ctx.fillStyle = bg; ctx.fillRect(0, 0, n, n); ctx.fillStyle = fg;
  ctx.beginPath();
  for (let t = 0; t <= Math.PI * 2 + 0.01; t += 0.05) {
    const x = 16 * Math.sin(t) ** 3, y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
    const px = n / 2 + (x * n) / 38, py = n / 2 - (y * n) / 38 + n * 0.02;
    t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  }
  ctx.fill();
};
const star = (bg, fg) => (ctx, n) => {
  ctx.fillStyle = bg; ctx.fillRect(0, 0, n, n); ctx.fillStyle = fg; ctx.beginPath();
  for (let i = 0; i < 10; i++) { const r = (i % 2 ? 0.2 : 0.46) * n, a = -Math.PI / 2 + (i * Math.PI) / 5; const x = n / 2 + r * Math.cos(a), y = n / 2 + 0.04 * n + r * Math.sin(a); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.fill();
};
const sun = (ctx, n) => { ctx.fillStyle = '#2f6fd6'; ctx.fillRect(0, 0, n, n); ctx.fillStyle = '#ffd23f'; ctx.beginPath(); ctx.arc(n / 2, n / 2, n * 0.3, 0, Math.PI * 2); ctx.fill(); };
const moon = (ctx, n) => { ctx.fillStyle = '#14163a'; ctx.fillRect(0, 0, n, n); ctx.fillStyle = '#f2f2e6'; ctx.beginPath(); ctx.arc(n / 2, n / 2, n * 0.3, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#14163a'; ctx.beginPath(); ctx.arc(n * 0.64, n * 0.42, n * 0.27, 0, Math.PI * 2); ctx.fill(); };
const noise = (seed) => (ctx, n) => {
  let a = seed >>> 0;
  const rnd = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const im = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) { for (let c = 0; c < 3; c++) im.data[4 * i + c] = Math.floor(rnd() * 256); im.data[4 * i + 3] = 255; }
  ctx.putImageData(im, 0, 0);
};

// Paper colours (Fig. 1 / Fig. 10 / Table 2 test cases)
export const PRESETS = {
  'stripes-rg': { label: 'Stripes red / green (Fig. 1)', draw: stripes('#e8202a', '#1e8a2e') },
  'stripes-bw': { label: 'Stripes black / white (Fig. 1)', draw: stripes('#111111', '#f4f4f4') },
  'stripes-yp': { label: 'Stripes yellow / purple (Fig. 1)', draw: stripes('#f7d23a', '#8a5cc2') },
  'diag-on': { label: 'Diagonal stripes navy / orange (printed result)', draw: diag('#0d1f7a', '#f39a5a') },
  'checker-rg': { label: 'Checks red / green (printed result)', draw: checker('#e0402a', '#2f9a3a') },
  'checker-bw': { label: 'Checks black / white (printed result)', draw: checker('#151530', '#f0f0f0') },
  'dots-yb': { label: 'Yellow dots on blue (printed result)', draw: dots('#3a3f9e', '#f7e36b') },
  sun: { label: 'Sun', draw: sun },
  moon: { label: 'Moon', draw: moon },
  heart: { label: 'Heart', draw: heart('#fde9ef', '#e0245e') },
  star: { label: 'Star', draw: star('#0f2a5c', '#ffd23f') },
  A: { label: 'Letter A', draw: letter('A', '#ffffff', '#1a1a1a') },
  B: { label: 'Letter B', draw: letter('B', '#1a1a1a', '#ffffff') },
  black: { label: 'Solid black', draw: fill('#000000') },
  white: { label: 'Solid white', draw: fill('#ffffff') },
  cyan: { label: 'Solid cyan', draw: fill('#00ffff') },
  magenta: { label: 'Solid magenta', draw: fill('#ff00ff') },
  yellow: { label: 'Solid yellow', draw: fill('#ffff00') },
  random: { label: 'Random noise', draw: noise(7) },
};

export function makeTarget(key) {
  const p = PRESETS[key];
  if (!p) throw new Error(`unknown target ${key}`);
  return draw(p.draw);
}
export const solid = (rgb) => ({ width: N, height: N, data: Float32Array.from({ length: N * N * 3 }, (_, i) => rgb[i % 3]) });
export const noiseImage = (seed) => draw(noise(seed));
export const stripesBW = () => draw(stripes('#000000', '#ffffff'));
export { hex };

// User image -> RGB float image (large images are first scaled to 1024 px, then the optimizer's
// nearest-neighbour resizes of image.py apply).
export async function fromFile(file) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const cv = document.createElement('canvas');
  cv.width = Math.max(1, Math.round(bmp.width * s)); cv.height = Math.max(1, Math.round(bmp.height * s));
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.drawImage(bmp, 0, 0, cv.width, cv.height);
  return fromCanvas(cv);
}

// The optimizer lays every desired image on the ground in one fixed orientation (image rows along x,
// columns along y), so a picture only looks upright from one side. These helpers rotate a picture so
// that it is upright for a camera with azimuth `az` (looking along (cos az, sin az) and down), and
// rotate stored images back for display. For multiples of 90 degrees they are exact permutations.
function frame(az) {
  const a = (az * Math.PI) / 180;
  return { up: [Math.cos(a), Math.sin(a)], right: [Math.sin(a), -Math.cos(a)] };
}
export function orient(img, az, N = 64) {
  const { up, right } = frame(az);
  const out = new Float32Array(N * N * 3);
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const ox = (r + 0.5) / N - 0.5, oy = (c + 0.5) / N - 0.5;
    const u = ox * right[0] + oy * right[1], v = ox * up[0] + oy * up[1];
    const pr = Math.min(img.height - 1, Math.max(0, Math.floor((0.5 - v) * img.height)));
    const pc = Math.min(img.width - 1, Math.max(0, Math.floor((u + 0.5) * img.width)));
    const s = (pr * img.width + pc) * 3, d = (r * N + c) * 3;
    out[d] = img.data[s]; out[d + 1] = img.data[s + 1]; out[d + 2] = img.data[s + 2];
  }
  return { width: N, height: N, data: out };
}
export function display(data, n, az, bg = 0.86) {
  const { up, right } = frame(az);
  const out = new Float32Array(n * n * 3);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const u = (j + 0.5) / n - 0.5, v = 0.5 - (i + 0.5) / n;
    const x = 0.5 + u * right[0] + v * up[0], y = 0.5 + u * right[1] + v * up[1];
    const d = (i * n + j) * 3;
    if (x < 0 || x >= 1 || y < 0 || y >= 1) { out[d] = out[d + 1] = out[d + 2] = bg; continue; }
    const s = (Math.floor(x * n) * n + Math.floor(y * n)) * 3;
    out[d] = data[s]; out[d + 1] = data[s + 1]; out[d + 2] = data[s + 2];
  }
  return out;
}

// Draw an RGB float image (n x n, row = optimizer x) onto a canvas.
export function paint(canvas, data, n) {
  if (canvas.width !== n) { canvas.width = n; canvas.height = n; }
  const ctx = canvas.getContext('2d');
  const im = ctx.createImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    for (let c = 0; c < 3; c++) im.data[4 * i + c] = Math.max(0, Math.min(255, Math.round(data[3 * i + c] * 255)));
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
