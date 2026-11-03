// three.js viewer for an optimized heightfield (bars on the unit square, z up, same coordinates as the
// optimizer). Optional "post-optimization projection": each desired image is projected along its camera
// direction onto the faces that camera can see (traced through the heightfield per pixel), later cameras
// overwriting earlier ones, as in create_mesh.py's projection step.
import * as THREE from '../vendor/three.module.min.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

const VERT = /* glsl */ `
in vec3 barColor;
out vec3 vColor; out vec3 vPos; out vec3 vNormal;
void main() {
  vColor = barColor; vPos = position; vNormal = normal;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp float; precision highp int; precision highp sampler2D;
in vec3 vColor; in vec3 vPos; in vec3 vNormal;
uniform sampler2D uHeights; uniform int uS; uniform int uT; uniform int uNC; uniform vec3 uDir[4]; uniform float uZc[4];
uniform sampler2D uImg0; uniform sampler2D uImg1; uniform sampler2D uImg2; uniform sampler2D uImg3;
uniform int uProject; uniform float uShade; uniform vec3 uLight; uniform int uHighlight;
out vec4 outColor;
float hAt(ivec2 c) { ivec2 m = ivec2(c.x % uS, c.y % uS); return texelFetch(uHeights, ivec2(m.y, m.x), 0).r; }
bool visible(vec3 P, vec3 d) {
  vec2 dir = -d.xy; float L = length(dir);
  if (L < 1e-7) return true;
  dir /= L; float slope = -d.z / L; float S = float(uS);
  vec2 p = P.xy; float z = P.z;
  ivec2 c = ivec2(floor(p * S));
  ivec2 st = ivec2(dir.x > 0.0 ? 1 : -1, dir.y > 0.0 ? 1 : -1);
  vec2 tD = vec2(abs(dir.x) > 1e-9 ? abs(1.0 / (dir.x * S)) : 1e9, abs(dir.y) > 1e-9 ? abs(1.0 / (dir.y * S)) : 1e9);
  vec2 nb = (vec2(c) + vec2(st.x > 0 ? 1.0 : 0.0, st.y > 0 ? 1.0 : 0.0)) / S;
  vec2 tM = vec2(abs(dir.x) > 1e-9 ? (nb.x - p.x) / dir.x : 1e9, abs(dir.y) > 1e-9 ? (nb.y - p.y) / dir.y : 1e9);
  float t = 0.0;
  for (int i = 0; i < 400; i++) {
    if (c.x < 0 || c.y < 0 || c.x >= uS * uT || c.y >= uS * uT) return true;
    float zE = z + slope * t;
    if (zE > 4.0) return true;
    if (zE < hAt(c) - 1e-5) return false;
    if (tM.x < tM.y) { t = tM.x; tM.x += tD.x; c.x += st.x; } else { t = tM.y; tM.y += tD.y; c.y += st.y; }
  }
  return true;
}
vec3 img(int q, vec2 uv) {
  if (q == 0) return texture(uImg0, uv).rgb;
  if (q == 1) return texture(uImg1, uv).rgb;
  if (q == 2) return texture(uImg2, uv).rgb;
  return texture(uImg3, uv).rgb;
}
void main() {
  vec3 col = clamp(vColor, 0.0, 1.0);
  vec3 N = normalize(vNormal);
  if (uProject == 1) {
    // projection (reproject.py): cameras that see this point, weighted by how directly they see it
    vec3 P = vPos + N * 3e-4;
    vec3 acc = vec3(0.0); float wsum = 0.0;
    for (int q = 0; q < 4; q++) {
      if (q >= uNC) break;
      vec3 d = uDir[q];
      float w = -dot(N, d);
      if (w <= 0.0) continue;               // face turned away from this camera
      if (!visible(P, d)) continue;
      float t = (uZc[q] - vPos.z) / d.z;    // where this point's camera ray crosses the camera's pixel plane
      vec3 Pp = vPos + t * d;
      if (uT == 1 && (Pp.x < 0.0 || Pp.x >= 1.0 || Pp.y < 0.0 || Pp.y >= 1.0)) continue;
      acc += w * img(q, vec2(fract(Pp.y), fract(Pp.x))); wsum += w;
    }
    if (wsum > 0.0) col = acc / wsum;
  }
  float sh = mix(1.0, 0.62 + 0.38 * max(dot(N, uLight), 0.0) + 0.1 * N.z, uShade);
  outColor = vec4(col * sh, 1.0);
}`;

const CAM_COLORS = ['#e8590c', '#1c7ed6', '#2b8a3e', '#ae3ec9'];

function labelSprite(text, color) {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(32, 32, 28, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#fff'; ctx.font = 'bold 34px "Google Sans", Arial, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 32, 34);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(0.12, 0.12, 1);
  sp.renderOrder = 10;
  return sp;
}

export class HeightfieldViewer {
  constructor(container, { background = '#f4f5f7' } = {}) {
    this.el = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // shader writes display values directly
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(background);
    this.persp = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
    this.persp.up.set(0, 0, 1);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 50);
    this.camera = this.persp;
    this.controls = new OrbitControls(this.persp, this.renderer.domElement);
    this.controls.target.set(0.5, 0.5, 0.35);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.6; this.controls.maxDistance = 8;
    this.controls.addEventListener('change', () => this.render());
    this.renderer.domElement.addEventListener('pointerdown', () => { if (this.camera === this.ortho) this.freeView(true); });
    this.renderer.domElement.addEventListener('wheel', () => { if (this.camera === this.ortho) this.freeView(true); }, { passive: true });
    // The mouse wheel scrolls the page unless Ctrl/⌘ is held (a trackpad pinch sends Ctrl) or the viewer was clicked.
    this.engaged = false;
    container.addEventListener('pointerdown', () => { this.engaged = true; }, true);
    container.addEventListener('pointerleave', () => { this.engaged = false; });
    this.wheelHint = Object.assign(document.createElement('div'), { className: 'wheel-hint', textContent: 'To zoom, click the view first or hold Ctrl (⌘)' });
    container.appendChild(this.wheelHint);
    container.addEventListener('wheel', (e) => {
      if (this.engaged || e.ctrlKey || e.metaKey) return;
      e.stopPropagation();
      this.wheelHint.classList.add('show');
      clearTimeout(this.wheelHintTimer);
      this.wheelHintTimer = setTimeout(() => this.wheelHint.classList.remove('show'), 1200);
    }, { capture: true, passive: true });

    this.uniforms = {
      uHeights: { value: null }, uS: { value: 1 }, uT: { value: 1 }, uNC: { value: 0 }, uZc: { value: [1, 1, 1, 1] },
      uDir: { value: [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()] },
      uImg0: { value: null }, uImg1: { value: null }, uImg2: { value: null }, uImg3: { value: null },
      uProject: { value: 0 }, uShade: { value: 1 }, uLight: { value: new THREE.Vector3(0.4, -0.55, 0.73).normalize() }, uHighlight: { value: -1 },
    };
    const blank = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    blank.needsUpdate = true;
    for (let q = 0; q < 4; q++) this.uniforms[`uImg${q}`].value = blank;
    this.material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms, glslVersion: THREE.GLSL3, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    this.scene.add(this.mesh);
    this.base = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 0.04), new THREE.MeshBasicMaterial({ color: '#2b2f36' }));
    this.scene.add(this.base);
    this.tiles = 1;
    this.camGroup = new THREE.Group();
    this.scene.add(this.camGroup);

    this.view = -1;
    this.cams = [];
    this.layout();
    let rq = false; // resize on the next frame, not inside the observer callback (see onWidth in widgets.js)
    new ResizeObserver(() => { if (!rq) { rq = true; requestAnimationFrame(() => { rq = false; this.resize(); }); } }).observe(container);
    this.resize();
  }

  layout() {
    const T = this.tiles || 1;
    this.base.scale.set(T + 0.04, T + 0.04, 1);
    this.base.position.set(T / 2, T / 2, -0.0201);
    this.controls.target.set(T / 2, T / 2, 0.35);
    this.controls.maxDistance = 8 * T;
    this.setOrbit(225, 34, 3.6 * Math.max(1, T * 0.72));
    if (this.cams && this.cams.length) this.setCameras(this.cams);
  }

  setOrbit(azDeg, elDeg, dist) {
    const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180, t = this.controls.target;
    this.persp.position.set(t.x + dist * Math.cos(el) * Math.cos(az), t.y + dist * Math.cos(el) * Math.sin(az), t.z + dist * Math.sin(el));
    this.persp.lookAt(t);
    this.controls.update();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.persp.aspect = w / h;
    this.persp.updateProjectionMatrix();
    if (this.view >= 0) this.setView(this.view);
    this.render();
  }

  // H: unit heights (u*u, repeated reps x reps), COL: S*S*3
  setSurface(H, u, reps, COL, S, tiles = this.tiles || 1) {
    if (tiles !== this.tiles) { this.tiles = tiles; this.layout(); }
    const nb = S * S * tiles * tiles, nv = nb * 20;
    const pos = new Float32Array(nv * 3), nor = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
    const idx = new Uint32Array(nb * 30);
    const hs = new Float32Array(S * S);
    let v = 0, k = 0;
    const quad = (p, n, c) => {
      for (let j = 0; j < 4; j++) { pos.set(p[j], 3 * (v + j)); nor.set(n, 3 * (v + j)); col.set(c, 3 * (v + j)); }
      idx.set([v, v + 1, v + 2, v, v + 2, v + 3], k);
      v += 4; k += 6;
    };
    for (let A = 0; A < S * tiles; A++) {
      for (let B = 0; B < S * tiles; B++) {
        const a = A % S, b = B % S;
        const h = H[(a % u) * u + (b % u)];
        hs[a * S + b] = h;
        const x0 = A / S, x1 = (A + 1) / S, y0 = B / S, y1 = (B + 1) / S;
        const c = [COL[(a * S + b) * 3], COL[(a * S + b) * 3 + 1], COL[(a * S + b) * 3 + 2]];
        quad([[x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h]], [0, 0, 1], c);
        quad([[x0, y0, 0], [x0, y1, 0], [x0, y1, h], [x0, y0, h]], [-1, 0, 0], c);
        quad([[x1, y0, 0], [x1, y0, h], [x1, y1, h], [x1, y1, 0]], [1, 0, 0], c);
        quad([[x0, y0, 0], [x0, y0, h], [x1, y0, h], [x1, y0, 0]], [0, -1, 0], c);
        quad([[x0, y1, 0], [x1, y1, 0], [x1, y1, h], [x0, y1, h]], [0, 1, 0], c);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
    g.setAttribute('barColor', new THREE.BufferAttribute(col, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    this.mesh.geometry.dispose();
    this.mesh.geometry = g;
    if (this.uniforms.uHeights.value) this.uniforms.uHeights.value.dispose();
    const tex = new THREE.DataTexture(hs, S, S, THREE.RedFormat, THREE.FloatType);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.needsUpdate = true;
    this.uniforms.uHeights.value = tex;
    this.uniforms.uS.value = S;
    this.uniforms.uT.value = tiles;
    this.maxH = Math.max(1, ...hs);
    this.render();
  }

  // cams: viewing directions; images: desired appearance images {width,height,data RGB float}
  setCameras(cams, images = null) {
    this.cams = cams.map((c) => { const l = Math.hypot(...c); return c.map((x) => x / l); });
    this.uniforms.uNC.value = Math.min(4, cams.length);
    this.cams.slice(0, 4).forEach((d, q) => this.uniforms.uDir.value[q].set(d[0], d[1], d[2]));
    if (images) {
      images.slice(0, 4).forEach((im, q) => {
        const n = im.width * im.height, data = new Uint8Array(n * 4);
        for (let i = 0; i < n; i++) { for (let c = 0; c < 3; c++) data[4 * i + c] = Math.max(0, Math.min(255, Math.round(im.data[3 * i + c] * 255))); data[4 * i + 3] = 255; }
        const t = new THREE.DataTexture(data, im.width, im.height, THREE.RGBAFormat);
        t.minFilter = THREE.LinearFilter; t.magFilter = THREE.NearestFilter; t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
        t.needsUpdate = true;
        const old = this.uniforms[`uImg${q}`].value;
        if (old && old.image && old.image.width > 1) old.dispose();
        this.uniforms[`uImg${q}`].value = t;
      });
    }
    // camera arrows (free the old labels' textures and the materials; the arrow geometries are shared by three.js)
    this.camGroup.traverse((o) => {
      if (o.isSprite) { o.material.map.dispose(); o.material.dispose(); }
      else if (o.isLine || o.isMesh) o.material.dispose();
    });
    this.camGroup.clear();
    this.cams.forEach((d, q) => {
      const dir = new THREE.Vector3(...d);
      const T = this.tiles || 1;
      const from = new THREE.Vector3(T / 2, T / 2, 0.35).addScaledVector(dir, -1.05 * Math.max(1, T * 0.8));
      const arrow = new THREE.ArrowHelper(dir, from, 0.42 * Math.max(1, T * 0.7), new THREE.Color(CAM_COLORS[q % 4]), 0.09, 0.05);
      arrow.line.material.depthTest = false; arrow.cone.material.depthTest = false;
      this.camGroup.add(arrow);
      const lab = labelSprite(String(q + 1), CAM_COLORS[q % 4]);
      lab.position.copy(from).addScaledVector(dir, -0.08);
      this.camGroup.add(lab);
    });
    // a camera view follows its camera; if that camera was removed, go back to orbiting
    if (this.view >= this.cams.length) this.freeView();
    else if (this.view >= 0) this.setView(this.view);
    this.render();
  }

  // Post-optimization projection: show the baked, textured mesh (exactly what the download contains)
  setProjection(on) { this.projectionOn = on; this._applyProjection(); }
  setTextured(baked) {
    this._dropTexMesh();
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(baked.mesh.pos, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(baked.mesh.nor, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(baked.mesh.uv, 2));
    g.setIndex(new THREE.BufferAttribute(baked.mesh.idx, 1));
    const tex = new THREE.DataTexture(baked.atlas.data, baked.atlas.width, baked.atlas.height, THREE.RGBAFormat);
    tex.flipY = false; tex.magFilter = THREE.NearestFilter; tex.minFilter = THREE.LinearFilter; tex.needsUpdate = true;
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, side: THREE.DoubleSide,
      uniforms: { map: { value: tex }, uShade: this.uniforms.uShade, uLight: this.uniforms.uLight },
      vertexShader: `out vec2 vUv; out vec3 vN; void main() { vUv = uv; vN = normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `precision highp float; uniform sampler2D map; uniform float uShade; uniform vec3 uLight; in vec2 vUv; in vec3 vN; out vec4 o;
        void main() { vec3 N = normalize(vN); vec3 c = texture(map, vUv).rgb; float sh = mix(1.0, 0.62 + 0.38 * max(dot(N, uLight), 0.0) + 0.1 * N.z, uShade); o = vec4(c * sh, 1.0); }`,
    });
    this.texMesh = new THREE.Mesh(g, mat);
    this.scene.add(this.texMesh);
    this._applyProjection();
  }
  clearTextured() { this._dropTexMesh(); this._applyProjection(); }
  // remove the projected mesh and free its geometry, texture (a shader uniform) and material
  _dropTexMesh() {
    if (!this.texMesh) return;
    this.scene.remove(this.texMesh);
    this.texMesh.geometry.dispose(); this.texMesh.material.uniforms.map.value.dispose(); this.texMesh.material.dispose();
    this.texMesh = null;
  }
  _applyProjection() {
    const show = !!(this.projectionOn && this.texMesh);
    if (this.texMesh) this.texMesh.visible = show;
    this.mesh.visible = !show;
    if (this.view >= 0) this.setView(this.view);
    this.render();
  }
  setCameraPlane(z) { this.uniforms.uZc.value = [z, z, z, z]; this.render(); }
  setShading(on) { this.uniforms.uShade.value = on ? 1 : 0; this.render(); }
  setArrows(on) { this.camGroup.visible = on; this.render(); }

  // Look exactly along camera q (orthographic), upright (world z up), as a person at that camera sees it.
  setView(q) {
    const d = this.cams[q];
    if (!d) return;
    this.view = q;
    const dir = new THREE.Vector3(...d).normalize();
    const T = this.tiles || 1;
    const c = new THREE.Vector3(T / 2, T / 2, 0.5);
    this.ortho.up.set(0, 0, 1);
    this.ortho.position.copy(c).addScaledVector(dir, -6);
    this.ortho.lookAt(c);
    this.ortho.updateMatrixWorld();
    // frame the surface that is shown (the projected mesh is sliced thinner than the optimized one)
    const inv = this.ortho.matrixWorldInverse;
    let zlo = 0, zhi = this.maxH || 1;
    if (this.texMesh && this.texMesh.visible) {
      const box = new THREE.Box3().setFromObject(this.texMesh);
      if (Number.isFinite(box.max.z)) { zlo = Math.min(0, box.min.z); zhi = box.max.z; }
    }
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const x of [0, T]) for (const y of [0, T]) for (const z of [zlo, zhi]) {
      const p = new THREE.Vector3(x, y, z).applyMatrix4(inv);
      xmin = Math.min(xmin, p.x); xmax = Math.max(xmax, p.x); ymin = Math.min(ymin, p.y); ymax = Math.max(ymax, p.y);
    }
    const w = this.el.clientWidth || 1, h = this.el.clientHeight || 1, asp = w / h;
    let hw = (xmax - xmin) / 2 * 1.04, hh = (ymax - ymin) / 2 * 1.04;
    if (hw / hh > asp) hh = hw / asp; else hw = hh * asp;
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    Object.assign(this.ortho, { left: cx - hw, right: cx + hw, top: cy + hh, bottom: cy - hh });
    this.ortho.updateProjectionMatrix();
    this.camera = this.ortho;
    this.controls.enabled = false;
    this.camGroup.visible = false;
    this.render();
    if (this.onViewChange) this.onViewChange(q);
  }

  freeView(keepDirection = false) {
    if (keepDirection && this.view >= 0) {
      const d = this.cams[this.view];
      const t = this.controls.target;
      const D = 2.7 * Math.max(1, (this.tiles || 1) * 0.72);
      this.persp.position.set(t.x - D * d[0], t.y - D * d[1], t.z - D * d[2]);
      this.persp.lookAt(t);
    }
    this.view = -1;
    this.camera = this.persp;
    this.controls.enabled = true;
    this.controls.update();
    this.camGroup.visible = this.arrowsOn !== false;
    this.render();
    if (this.onViewChange) this.onViewChange(-1);
  }

  render() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.renderer.render(this.scene, this.camera); });
  }

  snapshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL('image/png'); }
}

export { CAM_COLORS };
