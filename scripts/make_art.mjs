#!/usr/bin/env node
// Paints the website's marbled backgrounds once, so visitors download a small image instead of computing noise.
// Domain-warped noise shaded as a liquid surface, adapted from the glass-folder-cards component (21st.dev).
//   node scripts/make_art.mjs && for f in web/art/*.png; do sips -s format jpeg -s formatOptions 76 "$f" --out "${f%.png}.jpg"; rm "$f"; done
import { mkdirSync, writeFileSync } from "node:fs";
import { crc32, deflateSync } from "node:zlib";

const ART = {
  // Dark enough for text on top: blue ink and red ink in black water.
  desk: { width: 1600, height: 1000, seed: 7, colors: ["#05070b", "#0a1320", "#123058", "#2a66c9", "#090a12", "#5f1518", "#c93a26", "#f08a62"], scale: 0.5, warp: 3.4, angle: -0.5, gloss: 0.32, veins: 0.5 },
  // The back of the folder is meant to be seen, so it's brighter.
  back: { width: 900, height: 1000, seed: 21, colors: ["#0a1020", "#1d3f73", "#4f94ff", "#bcd6ff", "#1b1119", "#b3261e", "#ff6a3d", "#ffd2b3"], scale: 0.72, warp: 3.2, angle: 0.8, gloss: 0.55, veins: 0.4 },
};

const hexToRgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeFbm(seed) {
  const r = rng(seed);
  const perm = new Uint8Array(512);
  const grad = new Float32Array(512);
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  for (let i = 0; i < 256; i++) {
    const a = r() * Math.PI * 2;
    grad[i * 2] = Math.cos(a);
    grad[i * 2 + 1] = Math.sin(a);
  }
  const noise = (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const X = xi & 255, Y = yi & 255;
    const h00 = perm[X + perm[Y]] * 2, h10 = perm[X + 1 + perm[Y]] * 2;
    const h01 = perm[X + perm[Y + 1]] * 2, h11 = perm[X + 1 + perm[Y + 1]] * 2;
    const d00 = grad[h00] * xf + grad[h00 + 1] * yf, d10 = grad[h10] * (xf - 1) + grad[h10 + 1] * yf;
    const d01 = grad[h01] * xf + grad[h01 + 1] * (yf - 1), d11 = grad[h11] * (xf - 1) + grad[h11 + 1] * (yf - 1);
    const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10), v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
    return d00 + u * (d10 - d00) + v * (d01 - d00 + u * (d11 - d01 - d10 + d00));
  };
  return (x, y, octaves) => {
    let sum = 0, amp = 0.5;
    for (let i = 0; i < octaves; i++) {
      sum += amp * noise(x, y);
      const nx = 1.6 * x + 1.2 * y + 7.1;
      y = -1.2 * x + 1.6 * y + 3.3;
      x = nx;
      amp *= 0.5;
    }
    return sum;
  };
}

function paint({ width: w, height: h, seed, colors, scale, warp, angle, gloss, veins }) {
  const stops = colors.map((c, i) => ({ t: i / (colors.length - 1), c: hexToRgb(c) }));
  const fbm = makeFbm(seed * 977 + 13);
  const r = rng(seed + 1);
  const ox = r() * 50, oy = r() * 50;
  const a = angle + (r() - 0.5) * 0.5, ca = Math.cos(a), sa = Math.sin(a), m = Math.min(w, h);
  const F = new Float32Array(w * h), Q = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = (x - w / 2) / m, v = (y - h / 2) / m;
      const px = (u * ca - v * sa) * scale + ox, py = (u * sa + v * ca) * scale + oy;
      const qx = fbm(px, py, 3), qy = fbm(px + 5.2, py + 1.3, 3);
      const rx = fbm(px + warp * qx + 1.7, py + warp * qy + 9.2, 3), ry = fbm(px + warp * qx + 8.3, py + warp * qy + 2.8, 3);
      F[y * w + x] = fbm(px + warp * rx, py + warp * ry, 4);
      Q[y * w + x] = Math.sqrt(qx * qx + qy * qy);
    }
  }
  const sample = Array.from(F).filter((_, i) => i % 97 === 0).sort((p, q) => p - q);
  const lo = sample[Math.floor(sample.length * 0.02)], hi = sample[Math.floor(sample.length * 0.98)], k = 1 / (hi - lo || 1);
  const slope = k * m * 0.012;
  const rgb = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const t = Math.min(1, Math.max(0, (F[i] - lo) * k));
      let j = 1;
      while (j < stops.length - 1 && stops[j].t < t) j++;
      const s0 = stops[j - 1], s1 = stops[j];
      let f = Math.min(1, Math.max(0, (t - s0.t) / (s1.t - s0.t || 1)));
      f = f * f * (3 - 2 * f);
      const gx = (F[x < w - 1 ? i + 1 : i] - F[x > 0 ? i - 1 : i]) * slope, gy = (F[y < h - 1 ? i + w : i] - F[y > 0 ? i - w : i]) * slope;
      const nd = (gx * 0.55 + gy * 0.65 + 0.52) / Math.sqrt(gx * gx + gy * gy + 1);
      const spec = Math.pow(Math.max(nd, 0), 24) * gloss * 255;
      const vein = veins * Math.max(0, 1 - Math.abs(Q[i] - 0.42) * 14) * 0.6;
      const lit = (0.82 + 0.28 * nd) * (1 + 0.035 * Math.sin(t * 150 + Q[i] * 24)) * (1 - vein);
      for (let c = 0; c < 3; c++) rgb[i * 3 + c] = Math.min(255, Math.max(0, (s0.c[c] + (s1.c[c] - s0.c[c]) * f) * lit + spec));
    }
  }
  return rgb;
}

function png(w, h, rgb) {
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type), data]);
    const out = Buffer.alloc(data.length + 12);
    out.writeUInt32BE(data.length, 0);
    body.copy(out, 4);
    out.writeUInt32BE(crc32(body), data.length + 8);
    return out;
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(w, 0);
  header.writeUInt32BE(h, 4);
  header.set([8, 2, 0, 0, 0], 8);
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) Buffer.from(rgb.buffer, y * w * 3, w * 3).copy(raw, y * (w * 3 + 1) + 1);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

const out = new URL("../web/art/", import.meta.url);
mkdirSync(out, { recursive: true });
for (const [name, recipe] of Object.entries(ART)) {
  writeFileSync(new URL(`${name}.png`, out), png(recipe.width, recipe.height, paint(recipe)));
  console.log(name);
}
