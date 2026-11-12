"use strict";

// ---------- Math ----------

const N_POINTS = 1000;
const xs = new Float64Array(N_POINTS);
const log10xs = new Float64Array(N_POINTS);
const powN = new Float64Array(N_POINTS); // xs^N
const powM = new Float64Array(N_POINTS); // xs^m
const N_POW = 3, M_POW = 1;

for (let i = 0; i < N_POINTS; i++) {
  const t = -5 + (10 * i) / (N_POINTS - 1); // LinRange(-5, 5, 1000)
  xs[i] = Math.pow(10, t);
  log10xs[i] = t;
  powN[i] = Math.pow(xs[i], N_POW);
  powM[i] = Math.pow(xs[i], M_POW);
}

// LogGaussian, identical to the Julia version (note: no 1/x factor, on purpose)
function logGaussian(x, mu, sigma) {
  const exponent = -0.5 * Math.pow((Math.log(x) - Math.log(mu)) / sigma, 2);
  const normalization = 1 / (sigma * Math.sqrt(2 * Math.PI));
  return normalization * Math.exp(exponent);
}

// y = normalized LogGaussian scaled by L
function gaussianCurve(mean, std, L, out) {
  let s = 0;
  for (let i = 0; i < N_POINTS; i++) {
    out[i] = logGaussian(xs[i], mean, std);
    s += out[i];
  }
  for (let i = 0; i < N_POINTS; i++) out[i] = (out[i] / s) * L;
}

// getCn from the notebook: ratio Cn2/Cn1 with dist = y1 + y2
function getCn(p, y1, y2) {
  gaussianCurve(p.mean1, p.std1, p.L1, y1);
  gaussianCurve(p.mean2, p.std2, p.L2, y2);

  let s1n = 0, s1m = 0, s2n = 0, s2m = 0;
  for (let i = 0; i < N_POINTS; i++) {
    const xN = powN[i], xM = powM[i];
    s1n += y1[i] * xN;
    s1m += y1[i] * xM;
    const d = y1[i] + y2[i];
    s2n += d * xN;
    s2m += d * xM;
  }
  const Cn1 = s1n / s1m;
  const Cn2 = s2n / s2m;
  return Cn2 / Cn1;
}

// makeCnsMatrix from the notebook
const SCAN_MEANS = [];
const SCAN_STDS = [];
const SCAN_N = 40;
for (let i = 0; i < SCAN_N; i++) {
  SCAN_MEANS.push(Math.pow(10, -4 + (8 * i) / (SCAN_N - 1))); // 10 .^ LinRange(-4, 4, 40)
  SCAN_STDS.push(0.1 + (1.9 * i) / (SCAN_N - 1));             // LinRange(0.1, 2, 40)
}

function makeCnsMatrix(p) {
  const y1 = new Float64Array(N_POINTS);
  const y2 = new Float64Array(N_POINTS);
  gaussianCurve(p.mean1, p.std1, p.L1, y1); // y1 doesn't change across the scan

  const mat = new Float64Array(SCAN_N * SCAN_N);
  for (let i = 0; i < SCAN_N; i++) {
    for (let j = 0; j < SCAN_N; j++) {
      gaussianCurve(SCAN_MEANS[i], SCAN_STDS[j], p.L2, y2);
      let s1n = 0, s1m = 0, s2n = 0, s2m = 0;
      for (let k = 0; k < N_POINTS; k++) {
        const xN = powN[k], xM = powM[k];
        s1n += y1[k] * xN;
        s1m += y1[k] * xM;
        const d = y1[k] + y2[k];
        s2n += d * xN;
        s2m += d * xM;
      }
      mat[i * SCAN_N + j] = (s2n / s2m) / (s1n / s1m);
    }
  }
  return mat;
}

// ---------- Colormaps ----------

function viridis(t) {
  // Polynomial approximation of matplotlib's viridis
  t = Math.max(0, Math.min(1, t));
  const c = [
    [0.2777, 0.0055, 0.3340],
    [0.1057, 0.6179, 0.4696],
    [0.9932, 0.9061, 0.1439],
  ];
  // quadratic blend through three anchor points (good enough visually)
  const anchors = [
    [68, 1, 84],
    [71, 45, 123],
    [59, 81, 139],
    [44, 113, 142],
    [33, 144, 141],
    [39, 173, 129],
    [92, 200, 99],
    [170, 220, 50],
    [253, 231, 37],
  ];
  const x = t * (anchors.length - 1);
  const i = Math.min(anchors.length - 2, Math.floor(x));
  const f = x - i;
  const a = anchors[i], b = anchors[i + 1];
  return [
    Math.round(a[0] + f * (b[0] - a[0])),
    Math.round(a[1] + f * (b[1] - a[1])),
    Math.round(a[2] + f * (b[2] - a[2])),
  ];
}

const BERLIN = ["#fdb462", "#b3de69", "#fccde5", "#80b1d3"];

// ---------- Plot helpers ----------

function axes(ctx, w, h, mL, mR, mT, mB, opts) {
  const pw = w - mL - mR, ph = h - mT - mB;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.strokeRect(mL, mT, pw, ph);

  const xToPx = opts.xLog
    ? (v) => mL + ((v - opts.xMin) / (opts.xMax - opts.xMin)) * pw
    : (v) => mL + ((v - opts.xMin) / (opts.xMax - opts.xMin)) * pw;
  const yToPx = (v) => mT + ph - ((v - opts.yMin) / (opts.yMax - opts.yMin)) * ph;

  // ticks
  ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif";
  if (opts.xLog) {
    ctx.textAlign = "center";
    for (let e = Math.ceil(opts.xMin); e <= Math.floor(opts.xMax); e += 2) {
      const px = xToPx(e);
      ctx.beginPath();
      ctx.moveTo(px, mT + ph);
      ctx.lineTo(px, mT + ph + 5);
      ctx.stroke();
      ctx.fillText("10", px - 4, mT + ph + 16);
      ctx.font = "8px sans-serif";
      ctx.fillText(String(e), px + 4, mT + ph + 13);
      ctx.font = "11px sans-serif";
    }
  } else {
    ctx.textAlign = "center";
    const step = (opts.xMax - opts.xMin) / 4;
    for (let k = 0; k <= 4; k++) {
      const v = opts.xMin + k * step;
      const px = xToPx(v);
      ctx.beginPath();
      ctx.moveTo(px, mT + ph);
      ctx.lineTo(px, mT + ph + 5);
      ctx.stroke();
      ctx.fillText(fmt(v), px, mT + ph + 16);
    }
  }
  ctx.textAlign = "right";
  const ystep = (opts.yMax - opts.yMin) / 4;
  for (let k = 0; k <= 4; k++) {
    const v = opts.yMin + k * ystep;
    const py = yToPx(v);
    ctx.beginPath();
    ctx.moveTo(mL - 5, py);
    ctx.lineTo(mL, py);
    ctx.stroke();
    ctx.fillText(fmt(v), mL - 8, py + 4);
  }

  // labels
  ctx.textAlign = "center";
  if (opts.xlabel) ctx.fillText(opts.xlabel, mL + pw / 2, h - 4);
  if (opts.ylabel) {
    ctx.save();
    ctx.translate(12, mT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(opts.ylabel, 0, 0);
    ctx.restore();
  }
  if (opts.title) {
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(opts.title, mL + pw / 2, 14);
  }

  // clip region for data
  ctx.save();
  ctx.beginPath();
  ctx.rect(mL, mT, pw, ph);
  ctx.clip();

  return { xToPx, yToPx, mL, mT, pw, ph, undoClip: () => ctx.restore() };
}

function fmt(v) {
  if (Math.abs(v) >= 1000) return v.toExponential(0);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(1);
}

function plotLine(ctx, a, xArr, yArr, color, width) {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  for (let i = 0; i < xArr.length; i++) {
    const px = a.xToPx(xArr[i]);
    const py = a.yToPx(yArr[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

// ---------- Plot 1: ligand amount vs tau ----------

function drawLines(p) {
  const canvas = document.getElementById("linesCanvas");
  const ctx = canvas.getContext("2d");
  const y1 = new Float64Array(N_POINTS);
  const y2 = new Float64Array(N_POINTS);
  gaussianCurve(p.mean1, p.std1, p.L1, y1);
  gaussianCurve(p.mean2, p.std2, p.L2, y2);
  const y3 = new Float64Array(N_POINTS);
  for (let i = 0; i < N_POINTS; i++) y3[i] = y1[i] + y2[i];

  let ymax = 0;
  for (let i = 0; i < N_POINTS; i++) ymax = Math.max(ymax, y3[i]);
  ymax = Math.max(ymax, 1e-12) * 1.1;

  const a = axes(ctx, canvas.width, canvas.height, 60, 15, 20, 40, {
    xLog: true,
    xMin: -5, xMax: 5,
    yMin: 0, yMax: ymax,
    xlabel: "τ",
    ylabel: "Ligand amount",
  });

  plotLine(ctx, a, log10xs, y3, "purple", 4);
  plotLine(ctx, a, log10xs, y2, "blue", 4);
  plotLine(ctx, a, log10xs, y1, "red", 4);
  a.undoClip();

  // legend
  const legend = [["purple", "total"], ["blue", "antagonist"], ["red", "agonist"]];
  legend.forEach(([color, label], i) => {
    const lx = canvas.width - 120, ly = 30 + i * 18;
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.lineTo(lx + 25, ly);
    ctx.stroke();
    ctx.fillStyle = "#333";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, lx + 30, ly + 4);
  });
}

// ---------- Plot 2: heatmap ----------

const HEAT_RANGE = [-4, 4];

function drawHeatmap(p, mat) {
  const canvas = document.getElementById("heatCanvas");
  const ctx = canvas.getContext("2d");

  const a = axes(ctx, canvas.width, canvas.height, 60, 60, 30, 40, {
    xLog: true,
    xMin: -4, xMax: 4,
    yMin: 0.1, yMax: 2,
    ylabel: "Std Dev 2",
    title: `Mean1: ${p.mean1.toPrecision(3)}, Std1: ${p.std1}`,
  });

  // log10(mat) mapped into an offscreen image, then scaled up
  const img = ctx.createImageData(SCAN_N, SCAN_N);
  for (let i = 0; i < SCAN_N; i++) {
    for (let j = 0; j < SCAN_N; j++) {
      const v = Math.log10(mat[i * SCAN_N + j]);
      const t = (v - HEAT_RANGE[0]) / (HEAT_RANGE[1] - HEAT_RANGE[0]);
      const [r, g, b] = viridis(t);
      // rows flip: row 0 of the image = top = largest std
      const row = SCAN_N - 1 - i;
      const idx = (row * SCAN_N + j) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = SCAN_N;
  off.height = SCAN_N;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, a.mL, a.mT, a.pw, a.ph);

  // current point (like scatter! in the notebook)
  const mx = a.xToPx(Math.log10(Math.pow(10, p.m2)));
  const my = a.yToPx(p.std2);
  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.arc(mx, my, 7.5, 0, 2 * Math.PI);
  ctx.fill();
  a.undoClip();

  // colorbar
  const cbX = canvas.width - 42, cbW = 16;
  for (let y = 0; y < a.ph; y++) {
    const t = 1 - y / a.ph;
    const [r, g, b] = viridis(t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(cbX, a.mT + y, cbW, 1);
  }
  ctx.strokeStyle = "#333";
  ctx.strokeRect(cbX, a.mT, cbW, a.ph);
  ctx.fillStyle = "#333";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(String(HEAT_RANGE[1]), cbX + cbW + 3, a.mT + 8);
  ctx.fillText("0", cbX + cbW + 3, a.mT + a.ph / 2 + 3);
  ctx.fillText(String(HEAT_RANGE[0]), cbX + cbW + 3, a.mT + a.ph);
  ctx.save();
  ctx.translate(canvas.width - 6, a.mT + a.ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("log10(Cn/Cn(0))", 0, 0);
  ctx.restore();
}

// ---------- Plot 3: slices of the heatmap ----------

const SLICE_INDICES = [0, 9, 19, 29]; // Julia used [1, 10, 20, 30] (1-based)

function drawSlices(p, mat) {
  const canvas = document.getElementById("slicesCanvas");
  const ctx = canvas.getContext("2d");

  let ymin = Infinity, ymax = -Infinity;
  for (const j of SLICE_INDICES) {
    for (let i = 0; i < SCAN_N; i++) {
      const v = Math.log10(mat[i * SCAN_N + j]);
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
  }
  if (!isFinite(ymin)) { ymin = -4; ymax = 4; }

  const a = axes(ctx, canvas.width, canvas.height, 60, 60, 20, 40, {
    xLog: true,
    xMin: -4, xMax: 4,
    yMin: ymin, yMax: ymax,
    xlabel: "Mean 2",
    ylabel: "log10(Cn)",
  });

  SLICE_INDICES.forEach((j, k) => {
    const color = BERLIN[k];
    const ys = new Float64Array(SCAN_N);
    for (let i = 0; i < SCAN_N; i++) ys[i] = Math.log10(mat[i * SCAN_N + j]);
    plotLine(ctx, a, logLog10ScanMeans(), ys, color, 3);

    // horizontal line marking this slice's std (like hlines! in the notebook)
    const py = a.yToPx(Math.max(a.yMin, Math.min(a.yMax, SCAN_STDS[j])));
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(a.mL, py);
    ctx.lineTo(a.mL + a.pw, py);
    ctx.stroke();
    ctx.setLineDash([]);
  });
  a.undoClip();
}

const SCAN_MEANS_LOG = SCAN_MEANS.map((m) => Math.log10(m));
function logLog10ScanMeans() { return SCAN_MEANS_LOG; }

// ---------- Wiring ----------

function readParams() {
  return {
    m1: parseFloat(document.getElementById("m1").value),
    std1: parseFloat(document.getElementById("std1").value),
    m2: parseFloat(document.getElementById("m2").value),
    std2: parseFloat(document.getElementById("std2").value),
    L1: parseFloat(document.getElementById("L1").value),
    L2: parseFloat(document.getElementById("L2").value),
  };
}

function render() {
  const raw = readParams();
  const p = {
    ...raw,
    mean1: Math.pow(10, raw.m1),
    mean2: Math.pow(10, raw.m2),
  };

  document.getElementById("m1-out").textContent = raw.m1.toFixed(1);
  document.getElementById("std1-out").textContent = raw.std1.toFixed(1);
  document.getElementById("m2-out").textContent = raw.m2.toFixed(1);
  document.getElementById("std2-out").textContent = raw.std2.toFixed(1);

  const mat = makeCnsMatrix(p);
  drawLines(p);
  drawHeatmap(p, mat);
  drawSlices(p, mat);
}

["m1", "std1", "m2", "std2", "L1", "L2"].forEach((id) => {
  document.getElementById(id).addEventListener("input", render);
});

render();