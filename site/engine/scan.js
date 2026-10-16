// leakcheck scanning engine for the browser: a port of leakcheck/checks/*.py.
// Pure functions, no DOM and no network: runs in a Web Worker or in Node.
// Parity with the Python package is tested in tests/test_js_parity.py.

import { labelStrings, parseCSV, typedPair } from "./table.js";
import { rowHashes, strHash } from "./canon.js";
import { balancedAccuracy, fitTree } from "./tree1d.js";

export const VERSION = "0.2.0-js";
const SEVERITIES = ["high", "medium", "low", "info", "ok"];
const RANK = Object.fromEntries(SEVERITIES.map((s, i) => [s, i]));
const N_EXAMPLES = 5;

export const DEFAULTS = {
  nearEps: 0.01,
  textTau: 0.8,
  sample: 2000, // test rows checked by L2 (the browser does exact nearest neighbours)
  shortcutMaxRows: 200000,
  seed: 0,
};

// --- small helpers -----------------------------------------------------------------

export function grade(value, high, medium, low = 0) {
  if (value >= high) return "high";
  if (value >= medium) return "medium";
  if (value > low) return "low";
  return "ok";
}

export function wilson(k, n, z = 1.96) {
  if (n === 0) return [0, 1];
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function rng(seed) {
  // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleIdx(n, k, seed) {
  if (n <= k) return Array.from({ length: n }, (_, i) => i);
  const r = rng(seed);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(r() * (n - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, k).sort((a, b) => a - b);
}

const pct = (x) => `${(100 * x).toFixed(2)}%`;
const finding = (code, severity, value, summary, details = {}, examples = []) =>
  ({ code, severity, value, summary, details, examples });

// --- the split ----------------------------------------------------------------------

/**
 * Build a split from two parsed CSVs.
 * opts: { label, time?, group?, text?, ignore?: string[] }
 */
export function makeSplit(trainParsed, testParsed, opts) {
  const { train, test, raw } = typedPair(trainParsed, testParsed);
  const { label, time = null, group = null, text = null, ignore = [] } = opts;
  if (!label) throw Object.assign(new Error("choose a label column"), { code: "label" });
  for (const c of [label, time, group, text, ...ignore]) {
    if (c && !train.cols.has(c)) throw Object.assign(new Error(`column ${c} not found`), { code: "column" });
  }
  if (text && [label, time, group, ...ignore].includes(text)) {
    throw Object.assign(new Error(`text column ${text} must be a feature`), { code: "text" });
  }
  const drop = new Set([label, time, group, ...ignore]);
  const features = train.columns.filter((c) => !drop.has(c));
  if (!features.length) {
    throw Object.assign(new Error("there are no feature columns left to compare: every column is the label, time, group or left out"),
      { code: "features" });
  }
  for (const [name, t] of [["training", train], ["test", test]]) {
    if (t.n === 0) throw Object.assign(new Error(`the ${name} file has a header but no data rows`), { code: "rows" });
  }
  const s = { train, test, raw, label, time, group, text, ignore, features };
  s.yTrain = labelStrings(train.cols.get(label));
  s.yTest = labelStrings(test.cols.get(label));
  // typedPair gave each column one kind across both files, so hashes are comparable
  s.trainHash = rowHashes(train, features);
  s.testHash = rowHashes(test, features);
  return s;
}

// --- L1 / L3: exact copies and label conflicts ---------------------------------------

function dupRate(h) {
  return h.length ? 1 - new Set(h).size / h.length : 0;
}

export function checkWithin(s) {
  return [["train", s.trainHash], ["test", s.testHash]].map(([name, h]) => {
    const r = dupRate(h);
    return finding(`L1.within_${name}`, r > 0 ? "info" : "ok", r,
      `${pct(r)} of ${name} rows repeat another ${name} row`,
      { n_rows: h.length, n_unique: new Set(h).size });
  });
}

export function checkCross(s) {
  const firstTrain = new Map();
  s.trainHash.forEach((h, i) => { if (!firstTrain.has(h)) firstTrain.set(h, i); });
  const hit = Array.from(s.testHash, (h) => firstTrain.has(h));
  const nHit = hit.filter(Boolean).length;
  const rate = hit.length ? nHit / hit.length : 0;
  const uniq = [...new Set(s.testHash)];
  const uniqRate = uniq.length ? uniq.filter((h) => firstTrain.has(h)).length / uniq.length : 0;
  const perClass = {};
  s.yTest.forEach((c, i) => {
    perClass[c] ??= { n: 0, hits: 0 };
    perClass[c].n += 1;
    if (hit[i]) perClass[c].hits += 1;
  });
  for (const c of Object.keys(perClass)) {
    perClass[c] = { n: perClass[c].n, rate: perClass[c].hits / perClass[c].n };
  }
  const examples = [];
  for (let i = 0; i < hit.length && examples.length < N_EXAMPLES; i++) {
    if (hit[i]) examples.push({ test: i, train: firstTrain.get(s.testHash[i]) });
  }
  return finding("L1.cross", grade(rate, 0.05, 0.01), rate,
    `${pct(rate)} of test rows (${nHit}) exactly duplicate a training row`,
    { n_hit: nHit, unique_test_rate: uniqRate, per_class: perClass }, examples);
}

export function checkConflicts(s) {
  const nTr = s.trainHash.length;
  const labelsOf = new Map();
  const add = (h, y) => { if (!labelsOf.has(h)) labelsOf.set(h, new Set()); labelsOf.get(h).add(y); };
  s.trainHash.forEach((h, i) => add(h, s.yTrain[i]));
  s.testHash.forEach((h, i) => add(h, s.yTest[i]));
  const bad = new Set([...labelsOf].filter(([, ys]) => ys.size > 1).map(([h]) => h));
  let inBad = 0;
  const examples = [];
  s.trainHash.forEach((h, i) => { if (bad.has(h)) { inBad += 1; if (examples.length < N_EXAMPLES) examples.push(i); } });
  s.testHash.forEach((h) => { if (bad.has(h)) inBad += 1; });
  const total = nTr + s.testHash.length;
  const rate = total ? inBad / total : 0;

  // ceiling: rows with identical features must share a prediction
  const byHash = new Map();
  s.testHash.forEach((h, i) => {
    if (!byHash.has(h)) byHash.set(h, new Map());
    const m = byHash.get(h);
    m.set(s.yTest[i], (m.get(s.yTest[i]) || 0) + 1);
  });
  let best = 0;
  for (const m of byHash.values()) best += Math.max(...m.values());
  const ceiling = s.testHash.length ? best / s.testHash.length : 1;

  const trainHashes = new Set(s.trainHash);
  const trainPairs = new Set();
  s.trainHash.forEach((h, i) => trainPairs.add(`${h}|${s.yTrain[i]}`));
  const contra = [];
  s.testHash.forEach((h, i) => { if (trainHashes.has(h) && !trainPairs.has(`${h}|${s.yTest[i]}`)) contra.push(i); });
  const contraRate = s.testHash.length ? contra.length / s.testHash.length : 0;

  return [
    finding("L3.conflict", grade(rate, 0.05, 0.01), rate,
      `${pct(rate)} of rows share identical features with a differently labelled row (${bad.size} groups)`,
      { n_groups: bad.size, n_rows: inBad }, examples),
    finding("L3.ceiling", ceiling < 0.99 ? "medium" : ceiling < 1 ? "low" : "ok", ceiling,
      `no classifier on these features can exceed ${pct(ceiling)} test accuracy`),
    finding("L3.cross_contradict", grade(contraRate, 0.01, 0.001), contraRate,
      `${pct(contraRate)} of test rows match training rows that all carry a different label`,
      { n_rows: contra.length }, contra.slice(0, N_EXAMPLES)),
  ];
}

// --- L5: shortcuts and identifiers ----------------------------------------------------

const HIGH = 0.95;
const MEDIUM = 0.9;
const ID_NAME = /(^|[\s_.-])(id|uid|uuid|ip|addr|address|port|mac|timestamp|time|date|hash|md5|sha\d*)($|[\s_.-])/i;
const ID_MIN_UNIQUE = 50;

function encodeLabels(ytr, yte) {
  const classes = [...new Set(ytr)].sort();
  const id = new Map(classes.map((c, i) => [c, i]));
  const extra = new Map();
  const enc = (y) => Int32Array.from(y, (c) => (id.has(c) ? id.get(c) : (extra.has(c) ? extra.get(c)
    : (extra.set(c, classes.length + extra.size), extra.get(c)))));
  return { classes, ytr: enc(ytr), yte: enc(yte) };
}

/** Balanced test accuracy of the best one-feature classifier (see shortcut.py). */
export function singleFeatureScore(colTr, ytr, colTe, yte) {
  if (colTr.kind === "num" && colTe.kind === "num") {
    const { classes, ytr: a, yte: b } = encodeLabels(ytr, yte);
    const clean = (v) => Float64Array.from(v, (x) => (Number.isFinite(x) ? x : NaN));
    const predict = fitTree(clean(colTr.values), a, classes.length);
    return balancedAccuracy(b, predict(clean(colTe.values)));
  }
  const xs = (col) => (col.kind === "num"
    ? Array.from(col.values, (x) => (Number.isNaN(x) ? "<NA>" : String(x)))
    : col.values.map((v) => (v === null ? "<NA>" : v)));
  const xtr = xs(colTr);
  const xte = xs(colTe);
  const counts = new Map();
  const total = new Map();
  xtr.forEach((x, i) => {
    if (!counts.has(x)) counts.set(x, new Map());
    const m = counts.get(x);
    m.set(ytr[i], (m.get(ytr[i]) || 0) + 1);
    total.set(ytr[i], (total.get(ytr[i]) || 0) + 1);
  });
  const top = (m) => [...m].sort((p, q) => q[1] - p[1])[0][0];
  const fallback = top(total);
  const majority = new Map([...counts].map(([x, m]) => [x, top(m)]));
  const pred = xte.map((x) => (majority.has(x) ? majority.get(x) : fallback));
  return balancedAccuracy(yte, pred);
}

function subsetColumn(col, idx) {
  return col.kind === "num"
    ? { kind: "num", values: Float64Array.from(idx, (i) => col.values[i]) }
    : { kind: "str", values: idx.map((i) => col.values[i]) };
}

export function checkShortcuts(s, opts = {}, progress = () => {}) {
  const max = opts.shortcutMaxRows ?? DEFAULTS.shortcutMaxRows;
  const seed = opts.seed ?? DEFAULTS.seed;
  const itr = sampleIdx(s.train.n, max, seed);
  const ite = sampleIdx(s.test.n, max, seed + 1);
  const ytr = itr.map((i) => s.yTrain[i]);
  const yte = ite.map((i) => s.yTest[i]);
  const scores = [];
  s.features.forEach((c, k) => {
    progress(`Testing single features (${k + 1}/${s.features.length})`, k / s.features.length);
    scores.push([c, singleFeatureScore(subsetColumn(s.train.cols.get(c), itr), ytr,
      subsetColumn(s.test.cols.get(c), ite), yte)]);
  });
  scores.sort((p, q) => q[1] - p[1]);
  const [bestCol, best] = scores[0] || [null, 0];
  const nClasses = new Set(ytr).size;
  const flagged = scores.filter(([, v]) => v >= HIGH).map(([c]) => c);
  return finding("L5.shortcut", best >= HIGH ? "high" : best >= MEDIUM ? "medium" : "ok", best,
    `best single feature '${bestCol}' reaches ${pct(best)} balanced test accuracy `
    + `(chance ${pct(1 / Math.max(nClasses, 1))}); ${flagged.length} feature(s) >= 95%`,
    { top: scores.slice(0, 10).map(([feature, balanced_acc]) => ({ feature, balanced_acc })), flagged,
      n_classes: nClasses, rows_used: { train: itr.length, test: ite.length } });
}

export function checkIdentifiers(s) {
  const hits = [];
  const n = s.train.n;
  for (const c of s.features) {
    if (c === s.text) continue;
    const col = s.train.cols.get(c);
    const uniq = new Set();
    for (const v of col.values) if (!(v === null || (typeof v === "number" && Number.isNaN(v)))) uniq.add(v);
    const reasons = [];
    if (ID_NAME.test(c) && uniq.size >= ID_MIN_UNIQUE) reasons.push("name");
    if (col.kind !== "num" && n && uniq.size / n > 0.5) reasons.push("high-cardinality text");
    if (reasons.length) hits.push({ feature: c, reasons });
  }
  return finding("L5.identifier", hits.length ? "medium" : "ok", hits.length,
    `${hits.length} feature(s) look like identifiers or context (IPs, ports, IDs, timestamps)`, { columns: hits });
}

// --- L2: near copies (tabular) ------------------------------------------------------

const EPS_GRID = [0.001, 0.003, 0.01, 0.03, 0.1];
const MISSING = 1e3;

function numericCols(s) {
  return s.features.filter((c) => c !== s.text && s.train.cols.get(c).kind === "num");
}
function categoricalCols(s) {
  const num = new Set(numericCols(s));
  return s.features.filter((c) => !num.has(c) && c !== s.text);
}
const slog = (x) => Math.sign(x) * Math.log1p(Math.abs(x));

function percentile(sorted, q) { // numpy linear interpolation
  if (!sorted.length) return NaN;
  const r = (q / 100) * (sorted.length - 1);
  const lo = Math.floor(r);
  const hi = Math.ceil(r);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (r - lo);
}

/** Scaled numeric rows: Float64Array per split, row-major, RMS-normalised. */
function embed(trainTable, testTable, cols) {
  const d = cols.length;
  const stats = cols.map((c) => {
    const fin = [];
    for (const x of trainTable.cols.get(c).values) { const v = slog(x); if (Number.isFinite(v)) fin.push(v); }
    fin.sort((a, b) => a - b);
    const med = percentile(fin, 50);
    const iqr = percentile(fin, 75) - percentile(fin, 25);
    let std = NaN;
    if (fin.length) {
      const mean = fin.reduce((a, b) => a + b, 0) / fin.length;
      std = Math.sqrt(fin.reduce((a, b) => a + (b - mean) ** 2, 0) / fin.length);
    }
    const scale = iqr > 0 ? iqr : std > 0 ? std : 1;
    return { med: Number.isFinite(med) ? med : 0, scale: Number.isFinite(scale) ? scale : 1 };
  });
  const norm = Math.sqrt(Math.max(d, 1));
  const z = (table) => {
    const out = new Float64Array(table.n * d);
    cols.forEach((c, j) => {
      const v = table.cols.get(c).values;
      const { med, scale } = stats[j];
      for (let i = 0; i < table.n; i++) {
        const x = v[i];
        let y;
        if (Number.isNaN(x)) y = -MISSING;
        else if (x === Infinity) y = 2 * MISSING;
        else if (x === -Infinity) y = -2 * MISSING;
        else y = (slog(x) - med) / scale;
        out[i * d + j] = y / norm;
      }
    });
    return out;
  };
  return { tr: z(trainTable), te: z(testTable), d };
}

/** Exact nearest training row for chosen test rows, within equal categorical buckets. */
function nearest(trainTable, testTable, s, testIdx, progress = () => {}) {
  const cols = numericCols(s);
  const cats = categoricalCols(s);
  const dist = new Float64Array(testIdx.length).fill(Infinity);
  const nn = new Int32Array(testIdx.length).fill(-1);
  if (!cols.length || !trainTable.n) return { dist, nn };
  const { tr, te, d } = embed(trainTable, testTable, cols);
  const ktr = rowHashes(trainTable, cats);
  const kte = rowHashes(testTable, cats);
  const buckets = new Map();
  ktr.forEach((k, i) => { if (!buckets.has(k)) buckets.set(k, []); buckets.get(k).push(i); });
  testIdx.forEach((t, q) => {
    if (q % 50 === 0) progress(`Finding near copies (${q}/${testIdx.length})`, q / testIdx.length);
    const members = buckets.get(kte[t]);
    if (!members) return;
    let best = Infinity;
    let arg = -1;
    const off = t * d;
    for (const i of members) {
      const o = i * d;
      let ss = 0;
      for (let j = 0; j < d; j++) { const diff = te[off + j] - tr[o + j]; ss += diff * diff; if (ss >= best) break; }
      if (ss < best) { best = ss; arg = i; }
    }
    dist[q] = Math.sqrt(best);
    nn[q] = arg;
  });
  return { dist, nn };
}

function subTable(table, idx) {
  const cols = new Map();
  for (const [c, col] of table.cols) cols.set(c, subsetColumn(col, idx));
  return { columns: table.columns, cols, n: idx.length };
}

export function checkNear(s, opts = {}, progress = () => {}) {
  if (!numericCols(s).length) return null;
  const eps = opts.nearEps ?? DEFAULTS.nearEps;
  const sample = opts.sample ?? DEFAULTS.sample;
  const seed = opts.seed ?? DEFAULTS.seed;
  const idx = sampleIdx(s.test.n, sample, seed);
  const { dist, nn } = nearest(s.train, s.test, s, idx, (m, f) => progress(m, f * 0.7));
  const trainSet = new Set(s.trainHash);
  const exact = idx.map((i) => trainSet.has(s.testHash[i]));
  const grid = [...new Set([...EPS_GRID, eps])].sort((a, b) => a - b);
  const n = idx.length;
  let k = 0;
  const nearMask = idx.map((_, q) => { const v = dist[q] <= eps && !exact[q]; if (v) k += 1; return v; });
  const curve = Object.fromEntries(grid.map((e) => [String(e),
    n ? idx.filter((_, q) => dist[q] <= e && !exact[q]).length / n : 0]));

  // baseline: held-out training rows against the rest of training
  const kb = Math.min(sample, Math.floor(s.train.n / 2));
  let baseline = {};
  if (kb >= 1) {
    const r = rng(seed + 7);
    const perm = Array.from({ length: s.train.n }, (_, i) => i);
    for (let i = perm.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    const held = perm.slice(0, kb);
    const rest = perm.slice(kb);
    const restT = subTable(s.train, rest);
    const heldT = subTable(s.train, held);
    const b = nearest(restT, heldT, s, held.map((_, i) => i), (m, f) => progress(`Baseline: ${m}`, 0.7 + f * 0.3));
    const restHash = new Set(rest.map((i) => s.trainHash[i]));
    const ex = held.map((i) => restHash.has(s.trainHash[i]));
    baseline = Object.fromEntries(grid.map((e) => [String(e), held.filter((_, q) => b.dist[q] <= e && !ex[q]).length / kb]));
  }
  const rate = n ? k / n : 0;
  const examples = [];
  nearMask.forEach((v, q) => { if (v && examples.length < N_EXAMPLES) examples.push({ test: idx[q], train: nn[q], distance: dist[q] }); });
  return finding("L2.near", grade(rate, 0.05, 0.01), rate,
    `${pct(rate)} of test rows are near-duplicates of a training row (RMS scaled distance <= ${eps}, not exact)`,
    { eps, n_checked: n, n_near: k, sampled: n < s.test.n, ci95: wilson(k, n),
      exact_or_near_rate: n ? idx.filter((_, q) => dist[q] <= eps || exact[q]).length / n : 0,
      curve, train_baseline_curve: baseline,
      vs_train_baseline: baseline[String(eps)] ? rate / baseline[String(eps)] : null,
      numeric_features: numericCols(s).length, categorical_features: categoricalCols(s).length },
    examples);
}

// --- L2: near copies (text) ---------------------------------------------------------

const SHINGLE = 5;
const NUM_PERM = 128;
const BANDS = 32;
const ROWS = 4;
const BUCKET_CAP = 100;

export const normalizeText = (s) => String(s ?? "").toLowerCase().split(/\s+/).filter(Boolean).join(" ");

export function shingles(s) {
  s = normalizeText(s);
  if (s.length <= SHINGLE) return new Set([s]);
  const out = new Set();
  for (let i = 0; i <= s.length - SHINGLE; i++) out.add(s.slice(i, i + SHINGLE));
  return out;
}

export function jaccard(a, b) {
  if (!a.size && !b.size) return 1;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter += 1;
  return inter / (a.size + b.size - inter);
}

const SEEDS = Array.from({ length: NUM_PERM }, (_, i) => strHash(`perm${i}`, 0x5bd1e995));

function signature(sh) {
  const hs = [...sh].map((x) => strHash(x, 0x811c9dc5));
  const sig = new Uint32Array(NUM_PERM).fill(0xffffffff);
  for (const h of hs) {
    for (let p = 0; p < NUM_PERM; p++) {
      let v = Math.imul((h ^ SEEDS[p]) >>> 0, 0x9e3779b1) >>> 0;
      v = (v ^ (v >>> 15)) >>> 0;
      v = Math.imul(v, 0x85ebca77) >>> 0;
      v = (v ^ (v >>> 13)) >>> 0;
      if (v < sig[p]) sig[p] = v;
    }
  }
  return sig;
}

/** Best exact Jaccard to a training text among LSH candidates, per test index. */
export function nearestText(trainText, testText, testIdx, tau, progress = () => {}) {
  const trSh = trainText.map(shingles);
  const teSh = testIdx.map((i) => shingles(testText[i]));
  const sim = new Float64Array(testIdx.length);
  const best = new Int32Array(testIdx.length).fill(-1);
  progress("Signing training text", 0.1);
  const sigTr = trSh.map(signature);
  const sigTe = teSh.map(signature);
  const bandKey = (sig, b) => `${b}:${sig[b * ROWS]}:${sig[b * ROWS + 1]}:${sig[b * ROWS + 2]}:${sig[b * ROWS + 3]}`;
  const buckets = new Map();
  sigTr.forEach((sig, j) => {
    for (let b = 0; b < BANDS; b++) {
      const k = bandKey(sig, b);
      const list = buckets.get(k);
      if (!list) buckets.set(k, [j]);
      else if (list.length < BUCKET_CAP) list.push(j);
    }
  });
  sigTe.forEach((sig, q) => {
    if (q % 200 === 0) progress(`Comparing text (${q}/${sigTe.length})`, 0.3 + (0.7 * q) / sigTe.length);
    const cand = new Map();
    for (let b = 0; b < BANDS; b++) {
      for (const j of buckets.get(bandKey(sig, b)) || []) {
        if (cand.has(j)) continue;
        let eq = 0;
        for (let p = 0; p < NUM_PERM; p++) if (sig[p] === sigTr[j][p]) eq += 1;
        const est = eq / NUM_PERM;
        if (est >= tau - 0.15) cand.set(j, est);
      }
    }
    const top = [...cand].sort((a, b) => b[1] - a[1]).slice(0, 5);
    for (const [j] of top) {
      const v = jaccard(teSh[q], trSh[j]);
      if (v > sim[q]) { sim[q] = v; best[q] = j; }
    }
  });
  return { sim, best };
}

export function checkNearText(s, opts = {}, progress = () => {}) {
  if (!s.text) return null;
  const tau = opts.textTau ?? DEFAULTS.textTau;
  const idx = sampleIdx(s.test.n, opts.textSample ?? 20000, opts.seed ?? DEFAULTS.seed);
  const trainText = s.train.cols.get(s.text).values.map((v) => v ?? "");
  const testText = s.test.cols.get(s.text).values.map((v) => v ?? "");
  const { sim, best } = nearestText(trainText, testText, idx, tau, progress);
  const trNorm = new Set(trainText.map(normalizeText));
  const same = idx.map((i) => trNorm.has(normalizeText(testText[i])));
  let k = 0;
  const near = idx.map((_, q) => { const v = sim[q] >= tau && !same[q]; if (v) k += 1; return v; });
  const n = idx.length;
  const rate = n ? k / n : 0;
  const identical = n ? same.filter(Boolean).length / n : 0;
  const examples = [];
  near.forEach((v, q) => { if (v && examples.length < N_EXAMPLES) examples.push({ test: idx[q], train: best[q], jaccard: sim[q] }); });
  return finding("L2.near_text", grade(rate, 0.05, 0.01), rate,
    `${pct(rate)} of test '${s.text}' values are near-copies (Jaccard >= ${tau}) of a training value; `
    + `${pct(identical)} are identical after normalisation`,
    { tau, n_checked: n, n_near: k, identical_rate: identical, sampled: n < s.test.n, ci95: wilson(k, n),
      shingle: SHINGLE, num_perm: NUM_PERM, bands: BANDS }, examples);
}

// --- L4: time --------------------------------------------------------------------------

const ISO = /^(\d{4}-\d{2}(-\d{2})?)([ T](\d{2}:\d{2}(:\d{2}(\.\d+)?)?))?$/;

export function toSeconds(col) {
  if (col.kind === "num") return { secs: Float64Array.from(col.values), isDate: false };
  const secs = Float64Array.from(col.values, (v) => {
    if (v === null) return NaN;
    const m = ISO.exec(v);
    let t;
    if (m) t = Date.parse(`${m[1]}${m[3] ? `T${m[4]}` : ""}Z`);
    else { t = Date.parse(`${v} UTC`); if (Number.isNaN(t)) t = Date.parse(v); }
    return Number.isNaN(t) ? NaN : t / 1000;
  });
  return { secs, isDate: true };
}

function fmtTime(x, isDate) {
  if (!Number.isFinite(x)) return "n/a";
  if (!isDate) return x;
  return `${new Date(x * 1000).toISOString().replace("T", " ").replace(".000Z", "").replace("Z", "")}+00:00`;
}

export function checkTemporal(s) {
  if (!s.time) return [];
  const a1 = toSeconds(s.train.cols.get(s.time));
  const b1 = toSeconds(s.test.cols.get(s.time));
  const isDate = a1.isDate || b1.isDate;
  const okTr = Array.from(a1.secs, Number.isFinite);
  const okTe = Array.from(b1.secs, Number.isFinite);
  if (!okTr.some(Boolean) || !okTe.some(Boolean)) {
    return [finding("L4.after_test_start", "info", 0, `time column '${s.time}' could not be parsed`,
      { unparsed_train: okTr.filter((v) => !v).length, unparsed_test: okTe.filter((v) => !v).length })];
  }
  const a = Array.from(a1.secs).filter(Number.isFinite);
  const b = Array.from(b1.secs).filter(Number.isFinite).sort((x, y) => x - y);
  const t0 = b[0];
  const t1 = b[b.length - 1];
  const tmed = b.length % 2 ? b[(b.length - 1) / 2] : (b[b.length / 2 - 1] + b[b.length / 2]) / 2;
  const frac = (f) => a.filter(f).length / a.length;
  const after = frac((x) => x > t0);
  const perClass = {};
  const add = (secs, ys) => secs.forEach((x, i) => {
    if (!Number.isFinite(x)) return;
    const c = ys[i];
    perClass[c] ??= { min: Infinity, max: -Infinity, n: 0 };
    perClass[c].min = Math.min(perClass[c].min, x);
    perClass[c].max = Math.max(perClass[c].max, x);
    perClass[c].n += 1;
  });
  add(a1.secs, s.yTrain);
  add(b1.secs, s.yTest);
  for (const c of Object.keys(perClass)) {
    perClass[c] = { min: fmtTime(perClass[c].min, isDate), max: fmtTime(perClass[c].max, isDate), n: perClass[c].n };
  }
  const out = [finding("L4.after_test_start", grade(after, 0.1, 0.01), after,
    `${pct(after)} of training rows are newer than the earliest test row `
    + `(${pct(frac((x) => x > tmed))} newer than the median, ${pct(frac((x) => x > t1))} newer than all test rows)`,
    { train_range: [fmtTime(Math.min(...a), isDate), fmtTime(Math.max(...a), isDate)],
      test_range: [fmtTime(t0, isDate), fmtTime(t1, isDate)],
      after_test_median: frac((x) => x > tmed), after_test_end: frac((x) => x > t1),
      unparsed_train: okTr.filter((v) => !v).length, unparsed_test: okTe.filter((v) => !v).length,
      per_class: perClass })];
  const score = singleFeatureScore({ kind: "num", values: a1.secs }, s.yTrain, { kind: "num", values: b1.secs }, s.yTest);
  out.push(finding("L4.time_shortcut", score >= HIGH ? "high" : score >= MEDIUM ? "medium" : "ok", score,
    `the timestamp alone reaches ${pct(score)} balanced test accuracy`));
  return out;
}

// --- L6: groups ------------------------------------------------------------------------

function groupValues(s, table) {
  const col = table.cols.get(s.group);
  return col.kind === "num" ? labelStrings(col) : col.values.map((v) => (v === null ? "<NA>" : v));
}

export function checkGroup(s) {
  if (!s.group) return null;
  const gtr = groupValues(s, s.train);
  const gte = groupValues(s, s.test);
  const trSet = new Set(gtr);
  const nHit = gte.filter((g) => trSet.has(g)).length;
  const rate = gte.length ? nHit / gte.length : 0;
  const teSet = new Set(gte);
  const shared = [...teSet].filter((g) => trSet.has(g)).sort();
  return finding("L6.group", grade(rate, 0.05, 0.01), rate,
    `${pct(rate)} of test rows belong to a '${s.group}' value also present in training `
    + `(${shared.length} of ${teSet.size} test groups)`,
    { n_hit: nHit, shared_groups: shared.length, test_groups: teSet.size }, shared.slice(0, N_EXAMPLES));
}

// --- the scan ----------------------------------------------------------------------------

export const ALL_CHECKS = ["L1", "L2", "L3", "L4", "L5", "L6"];

export function worst(findings) {
  if (!findings.length) return "ok";
  return findings.map((f) => f.severity).sort((a, b) => RANK[a] - RANK[b])[0];
}

/** Run every check; returns the same shape as Python's Report.to_dict(). */
export function scan(s, opts = {}, progress = () => {}) {
  const checks = opts.checks ?? ALL_CHECKS;
  const findings = [];
  const skipped = [];
  const add = (name, r) => {
    if (r === null || (Array.isArray(r) && !r.length)) skipped.push(name);
    else if (Array.isArray(r)) findings.push(...r);
    else findings.push(r);
  };
  const step = (label, frac) => (m, f = 0) => progress(m || label, frac[0] + (frac[1] - frac[0]) * f);
  if (checks.includes("L1")) {
    progress(`Hashing ${s.train.n.toLocaleString("en")} + ${s.test.n.toLocaleString("en")} rows`, 0.05);
    add("L1.within", checkWithin(s));
    add("L1.cross", checkCross(s));
  }
  if (checks.includes("L2")) {
    add("L2.near", checkNear(s, opts, step("Finding near copies", [0.1, 0.55])));
    add("L2.near_text", checkNearText(s, opts, step("Comparing text", [0.55, 0.65])));
  }
  if (checks.includes("L3")) { progress("Checking label conflicts", 0.66); add("L3", checkConflicts(s)); }
  if (checks.includes("L4")) { progress("Checking time order", 0.68); add("L4", checkTemporal(s)); }
  if (checks.includes("L5")) {
    add("L5.shortcut", checkShortcuts(s, opts, step("Testing single features", [0.7, 0.98])));
    add("L5.identifier", checkIdentifiers(s));
  }
  if (checks.includes("L6")) add("L6.group", checkGroup(s));
  progress("Done", 1);
  const meta = {
    version: VERSION, engine: "javascript", n_train: s.train.n, n_test: s.test.n, n_features: s.features.length,
    label: s.label, time: s.time, group: s.group, text: s.text, ignore: s.ignore, checks, skipped,
    seed: opts.seed ?? DEFAULTS.seed, near_eps: opts.nearEps ?? DEFAULTS.nearEps,
    text_tau: opts.textTau ?? DEFAULTS.textTau, sample: opts.sample ?? DEFAULTS.sample,
  };
  return { meta, worst: worst(findings), findings };
}

// --- clean: drop test rows that exactly copy a training row, then prove it -----------

function csvField(v) {
  return /[",\r\n]/.test(v) || /^\s/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
}

/**
 * Returns { csv, removed, verified }: the test file without rows whose features
 * exactly match a training row. The output is re-parsed and re-hashed against
 * the training file; verified is true only if no copy remains.
 */
export function cleanTest(trainParsed, testParsed, s) {
  const trainSet = new Set(s.trainHash);
  const keep = [];
  const removed = [];
  s.testHash.forEach((h, i) => (trainSet.has(h) ? removed : keep).push(i));
  const lines = [testParsed.header.map(csvField).join(",")];
  for (const i of keep) lines.push(testParsed.rows[i].map(csvField).join(","));
  const csv = `${lines.join("\n")}\n`;
  // the claim is checked on the output as written, not on the in-memory rows
  const again = makeSplit(trainParsed, parseCSV(csv), {
    label: s.label, time: s.time, group: s.group, text: s.text, ignore: s.ignore });
  const againTrain = new Set(again.trainHash);
  const left = Array.from(again.testHash).filter((h) => againTrain.has(h)).length;
  return { csv, removed, kept: keep.length, verified: left === 0, remaining_copies: left };
}

export { parseCSV };
