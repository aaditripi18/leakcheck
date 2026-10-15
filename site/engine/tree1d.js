// One-feature decision tree, mirroring
// sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=32, random_state=0)
// as used by leakcheck's L5 check: gini, best-first growth, float32 inputs,
// missing values routed to whichever side gives the better split.

const FEATURE_THRESHOLD = 1e-7;
const EPSILON = 2.220446049250313e-16;

function gini(counts, n) {
  if (n <= 0) return 0;
  let s = 0;
  for (const c of counts) s += (c / n) * (c / n);
  return 1 - s;
}

/**
 * Fit on x (numbers, NaN = missing) and integer labels y in [0, K).
 * Returns predict(xs) -> Int32Array of class ids.
 */
export function fitTree(x, y, K, maxLeaves = 32) {
  const n = x.length;
  const xf = Float32Array.from(x); // sklearn trains on float32
  const order = [];
  const missing = [];
  for (let i = 0; i < n; i++) (Number.isNaN(xf[i]) ? missing : order).push(i);
  order.sort((a, b) => xf[a] - xf[b] || a - b);
  const m = order.length;
  const xs = new Float64Array(m);
  const cum = new Int32Array((m + 1) * K); // prefix class counts over sorted order
  for (let p = 0; p < m; p++) {
    xs[p] = xf[order[p]];
    for (let k = 0; k < K; k++) cum[(p + 1) * K + k] = cum[p * K + k];
    cum[(p + 1) * K + y[order[p]]] += 1;
  }
  const missCounts = new Array(K).fill(0);
  for (const i of missing) missCounts[y[i]] += 1;
  const nMissAll = missing.length;

  const countsRange = (lo, hi) => {
    const c = new Array(K);
    for (let k = 0; k < K; k++) c[k] = cum[hi * K + k] - cum[lo * K + k];
    return c;
  };

  // node: sorted range [lo, hi) plus whether the missing rows reach it
  function makeNode(lo, hi, hasMissing) {
    const counts = countsRange(lo, hi);
    if (hasMissing) for (let k = 0; k < K; k++) counts[k] += missCounts[k];
    const nn = hi - lo + (hasMissing ? nMissAll : 0);
    return { lo, hi, hasMissing, counts, n: nn, impurity: gini(counts, nn) };
  }

  function bestSplit(node) {
    const { lo, hi, hasMissing } = node;
    const nMiss = hasMissing ? nMissAll : 0;
    if (node.n < 2 || node.impurity <= EPSILON) return null;
    // sklearn skips a feature whose non-missing values are constant, missing or not
    if (hi - lo < 1 || xs[hi - 1] <= xs[lo] + FEATURE_THRESHOLD) return null;
    let best = null;
    const passes = nMiss > 0 ? [false, true] : [false]; // missing right, then missing left
    for (const missLeft of passes) {
      for (let p = lo + 1; p < hi; p++) {
        if (xs[p] <= xs[p - 1] + FEATURE_THRESHOLD) continue;
        const left = countsRange(lo, p);
        const right = countsRange(p, hi);
        let nl = p - lo;
        let nr = hi - p;
        if (missLeft) { for (let k = 0; k < K; k++) left[k] += missCounts[k]; nl += nMiss; }
        else if (nMiss) { for (let k = 0; k < K; k++) right[k] += missCounts[k]; nr += nMiss; }
        const proxy = -nl * gini(left, nl) - nr * gini(right, nr);
        if (!best || proxy > best.proxy) {
          let thr = xs[p - 1] / 2 + xs[p] / 2;
          if (thr === xs[p] || !Number.isFinite(thr)) thr = xs[p - 1];
          best = { proxy, pos: p, thr, missLeft, nl, nr, left, right };
        }
      }
    }
    if (nMiss > 0 && hi > lo) {
      // all non-missing left, missing right
      const left = countsRange(lo, hi);
      const right = missCounts.slice();
      const nl = hi - lo;
      const nr = nMiss;
      const proxy = -nl * gini(left, nl) - nr * gini(right, nr);
      if (!best || proxy > best.proxy) best = { proxy, pos: hi, thr: Infinity, missLeft: false, nl, nr, left, right };
    }
    if (!best) return null;
    const imp = (node.n / n) * (node.impurity - (best.nl / node.n) * gini(best.left, best.nl)
      - (best.nr / node.n) * gini(best.right, best.nr));
    best.improvement = imp;
    return best;
  }

  const root = makeNode(0, m, nMissAll > 0);
  root.split = bestSplit(root);
  const heap = [root]; // tiny: at most 32 leaves, a sorted array is fine
  let leaves = 1;
  while (heap.length && leaves < maxLeaves) {
    heap.sort((a, b) => (b.split ? b.split.improvement : -1) - (a.split ? a.split.improvement : -1));
    const node = heap[0];
    if (!node.split || node.split.improvement + EPSILON < 0) break;
    heap.shift();
    const s = node.split;
    const l = makeNode(node.lo, s.pos, node.hasMissing && s.missLeft);
    const r = makeNode(s.pos, node.hi, node.hasMissing && !s.missLeft);
    node.thr = s.thr;
    // nodes that saw no missing values in training send them to the larger child
    node.missLeft = node.hasMissing ? s.missLeft : l.n >= r.n;
    node.left = l;
    node.right = r;
    l.split = bestSplit(l);
    r.split = bestSplit(r);
    heap.push(l, r);
    leaves += 1;
  }

  const argmax = (c) => c.reduce((bi, v, i) => (v > c[bi] ? i : bi), 0);
  return function predict(xt) {
    const out = new Int32Array(xt.length);
    for (let i = 0; i < xt.length; i++) {
      const v = Math.fround(xt[i]);
      let node = root;
      while (node.left) {
        if (Number.isNaN(v)) node = node.missLeft ? node.left : node.right;
        else node = v <= node.thr ? node.left : node.right;
      }
      out[i] = argmax(node.counts);
    }
    return out;
  };
}

/** sklearn balanced_accuracy_score: mean recall over classes present in y_true. */
export function balancedAccuracy(yTrue, yPred) {
  const tot = new Map();
  const hit = new Map();
  for (let i = 0; i < yTrue.length; i++) {
    tot.set(yTrue[i], (tot.get(yTrue[i]) || 0) + 1);
    if (yTrue[i] === yPred[i]) hit.set(yTrue[i], (hit.get(yTrue[i]) || 0) + 1);
  }
  let s = 0;
  for (const [c, t] of tot) s += (hit.get(c) || 0) / t;
  return tot.size ? s / tot.size : 0;
}
