// Canonical row hashing: two rows hash equal exactly when Python's
// leakcheck.split.canonicalize would make them equal.
//   numbers: mantissa rounded to 30 bits (about 9 significant digits),
//            -0 -> 0, every NaN is one NaN
//   strings: trimmed, missing -> "<NA>"
// Hashes are 53-bit numbers built from two 32-bit lanes (usable as Map keys).
// Collision odds for a million rows are about 1 in 20,000.

export const MANTISSA_BITS = 30;
const SCALE = 2 ** MANTISSA_BITS;

function roundHalfEven(x) {
  const r = Math.round(x);
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
  return r;
}

/** numpy: ldexp(round(m * 2^30) / 2^30, e) where (m, e) = frexp(x). */
export function roundMantissa(x) {
  if (!Number.isFinite(x) || x === 0) return Number.isNaN(x) ? NaN : x === 0 ? 0 : x;
  let e = Math.floor(Math.log2(Math.abs(x))) + 1;
  let m = x / 2 ** e;
  if (Math.abs(m) >= 1) { m /= 2; e += 1; }
  if (Math.abs(m) < 0.5) { m *= 2; e -= 1; }
  const out = (roundHalfEven(m * SCALE) / SCALE) * 2 ** e;
  return out === 0 ? 0 : out;
}

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer);

function mix32(h) {
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return (h ^ (h >>> 16)) >>> 0;
}

export function strHash(s, seed) {
  let h = seed >>> 0;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  return mix32(h ^ s.length);
}

/** Two 32-bit words identifying a canonical cell value. */
function cellWords(col, i) {
  if (col.kind === "num") {
    let x = col.values[i];
    if (Number.isNaN(x)) return [0x7ff80000, 0x1];
    x = roundMantissa(x);
    f64[0] = x;
    return [u32[0], u32[1]];
  }
  const s = col.values[i] === null ? "<NA>" : col.values[i];
  return [strHash(s, 0x9e3779b9), strHash(s, 0x85ebca6b)];
}

/** 53-bit hash per row over the given columns (order matters). */
export function rowHashes(table, names) {
  const n = table.n;
  const cols = names.map((c) => table.cols.get(c));
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let h1 = 0x12345678;
    let h2 = 0x9abcdef0;
    for (let j = 0; j < cols.length; j++) {
      const [a, b] = cellWords(cols[j], i);
      h1 = mix32(h1 ^ a ^ Math.imul(j + 1, 0x27d4eb2d));
      h1 = mix32(h1 ^ b);
      h2 = mix32(h2 + a + Math.imul(j + 7, 0x165667b1));
      h2 = mix32(h2 ^ Math.imul(b, 0x2c1b3c6d));
    }
    out[i] = (h1 >>> 11) * 4294967296 + h2; // 21 + 32 = 53 bits
  }
  return out;
}
