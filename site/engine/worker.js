// The scan runs here, off the main thread. scan.html starts this worker from a
// blob: URL, so the worker inherits the page's Content-Security-Policy
// (connect-src 'none'): it cannot make network requests either.

import { cleanTest, makeSplit, parseCSV, scan } from "./scan.js";

const parsed = { train: null, test: null };
let split = null;

const post = (msg) => self.postMessage(msg);

async function sha256(bytes) {
  const d = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, "0")).join("");
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === "load") {
      const t0 = performance.now();
      const bytes = m.file ? new Uint8Array(await m.file.arrayBuffer()) : new TextEncoder().encode(m.text);
      const text = new TextDecoder("utf-8").decode(bytes);
      parsed[m.which] = parseCSV(text);
      split = null;
      post({
        type: "loaded", which: m.which, name: m.name, header: parsed[m.which].header,
        n: parsed[m.which].rows.length, size: bytes.length, sha256: await sha256(bytes),
        ms: Math.round(performance.now() - t0),
      });
    } else if (m.type === "scan") {
      let last = 0;
      const progress = (message, frac) => {
        const now = performance.now();
        if (now - last > 80 || frac >= 1) { last = now; post({ type: "progress", message, frac }); }
      };
      progress("Reading columns", 0.01);
      split = makeSplit(parsed.train, parsed.test, m.options);
      const report = scan(split, { sample: m.options.sample }, progress);
      post({ type: "result", report });
    } else if (m.type === "clean") {
      const r = cleanTest(parsed.train, parsed.test, split);
      post({ type: "cleaned", csv: r.csv, removed: r.removed.length, kept: r.kept, verified: r.verified,
        remaining: r.remaining_copies });
    }
  } catch (err) {
    post({ type: "error", message: err.message || String(err), code: err.code || null });
  }
};
