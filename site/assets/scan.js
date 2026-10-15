// Scanner page: file loading, column settings, running the scan in a worker,
// and rendering the report. Everything a user's file contains is inserted with
// textContent, never as HTML.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v);
  }
  for (const c of children) if (c != null) el.append(c);
  return el;
}

const pct = (x, digits = 1) => {
  if (x == null || Number.isNaN(x)) return "—";
  const v = x * 100;
  if (v === 0) return "0.0%";
  if (Math.abs(v) < 0.01) return `${v.toFixed(3)}%`;
  if (Math.abs(v) < 1) return `${v.toFixed(2)}%`;
  return `${v.toFixed(digits)}%`;
};
const fmtInt = (n) => n.toLocaleString("en");
const fmtBytes = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} KB`);

// --- the worker (blob URL so it inherits this page's CSP) ---------------------------

const workerUrl = new URL("../engine/worker.js", import.meta.url).href;
const worker = new Worker(URL.createObjectURL(new Blob([`import ${JSON.stringify(workerUrl)};`],
  { type: "text/javascript" })), { type: "module" });

const state = { files: { train: null, test: null }, header: null, report: null, preset: null, running: false };

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "loaded") onLoaded(m);
  else if (m.type === "progress") onProgress(m);
  else if (m.type === "result") onResult(m.report);
  else if (m.type === "cleaned") onCleaned(m);
  else if (m.type === "error") showError(m.message, m.code);
};
worker.onerror = (e) => showError(`The scanner could not start: ${e.message || "unknown error"}. It needs a recent browser and must be served over http(s), not opened as a file.`);

// --- loading files -----------------------------------------------------------------

function zone(which) { return $(`[data-drop="${which}"]`); }

function setZoneLoading(which, name) {
  const z = zone(which);
  $("[data-empty]", z).classList.add("hidden");
  $("[data-filled]", z).classList.replace("hidden", "flex");
  $('[data-field="name"]', z).textContent = name;
  $('[data-field="shape"]', z).textContent = "Reading…";
  $('[data-field="sha"]', z).textContent = "";
}

function loadFile(which, file) {
  if (!file) return;
  clearError();
  hideResults();
  state.files[which] = null;
  setZoneLoading(which, file.name);
  worker.postMessage({ type: "load", which, file, name: file.name });
  updateSettings();
}

function onLoaded(m) {
  state.files[m.which] = m;
  const z = zone(m.which);
  z.classList.remove("border-dashed");
  $("[data-loaded-badge]", z).classList.replace("hidden", "inline-flex");
  $('[data-field="shape"]', z).textContent = `${fmtInt(m.n)} rows · ${m.header.length} columns · ${fmtBytes(m.size)} · read in ${m.ms} ms`;
  $('[data-field="sha"]', z).textContent = `SHA-256 ${m.sha256.slice(0, 12)}…${m.sha256.slice(-4)}`;
  $('[data-field="sha"]', z).title = m.sha256;
  updateSettings();
}

for (const which of ["train", "test"]) {
  const z = zone(which);
  const input = $(`[data-input="${which}"]`);
  input.addEventListener("change", () => loadFile(which, input.files[0]));
  z.addEventListener("dragover", (e) => { e.preventDefault(); z.classList.add("is-over"); });
  z.addEventListener("dragleave", () => z.classList.remove("is-over"));
  z.addEventListener("drop", (e) => {
    e.preventDefault();
    z.classList.remove("is-over");
    loadFile(which, e.dataTransfer.files[0]);
  });
  $(`[data-replace="${which}"]`).addEventListener("click", () => { input.value = ""; input.click(); });
}

// examples ship as scripts, so loading them needs no network request beyond this site
$$("[data-example]").forEach((btn) => btn.addEventListener("click", () => {
  const key = btn.dataset.example;
  const go = () => {
    const ex = window.LEAKCHECK_EXAMPLES[key];
    state.preset = { label: ex.label, ignore: ex.ignore };
    clearError();
    hideResults();
    for (const which of ["train", "test"]) {
      state.files[which] = null;
      setZoneLoading(which, ex.files[which]);
      worker.postMessage({ type: "load", which, text: ex[which], name: ex.files[which] });
    }
    const note = $("[data-example-note]");
    note.textContent = `${ex.name}. Source: ${ex.source}. Licence: ${ex.licence}.`;
    note.hidden = false;
    updateSettings();
  };
  if (window.LEAKCHECK_EXAMPLES && window.LEAKCHECK_EXAMPLES[key]) return go();
  // the old files must not stay runnable while the example script loads
  clearError();
  hideResults();
  for (const which of ["train", "test"]) { state.files[which] = null; setZoneLoading(which, `${key} example`); }
  updateSettings();
  const s = document.createElement("script");
  s.src = `examples/${key}.js`;
  s.onload = go;
  s.onerror = () => showError("Could not load the example.");
  document.head.append(s);
}));

// --- settings ----------------------------------------------------------------------

const LABEL_GUESS = ["label", "Label", "class", "Class", "target", "Result", "is_attack", "y"];
const sel = (f) => $(`[data-opt="${f}"]`);
const ignored = new Set();

function fillSelect(el, header, none) {
  const keep = el.value;
  el.replaceChildren(h("option", { value: "", text: none ? "None" : "Choose…" }),
    ...header.map((c) => h("option", { value: c, text: c })));
  if (header.includes(keep)) el.value = keep;
  el.disabled = false;
}

function updateSettings() {
  const { train, test } = state.files;
  const run = $("[data-run]");
  const schema = $("[data-schema]");
  if (!train || !test) {
    run.disabled = true;
    schema.textContent = "Load both files to continue";
    return;
  }
  const a = new Set(train.header);
  const missing = train.header.filter((c) => !test.header.includes(c)).concat(test.header.filter((c) => !a.has(c)));
  if (missing.length) {
    run.disabled = true;
    schema.textContent = "Columns differ";
    setErrorText(`The two files have different columns. Only in one of them: ${missing.join(", ")}.`);
    return;
  }
  schema.textContent = `${train.header.length} columns in both files`;
  const header = train.header;
  const fresh = JSON.stringify(header) !== JSON.stringify(state.header);
  state.header = header;
  if (fresh) {
    fillSelect(sel("label"), header, false);
    for (const f of ["time", "group", "text"]) fillSelect(sel(f), header, true);
    ignored.clear();
    const label = state.preset?.label && header.includes(state.preset.label) ? state.preset.label
      : LABEL_GUESS.find((c) => header.includes(c)) || "";
    sel("label").value = label;
    for (const c of state.preset?.ignore || []) if (header.includes(c)) ignored.add(c);
  }
  state.preset = null;
  renderIgnore();
  run.disabled = !sel("label").value || state.running;
}

function chosen() {
  return { label: sel("label").value, time: sel("time").value || null, group: sel("group").value || null,
    text: sel("text").value || null };
}

function renderIgnore() {
  const list = $("[data-ignore-list]");
  const c = chosen();
  const roles = new Set([c.label, c.time, c.group].filter(Boolean));
  list.replaceChildren(...state.header.filter((col) => !roles.has(col)).map((col) => {
    const on = ignored.has(col);
    const b = h("button", {
      type: "button", "aria-pressed": String(on),
      class: `px-2 py-0.5 font-mono-data-sm text-mono-data-sm border transition-colors ${on
        ? "bg-on-surface text-surface border-on-surface line-through" : "bg-surface-container border-outline-variant text-on-surface hover:border-outline"}`,
      text: col,
    });
    b.addEventListener("click", () => { on ? ignored.delete(col) : ignored.add(col); renderIgnore(); });
    return b;
  }));
}

for (const f of ["label", "time", "group", "text"]) {
  sel(f).addEventListener("change", () => { renderIgnore(); $("[data-run]").disabled = !sel("label").value || state.running; });
}

// --- running -----------------------------------------------------------------------

function run() {
  if ($("[data-run]").disabled) return;
  clearError();
  hideResults();
  const c = chosen();
  const roles = new Set([c.label, c.time, c.group]);
  state.running = true;
  $("[data-run]").disabled = true;
  const p = $("[data-progress]");
  p.classList.replace("hidden", "flex");
  onProgress({ message: "Starting", frac: 0 });
  worker.postMessage({ type: "scan", options: { ...c, ignore: [...ignored].filter((x) => !roles.has(x)),
    sample: Number(sel("sample").value) } });
}

$("[data-run]").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); run(); } });

function onProgress(m) {
  const f = Math.max(0, Math.min(1, m.frac || 0));
  $("[data-progress-bar]").style.width = `${(f * 100).toFixed(0)}%`;
  $("[data-progress-pct]").textContent = `${(f * 100).toFixed(0)}%`;
  $("[data-progress-log]").textContent = m.message;
  $("[data-progress-title]").textContent = f >= 1 ? "Scan complete" : "Scanning";
}

// --- results -------------------------------------------------------------------------

const TITLES = {
  "L1.cross": "Exact copies of training rows", "L1.within_train": "Repeats inside the training file",
  "L1.within_test": "Repeats inside the test file", "L2.near": "Near copies", "L2.near_text": "Near-copy text",
  "L3.conflict": "Label conflicts", "L3.ceiling": "Accuracy ceiling", "L3.cross_contradict": "Contradicted test rows",
  "L4.after_test_start": "Training data from after the test period", "L4.time_shortcut": "Timestamp as a shortcut",
  "L5.shortcut": "One-column shortcut", "L5.identifier": "Identifier-like columns", "L6.group": "Group overlap",
};
const PILL = {
  high: ["HIGH", "bg-primary text-on-primary"],
  medium: ["MEDIUM", "bg-primary/15 text-primary border border-primary/40"],
  low: ["LOW", "bg-surface-container-high text-on-surface border border-outline-variant"],
  info: ["INFO", "bg-surface-container-high text-on-surface-variant border border-outline-variant"],
  ok: ["CLEAN", "bg-success/10 text-success border border-success/30"],
};

function headline(f) {
  switch (f.code) {
    case "L5.shortcut": return `${pct(f.value)} · ${f.details.top?.[0]?.feature ?? ""}`;
    case "L5.identifier": return `${f.value} column${f.value === 1 ? "" : "s"}`;
    case "L1.cross": return `${fmtInt(f.details.n_hit)} rows · ${pct(f.value)}`;
    case "L3.conflict": return `${fmtInt(f.details.n_groups)} groups · ${pct(f.value)}`;
    default: return pct(f.value);
  }
}

function examplesTable(f) {
  if (!f.examples?.length) return null;
  const rows = f.examples.map((x) => (typeof x === "object" ? x : { [f.code === "L6.group" ? "group" : "row"]: x }));
  const cols = Object.keys(rows[0]);
  const isRow = (k) => ["test", "train", "row"].includes(k) && f.code !== "L6.group";
  const cell = (k, v) => (isRow(k) && typeof v === "number" ? `#${fmtInt(v + 1)}` : typeof v === "number" ? (Number.isInteger(v) ? fmtInt(v) : v.toFixed(4)) : String(v));
  const table = h("table", { class: "w-full text-left font-mono-data-sm text-mono-data-sm border-collapse" },
    h("thead", {}, h("tr", { class: "border-b border-outline-variant bg-surface-container text-on-surface-variant sec-label" },
      ...cols.map((k) => h("th", { scope: "col", class: "py-2 px-space-sm", text: isRow(k) ? `${k} row` : k })))),
    h("tbody", { class: "divide-y divide-outline-variant/60 text-on-surface" },
      ...rows.map((r) => h("tr", {}, ...cols.map((k) => h("td", { class: "py-1.5 px-space-sm tabular", text: cell(k, r[k]) }))))));
  const note = cols.some(isRow) ? h("p", { class: "font-body-sm text-body-sm text-on-surface-variant mt-1", text: "Row numbers count data rows from 1, after the header." }) : null;
  return h("details", { class: "group" },
    h("summary", { class: "cursor-pointer font-mono-data-sm text-mono-data-sm text-on-surface underline underline-offset-4 hover:text-primary w-fit",
      text: `Show examples (${f.examples.length})` }),
    h("div", { class: "overflow-x-auto border border-outline-variant bg-surface-container-low mt-space-sm" }, table), note);
}

function card(f) {
  const [pillText, pillCls] = PILL[f.severity];
  return h("div", { class: "bg-surface-container-lowest border border-outline-variant p-space-md flex flex-col gap-space-sm" },
    h("div", { class: "flex flex-col md:flex-row md:items-center justify-between gap-space-sm" },
      h("div", { class: "flex items-center gap-space-sm flex-wrap" },
        h("span", { class: "font-mono-data-md text-mono-data-md font-medium text-on-surface", text: `${f.code.split(".")[0]} // ${TITLES[f.code] || f.code}` }),
        h("span", { class: `px-1.5 py-0.5 sec-label ${pillCls}`, text: pillText })),
      h("span", { class: `font-mono-data-md text-mono-data-md font-semibold tabular ${f.severity === "high" ? "text-primary" : "text-on-surface"}`, text: headline(f) })),
    h("p", { class: "font-body-md text-body-md text-on-surface-variant", text: f.summary }),
    examplesTable(f));
}

function metric(label, value, sub, hot) {
  return h("div", { class: "flex flex-col" },
    h("span", { class: "sec-label text-on-surface-variant", text: label }),
    h("span", { class: `font-mono-data-lg text-mono-data-lg font-semibold tabular ${hot ? "text-primary" : "text-on-surface"}`, text: value }),
    h("span", { class: "font-mono-data-sm text-mono-data-sm text-on-surface-variant", text: sub }));
}

function onResult(report) {
  state.running = false;
  state.report = report;
  updateSettings();
  const by = Object.fromEntries(report.findings.map((f) => [f.code, f]));
  const worst = report.worst;
  const banner = $("[data-banner]");
  banner.className = `border-2 p-space-lg flex flex-col gap-space-md ${worst === "high" ? "border-primary bg-primary/5"
    : worst === "medium" ? "border-primary/50 bg-primary/5" : "border-success/40 bg-success/5"}`;
  const [pillText, pillCls] = PILL[worst === "info" ? "ok" : worst];
  const pill = $("[data-banner-pill]");
  pill.className = `px-2 py-0.5 sec-label tracking-wider ${pillCls}`;
  pill.textContent = worst === "high" ? "HIGH LEAKAGE" : worst === "medium" ? "SOME LEAKAGE" : pillText === "CLEAN" ? "NO LEAKAGE FOUND" : pillText;
  $("[data-banner-meta]").textContent = `${fmtInt(report.meta.n_train)} train · ${fmtInt(report.meta.n_test)} test rows · ${report.meta.n_features} features`;
  const cross = by["L1.cross"];
  let title;
  if (cross && cross.value > 0) title = `${pct(cross.value)} of your test rows are exact copies of training rows.`;
  else {
    const top = report.findings.find((f) => f.severity === "high") || report.findings.find((f) => f.severity === "medium");
    title = top ? top.summary.charAt(0).toUpperCase() + top.summary.slice(1) + "." : "No exact copies, and nothing else above low severity.";
  }
  $("[data-banner-title]").textContent = title;
  const near = by["L2.near"] || by["L2.near_text"];
  const conf = by["L3.conflict"];
  const short = by["L5.shortcut"];
  $("[data-scoreboard]").replaceChildren(
    metric("Exact copies", cross ? fmtInt(cross.details.n_hit) : "—", cross ? `${pct(cross.value)} of test rows` : "", cross?.severity === "high"),
    metric("Near copies", near ? pct(near.value) : "—", near ? (near.details.vs_train_baseline ? `×${near.details.vs_train_baseline.toFixed(2)} train baseline · ${fmtInt(near.details.n_checked)} rows checked` : `${fmtInt(near.details.n_checked)} rows checked`) : "no numeric or text columns", near?.severity === "high"),
    metric("Label conflicts", conf ? fmtInt(conf.details.n_groups) : "—", conf ? "groups of identical rows, different labels" : "", conf?.severity === "high"),
    metric("Best single feature", short ? pct(short.value) : "—", short ? `balanced accuracy · ${short.details.top[0]?.feature ?? ""}` : "", short?.severity === "high"));
  const order = ["L1.cross", "L1.within_train", "L1.within_test", "L2.near", "L2.near_text", "L3.conflict", "L3.ceiling",
    "L3.cross_contradict", "L4.after_test_start", "L4.time_shortcut", "L5.shortcut", "L5.identifier", "L6.group"];
  const findings = order.filter((c) => by[c]).map((c) => by[c]);
  $("[data-cards]").replaceChildren(...findings.map(card));
  const flagged = findings.filter((f) => f.severity === "high" || f.severity === "medium").length;
  $("[data-count]").textContent = `${findings.length} findings · ${flagged} flagged`
    + (report.meta.skipped.length ? ` · not applicable: ${report.meta.skipped.join(", ")}` : "");
  $("[data-results-meta]").textContent = `Scanned in this tab · engine ${report.meta.version}`;
  const r = $("[data-results]");
  r.classList.replace("hidden", "flex");
  $("[data-download-clean]").disabled = !(cross && cross.details.n_hit > 0);
  $("[data-download-clean] span").textContent = cross && cross.details.n_hit > 0
    ? `Download test set without the ${fmtInt(cross.details.n_hit)} exact copies (CSV)` : "No exact copies to remove";
  r.scrollIntoView({ behavior: "smooth", block: "start" });
}

function save(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("[data-download-report]").addEventListener("click", () => {
  if (state.report) save("leakcheck-report.json", JSON.stringify(state.report, null, 2), "application/json");
});
$("[data-download-clean]").addEventListener("click", () => {
  $("[data-clean-note]").textContent = "Removing copies and re-checking the result…";
  worker.postMessage({ type: "clean" });
});

function onCleaned(m) {
  const note = $("[data-clean-note]");
  if (!m.verified) {
    note.textContent = `The cleaned file still had ${m.remaining} exact copies when re-checked, so it was not saved. Please report this.`;
    return;
  }
  const base = (state.files.test?.name || "test.csv").replace(/\.csv$/i, "");
  save(`${base}-no-exact-copies.csv`, m.csv, "text/csv");
  note.textContent = `Removed ${fmtInt(m.removed)} rows, kept ${fmtInt(m.kept)}. The saved file was re-read and re-checked: 0 exact copies remain. For near-copies across every row, use leakcheck clean from the Python package.`;
}

// --- errors ---------------------------------------------------------------------------

function setErrorText(message) {
  const el = $("[data-error]");
  const text = String(message);
  el.textContent = text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".");
  el.classList.remove("hidden");
}
function showError(message) {
  state.running = false;
  setErrorText(message);
  $("[data-progress]").classList.replace("flex", "hidden");
  updateSettings();
}
function clearError() { $("[data-error]").classList.add("hidden"); }
function hideResults() { $("[data-results]").classList.replace("flex", "hidden"); }
