// CSV reading and column typing, matching how leakcheck's Python side reads
// files (pandas.read_csv with skipinitialspace=True and default NA tokens).
//
// A table is { columns: string[], cols: Map<name, Column>, n } where a Column
// is { kind: "num", values: Float64Array } or { kind: "str", values: (string|null)[] }.

// pandas' default missing-value tokens
export const NA_TOKENS = new Set([
  "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
  "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
]);
const NUMBER = /^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/;
const INF = /^([+-]?)(inf|infinity)$/i;
const BOOL = { True: 1, TRUE: 1, true: 1, False: 0, FALSE: 0, false: 0 };

/** Parse RFC 4180 CSV text into a header and rows of raw string fields. */
export function parseCSV(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // BOM
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  let fieldStart = true; // still in leading whitespace of an unquoted field
  const n = text.length;
  for (let i = 0; i < n; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else inQuotes = false;
      } else field += c;
      continue;
    }
    if (c === '"' && fieldStart) { inQuotes = true; fieldStart = false; continue; }
    if (c === ",") { row.push(field); field = ""; fieldStart = true; continue; }
    if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      if (!(row.length === 1 && row[0] === "")) rows.push(row); // skip blank lines, like pandas
      row = []; field = ""; fieldStart = true;
      continue;
    }
    if (fieldStart && c === " ") continue; // skipinitialspace
    fieldStart = false;
    field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) throw new Error("the file is empty");
  const header = rows.shift();
  return { header, rows };
}

function numericValue(s) {
  if (NUMBER.test(s)) return Number(s);
  const m = INF.exec(s);
  if (m) return m[1] === "-" ? -Infinity : Infinity;
  return undefined;
}

/** Decide a column's kind from the raw strings of every file that shares it. */
export function columnKind(rawLists) {
  let sawValue = false;
  let allBool = true;
  let anyNA = false;
  for (const list of rawLists) {
    for (const s of list) {
      if (NA_TOKENS.has(s)) { anyNA = true; continue; }
      sawValue = true;
      if (!(s in BOOL)) allBool = false;
      if (!(s in BOOL) && numericValue(s) === undefined) return "str";
    }
  }
  if (!sawValue) return "num"; // all missing: pandas makes it float NaN
  if (allBool && !anyNA) return "bool";
  if (allBool) return "str"; // pandas leaves a bool column with gaps as object
  return "num";
}

export function toColumn(raw, kind) {
  if (kind === "num" || kind === "bool") {
    const v = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const s = raw[i];
      if (NA_TOKENS.has(s)) v[i] = NaN;
      else if (s in BOOL) v[i] = BOOL[s];
      else v[i] = numericValue(s);
    }
    return { kind: "num", values: v };
  }
  return { kind: "str", values: raw.map((s) => (NA_TOKENS.has(s) ? null : s.trim())) };
}

/** Raw column strings of a parsed CSV (short rows padded with "", like pandas NaN). */
export function rawColumns(parsed) {
  const { header, rows } = parsed;
  const out = new Map();
  header.forEach((name, j) => {
    const col = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) col[i] = rows[i][j] ?? "";
    out.set(name, col);
  });
  return out;
}

/** Build typed tables for train and test together, so a column gets one kind in both. */
export function typedPair(trainParsed, testParsed) {
  const a = new Set(trainParsed.header);
  const b = new Set(testParsed.header);
  const missing = [...a].filter((c) => !b.has(c)).concat([...b].filter((c) => !a.has(c)));
  if (missing.length) {
    const err = new Error(`train and test have different columns: ${missing.sort().join(", ")}`);
    err.code = "columns";
    err.columns = missing;
    throw err;
  }
  const rawTr = rawColumns(trainParsed);
  const rawTe = rawColumns(testParsed);
  const tr = new Map();
  const te = new Map();
  for (const name of trainParsed.header) {
    const kind = columnKind([rawTr.get(name), rawTe.get(name)]);
    tr.set(name, toColumn(rawTr.get(name), kind));
    te.set(name, toColumn(rawTe.get(name), kind));
  }
  return {
    train: { columns: trainParsed.header, cols: tr, n: trainParsed.rows.length },
    test: { columns: trainParsed.header, cols: te, n: testParsed.rows.length },
    raw: { train: rawTr, test: rawTe },
  };
}

/** Label values as strings, the way Python's Split.y_train reads them. */
export function labelStrings(col) {
  if (col.kind === "num") {
    const allInt = Array.prototype.every.call(col.values, (x) => Number.isNaN(x) || Number.isInteger(x));
    const anyNaN = Array.prototype.some.call(col.values, (x) => Number.isNaN(x));
    return Array.from(col.values, (x) => {
      if (Number.isNaN(x)) return "<NA>";
      if (allInt && !anyNaN) return String(x);
      return Number.isInteger(x) ? x.toFixed(1) : String(x);
    });
  }
  return col.values.map((s) => (s === null ? "<NA>" : s));
}
