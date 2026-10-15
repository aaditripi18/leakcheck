// Node entry point used by the parity tests:
//   node site/engine/cli.mjs scan  --train a.csv --test b.csv --label y [--time t] [--group g] [--text x]
//                                  [--ignore a,b] [--sample N] [--checks L1,L3]
//   node site/engine/cli.mjs clean --train a.csv --test b.csv --label y --out cleaned.csv
import { readFileSync, writeFileSync } from "node:fs";
import { cleanTest, makeSplit, parseCSV, scan } from "./scan.js";

const [cmd, ...rest] = process.argv.slice(2);
const args = {};
for (let i = 0; i < rest.length; i += 2) args[rest[i].replace(/^--/, "")] = rest[i + 1];

const tr = parseCSV(readFileSync(args.train, "utf8"));
const te = parseCSV(readFileSync(args.test, "utf8"));
const s = makeSplit(tr, te, {
  label: args.label, time: args.time ?? null, group: args.group ?? null, text: args.text ?? null,
  ignore: args.ignore ? args.ignore.split(",") : [],
});
if (cmd === "scan") {
  const opts = {};
  if (args.sample) opts.sample = Number(args.sample);
  if (args.checks) opts.checks = args.checks.split(",");
  process.stdout.write(JSON.stringify(scan(s, opts)));
} else if (cmd === "clean") {
  const r = cleanTest(tr, te, s);
  writeFileSync(args.out, r.csv);
  process.stdout.write(JSON.stringify({ removed: r.removed.length, kept: r.kept, verified: r.verified }));
} else {
  console.error("usage: cli.mjs scan|clean ...");
  process.exit(3);
}
