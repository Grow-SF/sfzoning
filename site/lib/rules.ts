// Plain-language district rules, built from docs/research/zoning-districts.csv into
// public/data/rules.json by scripts/build-rules.mjs (run on predev/prebuild).

export type Rule = {
  era_start: number;
  /** null: still in force ("present") or end not known */
  era_end: number | null;
  code: string;
  name: string;
  category: string;
  allowed_uses: string;
  max_units_per_lot: string;
  height_limit_ft: string;
  notes: string;
  source: string;
  confidence: "high" | "medium" | "low";
};

export const RULES_URL = "/data/rules.json";

function codesOf(rule: Rule): string[] {
  return rule.code.split("/").map((c) => c.trim()).filter(Boolean);
}

function norm(code: string): string {
  return code.trim().toUpperCase();
}

/**
 * The rule in force in `year` for district `code`. An era runs from era_start up to (not
 * including) era_end, so a code that changed in 1960 resolves to the 1960 rule. When several
 * rules match, the most recently adopted wins.
 */
export function lookupRule(rules: Rule[], year: number, code: string | null | undefined): Rule | undefined {
  if (!code || !code.trim()) return undefined;
  const want = norm(code);
  let best: Rule | undefined;
  for (const r of rules) {
    if (r.era_start > year) continue;
    if (r.era_end !== null && year >= r.era_end) continue;
    if (!codesOf(r).some((c) => norm(c) === want)) continue;
    if (!best || r.era_start > best.era_start) best = r;
  }
  return best;
}

/** Words for the pipeline's 0–1 map-reading confidence. */
export function confidenceLabel(v: number | string | null | undefined): "High" | "Medium" | "Low" | "Not read" {
  const n = typeof v === "string" ? Number(v) : v;
  if (n == null || !Number.isFinite(n) || n <= 0) return "Not read";
  if (n >= 0.85) return "High";
  if (n >= 0.65) return "Medium";
  return "Low";
}

let rulesPromise:Promise<Rule[]> | null = null;

/** Fetches rules.json once; resolves to [] if it is missing. */
export function loadRules(): Promise<Rule[]> {
  if (!rulesPromise) {
    rulesPromise = fetch(RULES_URL)
      .then((r) => (r.ok ? (r.json() as Promise<Rule[]>) : []))
      .catch(() => []);
  }
  return rulesPromise;
}
