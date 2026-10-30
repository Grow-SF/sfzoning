import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { CATEGORIES, CATEGORY_COLORS, CATEGORY_LABELS, fillColorExpression } from "@/lib/palette";
import { confidenceLabel, lookupRule, type Rule } from "@/lib/rules";
import { threePlusShare, type DistrictStats } from "@/lib/stats";
import { ERAS } from "@/lib/eras";
import { districtSource, districtStatsUrl } from "@/lib/tiles";
import { buildRules, parseCsv } from "../scripts/build-rules.mjs";

const PIPELINE_CATEGORIES = [
  "single_family",
  "two_unit",
  "three_unit",
  "multi_family",
  "mixed",
  "commercial",
  "industrial",
  "public",
  "unrestricted",
  "unknown",
];

describe("palette", () => {
  it("covers every pipeline category with a distinct colour and a label", () => {
    expect([...CATEGORIES].sort()).toEqual([...PIPELINE_CATEGORIES].sort());
    for (const c of PIPELINE_CATEGORIES) {
      expect(CATEGORY_COLORS[c as keyof typeof CATEGORY_COLORS]).toMatch(/^#[0-9a-f]{6}$/i);
      expect(CATEGORY_LABELS[c as keyof typeof CATEGORY_LABELS]?.trim()).toBeTruthy();
    }
    const colors = Object.values(CATEGORY_COLORS).map((c) => c.toLowerCase());
    expect(new Set(colors).size).toBe(colors.length);
  });
  it("builds a match expression that falls back to the unknown colour", () => {
    const expr = fillColorExpression() as unknown[];
    expect(expr[0]).toBe("match");
    expect(expr[expr.length - 1]).toBe(CATEGORY_COLORS.unknown);
    for (const c of PIPELINE_CATEGORIES) expect(expr).toContain(c);
  });
});

describe("district tile paths", () => {
  it("follow the pipeline's out/<series>/districts/<series>-districts.pmtiles layout", () => {
    expect(districtSource("1948")).toMatch(/\/1948\/districts\/1948-districts\.pmtiles$/);
    expect(districtSource("today")).toMatch(/\/today\/districts\/today-districts\.pmtiles$/);
    expect(districtStatsUrl("1948")).toMatch(/^https?:\/\/.*\/1948\/districts\/stats\.json$/);
  });
  it("gives every era, including today, a district series", () => {
    for (const e of ERAS) expect(e.districtSeries).toBeTruthy();
    expect(ERAS.find((e) => e.id === "today")!.districtSeries).toBe("today");
  });
});

describe("build-rules", () => {
  it("parses quoted CSV with embedded commas and quotes", () => {
    const rows = parseCsv('a,b\n"x, y","say ""hi"""\n');
    expect(rows).toEqual([
      ["a", "b"],
      ["x, y", 'say "hi"'],
    ]);
  });

  const csv = fs.readFileSync(path.resolve(__dirname, "../../docs/research/zoning-districts.csv"), "utf8");
  const rules = buildRules(csv) as Rule[];

  it("emits the rules.json schema", () => {
    expect(rules.length).toBeGreaterThan(50);
    for (const r of rules) {
      expect(typeof r.era_start).toBe("number");
      expect(r.era_end === null || typeof r.era_end === "number").toBe(true);
      expect(typeof r.code).toBe("string");
      expect(r.code.length).toBeGreaterThan(0);
      for (const k of ["name", "category", "allowed_uses", "max_units_per_lot", "height_limit_ft", "source", "confidence"] as const)
        expect(typeof r[k]).toBe("string");
      expect(["high", "medium", "low"]).toContain(r.confidence);
    }
  });
  it("turns 'present' and 'unknown' era ends into open-ended rules", () => {
    expect(rules.find((r) => r.code === "RH-1")!.era_end).toBeNull();
    expect(rules.find((r) => r.code === "RC-1")!.era_end).toBeNull();
  });

  it("looks rules up by era year and code", () => {
    expect(lookupRule(rules, 1921, "2R")!.era_start).toBe(1921);
    expect(lookupRule(rules, 1948, "1R")!.era_start).toBe(1946);
    expect(lookupRule(rules, 1948, "1R")!.max_units_per_lot).toMatch(/^1/);
    // An era's end year belongs to the next era.
    expect(lookupRule(rules, 1960, "C-2")!.era_start).toBe(1960);
    expect(lookupRule(rules, 1960, "1R")).toBeUndefined();
    expect(lookupRule(rules, 2026, "RH-2")!.name).toMatch(/RH-2|Two/i);
    expect(lookupRule(rules, 1985, "C-3-O")!.era_start).toBe(1985);
    // Slash-joined codes in the CSV match each member.
    expect(lookupRule(rules, 2026, "RTO-M")!.code).toBe("RTO / RTO-M");
    expect(lookupRule(rules, 2026, "NOPE")).toBeUndefined();
    expect(lookupRule(rules, 2026, "")).toBeUndefined();
  });
});

describe("confidenceLabel", () => {
  it("words a 0–1 map-reading score", () => {
    expect(confidenceLabel(0.99)).toBe("High");
    expect(confidenceLabel(0.7)).toBe("Medium");
    expect(confidenceLabel(0.3)).toBe("Low");
    expect(confidenceLabel(0)).toBe("Not read");
    expect(confidenceLabel(undefined)).toBe("Not read");
    expect(confidenceLabel("0.9")).toBe("High");
  });
});

describe("threePlusShare", () => {
  const z = { parcels: 0, area_sqft: 0 };
  const stats = (o: Partial<Record<string, number>>): DistrictStats =>
    Object.fromEntries(PIPELINE_CATEGORIES.map((c) => [c, { ...z, parcels: o[c] ?? 0 }])) as unknown as DistrictStats;
  it("is the share of residential parcels that allow three or more homes", () => {
    expect(threePlusShare(stats({ single_family: 50, two_unit: 25, three_unit: 5, multi_family: 10, mixed: 10, commercial: 999 }))).toBeCloseTo(0.25);
  });
  it("is null with no residential parcels", () => {
    expect(threePlusShare(stats({ commercial: 4 }))).toBeNull();
  });
});
