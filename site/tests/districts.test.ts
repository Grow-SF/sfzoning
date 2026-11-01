import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { CATEGORIES, CATEGORY_COLORS, CATEGORY_LABELS } from "@/lib/palette";
import { gunzipped, lotCard, lotCategory, NO_VALUE, parseManifest, type SharedDistricts } from "@/lib/shared";
import { confidenceLabel, lookupRule, type Rule } from "@/lib/rules";
import { isPartialMap, threePlusShare, type DistrictStats } from "@/lib/stats";
import { ERAS } from "@/lib/eras";
import { districtStatsUrl, publishedPath } from "@/lib/tiles";
import { hashedName, plan, uploadedBytesMatch } from "../scripts/publish-tiles.mjs";
import { buildRules, parseCsv } from "../scripts/build-rules.mjs";

const PIPELINE_CATEGORIES = [
  "single_family",
  "two_unit",
  "three_unit",
  "homes_low",
  "homes_mid",
  "homes_high",
  "homes_top",
  "commercial_light",
  "commercial_dense",
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
});

describe("shared archive", () => {
  const manifest = {
    version: 2,
    archive: "shared/districts-abc.pmtiles",
    table: "shared/districts-abc.table.gz",
    cards: "shared/districts-abc.cards.json.gz",
    categories: ["single_family", "commercial_dense"],
    lots: 3,
    lineSets: 1,
    eras: ["1929", "1948"],
  };
  // Two eras, three lots, one line set. 1929's map leaves out lot 2.
  const shared: SharedDistricts = {
    ...parseManifest(manifest)!,
    values: new Uint8Array([0, 1, NO_VALUE, 1, 1, 0, 1, 1]),
  };

  it("refuses a pointer it can't draw, so the site falls back to per-era archives", () => {
    expect(parseManifest(manifest)).not.toBeNull();
    expect(parseManifest({ ...manifest, version: 1 })).toBeNull();
    expect(parseManifest({ ...manifest, categories: ["brand_new_use"] })).toBeNull();
    expect(parseManifest({ ...manifest, table: undefined })).toBeNull();
    expect(parseManifest(null)).toBeNull();
  });
  it("finds a lot's category per era, and none where the map doesn't cover it", () => {
    expect(lotCategory(shared, 0, 1)).toBe("commercial_dense");
    expect(lotCategory(shared, 1, 2)).toBe("single_family");
    expect(lotCategory(shared, 0, 2)).toBeNull();
  });
  it("reads a lot's card for one era, leaving other eras' values alone", () => {
    const cards = {
      blklot: ["0001001", "0001002"],
      strings: ["C-2", "40-X", "RH-1"],
      eras: {
        "1948": { code: [0, null], height: [null, null], confidence: [0.9, null] },
        "1990-1996-zoning": { code: [2, 2], height: [1, 1], confidence: [0.5, 0.5] },
      },
    };
    expect(lotCard(cards, "1948", 0)).toEqual({ blklot: "0001001", code: "C-2", height_code: undefined, confidence: 0.9 });
    expect(lotCard(cards, "1990-1996-zoning", 1)).toEqual({ blklot: "0001002", code: "RH-1", height_code: "40-X", confidence: 0.5 });
  });
  it("unpacks a .gz file whether or not the host already did", async () => {
    const text = new TextEncoder().encode("hello");
    const packed = await new Response(new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
    expect(await gunzipped(packed)).toEqual(text);
    expect(await gunzipped(text.buffer)).toEqual(text);
  });
});

describe("published tile names", () => {
  const map = { "1975/districts/stats.json": "1975/districts/stats-abc123.json" };
  const withBase = (base: string, run: () => void) => {
    const was = process.env.NEXT_PUBLIC_TILES_BASE_URL;
    process.env.NEXT_PUBLIC_TILES_BASE_URL = base;
    try {
      run();
    } finally {
      if (was === undefined) delete process.env.NEXT_PUBLIC_TILES_BASE_URL;
      else process.env.NEXT_PUBLIC_TILES_BASE_URL = was;
    }
  };
  it("reads a changed file under its new name and an unchanged one where it always was", () => {
    withBase("https://tiles.example.org", () => {
      expect(publishedPath("1975/districts/stats.json", map)).toBe("1975/districts/stats-abc123.json");
      expect(publishedPath("1948/districts/stats.json", map)).toBe("1948/districts/stats.json");
    });
  });
  it("keeps the usual names on a local tile server, whose files are the pipeline's own output", () => {
    withBase("http://127.0.0.1:8787", () => expect(publishedPath("1975/districts/stats.json", map)).toBe("1975/districts/stats.json"));
  });
  it("puts the hash after the name, leaving only the last extension", () => {
    expect(hashedName("1975/districts/stats.json", "abc")).toBe("1975/districts/stats-abc.json");
    expect(hashedName("shared/districts-x.cards.json.gz", "abc")).toBe("shared/districts-x.cards.json-abc.gz");
  });
  it("never plans to replace a published file", () => {
    expect(plan("a/stats.json", "h1", undefined, "h1").action).toBe("keep");
    expect(plan("a/stats.json", "h1", undefined, null)).toEqual({ action: "upload", to: "a/stats.json", map: false });
    expect(plan("a/stats.json", "h2", undefined, "h1")).toEqual({ action: "upload", to: "a/stats-h2.json", map: true });
    expect(plan("a/stats.json", "h3", "a/stats-h2.json", "h2")).toEqual({ action: "upload", to: "a/stats-h3.json", map: true });
    expect(plan("a/stats.json", "h2", "a/stats-h2.json", "h2").action).toBe("keep");
  });
});

describe("district tile paths", () => {
  it("follow the pipeline's out/<series>/districts/ layout", () => {
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

describe("isPartialMap", () => {
  const stats = (parcels: number, unmapped?: number) =>
    ({ single_family: { parcels, area_sqft: 0 }, unmapped_parcels: unmapped }) as unknown as DistrictStats;
  it("flags a map that leaves out part of the city", () => {
    expect(isPartialMap(stats(200_292, 27_679))).toBe(true); // 1929
    expect(isPartialMap(stats(210_410, 17_561))).toBe(true); // 1990-96
  });
  it("accepts a map that only misses scattered lots", () => {
    expect(isPartialMap(stats(223_252, 4_719))).toBe(false); // 1921, the lowest of the whole maps
    expect(isPartialMap(stats(227_699, 272))).toBe(false); // today
  });
  it("makes no claim without the count", () => {
    expect(isPartialMap(stats(100))).toBe(false);
  });
});

describe("threePlusShare", () => {
  const z = { parcels: 0, area_sqft: 0 };
  const stats = (o: Partial<Record<string, number>>): DistrictStats =>
    Object.fromEntries(PIPELINE_CATEGORIES.map((c) => [c, { ...z, parcels: o[c] ?? 0 }])) as unknown as DistrictStats;
  it("is the share of residential parcels that allow three or more homes", () => {
    expect(threePlusShare(stats({ single_family: 50, two_unit: 25, three_unit: 5, homes_low: 4, homes_mid: 3, homes_high: 2, homes_top: 1, commercial_dense: 999 }))).toBeCloseTo(15 / 90);
  });
  it("counts all homes_* steps and three_unit as three-plus, and adds one and two homes to the total", () => {
    expect(threePlusShare(stats({ three_unit: 1, homes_low: 1, homes_mid: 1, homes_high: 1, homes_top: 1 }))).toBe(1);
    expect(threePlusShare(stats({ single_family: 3, two_unit: 3, homes_top: 4 }))).toBeCloseTo(0.4);
    expect(threePlusShare(stats({ commercial_light: 5, industrial: 5, homes_top: 1 }))).toBe(1);
  });
  it("is null with no residential parcels", () => {
    expect(threePlusShare(stats({ commercial_dense: 4 }))).toBeNull();
  });
});

describe("uploadedBytesMatch", () => {
  const noWait = async () => {};
  const reads = (...hashes: (string | null)[]) => {
    const calls = { n: 0 };
    return { calls, read: async () => hashes[Math.min(calls.n++, hashes.length - 1)] };
  };

  it("waits out a store that hasn't caught up yet", async () => {
    const { read, calls } = reads(null, "old", "h1");
    expect(await uploadedBytesMatch(read, "h1", noWait)).toBe(true);
    expect(calls.n).toBe(3);
  });

  it("still fails on a real mismatch, after a bounded number of reads", async () => {
    const { read, calls } = reads("other");
    expect(await uploadedBytesMatch(read, "h1", noWait)).toBe(false);
    expect(calls.n).toBeLessThan(10);
    expect(calls.n).toBeGreaterThan(1);
  });
});
