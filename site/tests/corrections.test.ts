import { afterEach, describe, expect, it, vi } from "vitest";
import {
  addCorrection,
  clearCorrections,
  codesForYear,
  correctionsFile,
  FENCE,
  getCorrections,
  issueBody,
  issueUrl,
  MAX_URL,
  removeCorrection,
  resetCorrections,
  type Correction,
} from "@/lib/corrections";
import type { Rule } from "@/lib/rules";

const c = (over: Partial<Correction> = {}): Correction => ({
  era: "1980",
  series: "1980",
  blklot: "2842007",
  shown: "RH-1",
  proposed: "P",
  note: "Laguna Honda",
  ...over,
});

function stubStorage(initial: Record<string, string> = {}) {
  const store = { ...initial };
  vi.stubGlobal("window", {
    localStorage: { getItem: (k: string) => store[k] ?? null, setItem: (k: string, v: string) => (store[k] = v) },
  });
  return store;
}

describe("the corrections list", () => {
  afterEach(() => {
    resetCorrections();
    vi.unstubAllGlobals();
  });

  it("keeps one correction per lot and era between visits", () => {
    const store = stubStorage();
    addCorrection(c());
    addCorrection(c({ proposed: "RH-1(D)" }));            // same lot, same era: replaced
    addCorrection(c({ era: "1975", series: "1975" }));    // another era: kept
    expect(getCorrections().map((x) => `${x.era} ${x.proposed}`)).toEqual(["1980 RH-1(D)", "1975 P"]);
    resetCorrections();
    expect(JSON.parse(store["sfz-corrections"])).toHaveLength(2);
    removeCorrection("1975", "2842007");
    expect(getCorrections()).toHaveLength(1);
    clearCorrections();
    expect(getCorrections()).toEqual([]);
  });

  it("starts empty when storage is blocked or holds junk", () => {
    stubStorage({ "sfz-corrections": "{not json" });
    expect(getCorrections()).toEqual([]);
    resetCorrections();
    stubStorage({ "sfz-corrections": JSON.stringify([{ nope: 1 }, c()]) });
    expect(getCorrections()).toHaveLength(1);
  });
});

describe("the issue", () => {
  it("has a table for people and a fenced block the pipeline can read back", () => {
    const body = issueBody([c(), c({ blklot: "3179010", shown: "", proposed: "P", note: "a | b" })]);
    expect(body).toContain("| 1980 | 2842/007 | RH-1 | P | Laguna Honda |");
    expect(body).toContain("| 1980 | 3179/010 | none | P | a / b |");
    const block = body.split("```" + FENCE + "\n")[1].split("\n```")[0];
    expect(JSON.parse(block)).toEqual([c(), c({ blklot: "3179010", shown: "", proposed: "P", note: "a | b" })]);
  });

  it("opens a pre-filled GitHub issue, or falls back to the file when the link would be too long", () => {
    const url = issueUrl([c()])!;
    expect(url.startsWith("https://github.com/Grow-SF/sfzoning/issues/new?")).toBe(true);
    const q = new URL(url).searchParams;
    expect(q.get("title")).toBe("Map corrections: 1 lot (1980)");
    expect(q.get("labels")).toBe("correction");
    expect(q.get("body")).toBe(issueBody([c()]));
    const many = Array.from({ length: 400 }, (_, i) => c({ blklot: String(1000000 + i) }));
    expect(issueUrl(many)).toBeNull();
    expect(correctionsFile(many).length).toBeGreaterThan(MAX_URL);
    expect(issueUrl([])).toBeNull();
  });
});

describe("codesForYear", () => {
  const r = (code: string, era_start: number, era_end: number | null) => ({ code, era_start, era_end }) as Rule;
  it("lists the codes in force that year, numerically sorted", () => {
    const rules = [r("R-1", 1960, 1978), r("RH-1 / RH-1(D)", 1978, null), r("RH-10", 1978, null), r("RH-2", 1978, null)];
    expect(codesForYear(rules, 1980)).toEqual(["RH-1", "RH-1(D)", "RH-2", "RH-10"]);
    expect(codesForYear(rules, 1964)).toEqual(["R-1"]);
  });
});
