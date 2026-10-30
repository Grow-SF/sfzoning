import { describe, expect, it } from "vitest";
import { ERAS } from "@/lib/eras";
import { STORY } from "@/content/story";
import { SOURCES, unresolvedSources } from "@/content/sources";

describe("sources", () => {
  it("resolves every source id used by the story and the eras", () => {
    expect(unresolvedSources(STORY, ERAS)).toEqual([]);
  });

  it("reports an id that is not in the registry", () => {
    const step = { ...STORY[0], id: "x", sources: ["no-such-source"] };
    expect(unresolvedSources([step], [])).toEqual(['step x: unknown source "no-such-source"']);
    expect(unresolvedSources([], [{ id: "e", sources: ["nope"] }])).toEqual(['era e: unknown source "nope"']);
  });

  it("gives every source a title and a primary/secondary kind, and only https links", () => {
    for (const [id, s] of Object.entries(SOURCES)) {
      expect(s.title.trim(), id).not.toBe("");
      expect(["primary", "secondary"], id).toContain(s.kind);
      if (s.url) expect(s.url, id).toMatch(/^https:\/\//);
    }
  });

  it("cites the Final EIR on the 1978 step that quotes 180,000", () => {
    const steps = STORY.filter((s) => s.body.some((b) => b.text.includes("180,000")));
    expect(steps.length).toBeGreaterThan(0);
    for (const s of steps) expect(s.sources).toContain("finalenvironment2719sanf");
  });

  it("puts the 1948 steps on the Rumsey atlas source", () => {
    const steps = STORY.filter((s) => s.era === "1948");
    expect(steps.length).toBeGreaterThan(0);
    for (const s of steps) expect(s.sources).toContain("rumsey-1948-atlas");
  });
});
