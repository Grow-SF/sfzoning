import { describe, expect, it } from "vitest";
import { STORY } from "@/content/story";
import { heroCue } from "@/lib/eras";

const text = (id: string) =>
  STORY.find((s) => s.id === id)!
    .body.map((b) => b.text)
    .join(" ");

describe("story content flags", () => {
  it("the hero cue names the first step's era, not 1921", () => {
    expect(STORY[0].era).toBe("today");
    expect(heroCue(STORY[0])).toBe("Scroll to begin with today’s map");
    expect(heroCue({ era: "1948" })).toBe("Scroll to begin in 1948");
  });
  it("the opening does not claim apartments were legal on most residential land", () => {
    expect(text("today-opening")).not.toMatch(/most of the city’s residential land/);
    expect(text("today-opening")).toMatch(/legal across large parts of the city/);
  });
  it("the 1964 caps are not framed as a 40-foot floor", () => {
    const t = text(STORY.find((s) => s.era === "1972")!.id);
    expect(t).not.toMatch(/from 40 feet in the Marina/);
    expect(t).toMatch(/Anza Vista/);
  });
  it("Prop M makes no unsupported 'standard hook' claim", () => {
    expect(text("1986-prop-m")).not.toMatch(/standard hook/);
    expect(STORY.find((s) => s.id === "1986-prop-m")!.sources).toContain("spur-prop-m");
  });
});
