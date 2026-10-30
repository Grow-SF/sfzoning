import { describe, expect, it } from "vitest";
import { ERAS } from "@/lib/eras";
import { validateStory, type Step } from "@/content/schema";
import { STORY } from "@/content/story";

const base: Step = {
  id: "s1",
  era: "1921",
  view: "scan",
  camera: { center: [-122.44, 37.76], zoom: 12 },
  title: "A title",
  body: [{ kind: "p", text: "Body." }],
  sources: ["sf-zoning-1921-10"],
};

describe("validateStory", () => {
  it("accepts the shipped story", () => {
    expect(validateStory(STORY, ERAS)).toEqual([]);
  });
  it("catches an unknown era", () => {
    const errs = validateStory([{ ...base, era: "1999" }], ERAS);
    expect(errs.join("\n")).toMatch(/s1.*unknown era "1999"/);
  });
  it("catches duplicate ids, bad views, empty titles and missing sources", () => {
    const errs = validateStory([base, { ...base, view: "x" as Step["view"], title: " ", sources: [] }], ERAS).join("\n");
    expect(errs).toMatch(/duplicate id "s1"/);
    expect(errs).toMatch(/unknown view "x"/);
    expect(errs).toMatch(/empty title/);
    expect(errs).toMatch(/no sources/);
  });
  it("catches an out-of-range camera", () => {
    const errs = validateStory([{ ...base, camera: { center: [0, 0], zoom: 30 } }], ERAS).join("\n");
    expect(errs).toMatch(/camera/);
  });
});
