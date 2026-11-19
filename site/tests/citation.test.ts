import { describe, expect, it } from "vitest";
import { DATA_CREDIT } from "../lib/credits";
import { citations } from "../lib/citation.mjs";

const REL = { id: "2026-11-18", date: "2026-11-18" };

describe("citations", () => {
  it("builds the APA entry for a release", () => {
    expect(citations(REL).apa).toBe(
      "Bacio, S. (2026). A History of San Francisco’s Zoning (Version 2026-11-18) [Data set]. GrowSF. https://sfzoning.com/data/releases/2026-11-18",
    );
  });

  it("builds the Chicago entry for a release", () => {
    expect(citations(REL).chicago).toBe(
      "Bacio, Steven. 2026. A History of San Francisco’s Zoning. Data release 2026-11-18. GrowSF. https://sfzoning.com/data/releases/2026-11-18.",
    );
  });

  it("builds the BibTeX entry exactly", () => {
    expect(citations(REL).bibtex).toBe(
      [
        "@misc{bacio_sfzoning_2026,",
        "  author    = {Bacio, Steven},",
        "  title     = {A History of San Francisco’s Zoning},",
        "  year      = {2026},",
        "  version   = {2026-11-18},",
        "  publisher = {GrowSF},",
        "  url       = {https://sfzoning.com/data/releases/2026-11-18},",
        "  note      = {Data set, CC BY-NC-SA 4.0}",
        "}",
      ].join("\n"),
    );
  });

  it("builds the CFF file with the release fields and the credit line", () => {
    const cff = citations(REL).cff.split("\n");
    for (const line of [
      "cff-version: 1.2.0",
      "type: dataset",
      'title: "A History of San Francisco’s Zoning"',
      "  - family-names: Bacio",
      "    given-names: Steven",
      'version: "2026-11-18"',
      "date-released: 2026-11-18",
      'url: "https://sfzoning.com/data/releases/2026-11-18"',
      "license: CC-BY-NC-SA-4.0",
    ]) {
      expect(cff).toContain(line);
    }
    expect(cff.some((l) => l.startsWith("message:") && l.includes(DATA_CREDIT))).toBe(true);
  });

  it("uses the full release id for the url and version, and the date's year", () => {
    const c = citations({ id: "2026-11-18-2", date: "2026-11-18" });
    expect(c.url).toBe("https://sfzoning.com/data/releases/2026-11-18-2");
    expect(c.apa).toContain("(Version 2026-11-18-2)");
    expect(c.apa).toContain("(2026)");
    expect(c.bibtex).toContain("version   = {2026-11-18-2},");
    expect(c.bibtex).toContain("year      = {2026},");
    expect(c.cff).toContain('version: "2026-11-18-2"');
  });

  it("keeps the credit text equal to DATA_CREDIT", () => {
    expect(citations(REL).cff).toContain(DATA_CREDIT);
  });
});
