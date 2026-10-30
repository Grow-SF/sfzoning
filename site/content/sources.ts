// Source registry. Every entry mirrors a row of docs/research/sources.md; ids are the
// archive.org identifier where there is one, otherwise a short slug. Steps and eras cite
// these ids, and tests/sources.test.ts checks that every id resolves.
import type { Step } from "./schema";
import type { Era } from "@/lib/eras";

export type Source = {
  /** Short label for step cards. */
  short: string;
  /** Full citation for the sources list. */
  title: string;
  kind: "primary" | "secondary";
  url?: string;
  /** Where the copy is, when it is not a public link (e.g. the owner's scans). */
  where?: string;
  note?: string;
};

const ia = (id: string) => `https://archive.org/details/${id}`;
const OWNER_SCANS = "Owner's scans, SF Planning Department";

export const SOURCES: Record<string, Source> = {
  // ---- 1921–1935 ----
  "sf-zoning-1921-10": {
    short: "Building Zone Ordinance, Ord. 5464, 1921",
    title: "Building Zone Ordinance, Ord. No. 5464 (N.S.), City and County of San Francisco, approved Oct. 3, 1921",
    kind: "primary",
    url: ia("sf-zoning-1921-10"),
    where: `${OWNER_SCANS} (pipeline/data/raw/1921); SFPL call no. 917.946 Sa52b 1921`,
  },
  "use-of-property-1921-legend": {
    short: "1921 zone map legend",
    title: "“Map Designations for Use of Property Zone Maps,” legend and index map, 1921",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1921)`,
  },
  "charter-1929": {
    short: "Charter Art. XVIII, 1928–29",
    title: "Charter Article XVIII (City Planning Commission), approved Nov. 6, 1928, ratified Jan. 17, 1929, with a reprint of Ord. 5464 as amended",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1929)`,
  },
  "sf-zoning-1935": {
    short: "Zone maps, March 1935",
    title: "Building Zone Ordinance and Zone Maps, March 1935",
    kind: "primary",
    url: ia("sf-zoning-1935"),
  },
  // ---- 1946–1957 ----
  "planning-code-1947": {
    short: "City Planning Code, July 1947, Sec. 99",
    title: "City Planning Code, amended and corrected as of July 1947, pp. 24–27, incl. Sec. 99 (Ord. 4162, approved Nov. 26, 1946)",
    kind: "primary",
    where: "pipeline/data/raw/1948/originals/planning-code.jpg",
  },
  cityplanningcod1948sanf_0: {
    short: "City Planning Code to Dec. 1948",
    title: "City Planning Code with amendments to Dec. 1, 1948 (SFPL copy)",
    kind: "primary",
    url: ia("cityplanningcod1948sanf_0"),
  },
  "rumsey-1948-atlas": {
    short: "1948 Zoning Map Book, David Rumsey Map Collection",
    title: "Zoning Map Book, San Francisco, 1948. © Cartography Associates, David Rumsey Map Collection",
    kind: "primary",
    note: "Source of the 1948 map images.",
  },
  "san-francisco-planning-code-1957": {
    short: "Planning Code to Dec. 1957, Art. 4.5",
    title: "Planning Code with amendments to Dec. 1, 1957, incl. Art. 4.5 off-street parking",
    kind: "primary",
    url: ia("san-francisco-planning-code-1957"),
  },
  // ---- 1960–1975 ----
  "planning-code-1960": {
    short: "City Planning Code, 1960; Summary of Zoning Provisions",
    title: "City Planning Code, effective May 2, 1960, and “Summary of Zoning Provisions of the City Planning Code,” March 1, 1960",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1960/originals/1960.pdf, pp. 1–2)`,
  },
  "1964-sf-zoning-partial": {
    short: "1960 code to July 1964; Art. 2.5",
    title: "City Planning Code with amendments to July 10, 1964: summary chart and Art. 2.5 Special Height Districts (Ord. 35-64)",
    kind: "primary",
    url: ia("1964-sf-zoning-partial"),
  },
  "map-legend-1968": {
    short: "Zoning map legend, Oct. 1968",
    title: "Zoning map book, amendments to Oct. 1, 1968: cover and legend",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1968)`,
  },
  "map-legend-1972": {
    short: "Zoning map legend, Oct. 1970",
    title: "Zoning map book (1972 series), legend with amendments to Oct. 10, 1970",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1972)`,
  },
  "map-legend-1975": {
    short: "Zoning and height maps, Feb. 1975",
    title: "Zoning map book, amendments to Feb. 2, 1975: use legend and Height & Bulk sheet 1H",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1975)`,
  },
  proposedheightbu2519sanf: {
    short: "Proposed height & bulk controls, 1972",
    title: "Proposed Height and Bulk Controls, Dept. of City Planning staff report, May 25, 1972",
    kind: "primary",
    url: ia("proposedheightbu2519sanf"),
  },
  "gocodebook-sec-252": {
    short: "Planning Code Sec. 252 history (Ord. 234-72)",
    title: "Planning Code Sec. 252, Classes of Height and Bulk Districts; history note: Ord. 234-72, App. 8/18/72 (third-party mirror)",
    kind: "secondary",
    url: "https://gocodebook.com/library/us/ca/san-francisco-planning/article-2.5-height-and-bulk-districts/sec.-252.-classes-of-height-and-bulk-districts",
  },
  "spur-high-rise-hegira": {
    short: "SPUR, “High-Rise Hegira,” 1999",
    title: "SPUR, “High-Rise Hegira,” The Urbanist, July 1, 1999",
    kind: "secondary",
    url: "https://www.spur.org/publications/urbanist-article/1999-07-01/high-rise-hegira",
  },
  // ---- 1977–1978 ----
  draftenvironment1319sanf: {
    short: "Draft EIR, residential rezoning, 1977",
    title: "Draft Environmental Impact Report, residential zoning (EE 76.182), Dept. of City Planning, Dec. 13, 1977",
    kind: "primary",
    url: ia("draftenvironment1319sanf"),
  },
  finalenvironment2719sanf: {
    short: "Final EIR, residential rezoning, 1978, p. 122",
    title: "Final Environmental Impact Report, residential zoning amendments, June 27, 1978 (p. 122)",
    kind: "primary",
    url: ia("finalenvironment2719sanf"),
  },
  "map-legend-1980": {
    short: "Zoning map legend, Nov. 1978",
    title: "Zoning map book (1980 series), legend with RH/RM/RC districts, amendments to Nov. 6, 1978",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1980)`,
  },
  "sbuss-when-did-things-go-wrong": {
    short: "S. Buss, “When did things go wrong in San Francisco?”",
    title: "Steven Buss, “When did things go wrong in San Francisco?” (Substack)",
    kind: "secondary",
    url: "https://sbuss.substack.com/p/when-did-things-go-wrong-in-san-francisco",
  },
  // ---- 1984–1990 ----
  C124887752: {
    short: "Downtown Plan, Ord. 414-85",
    title: "Downtown Plan ordinance, Ord. 414-85, effective Oct. 17, 1985 (Sec. 124 FAR table)",
    kind: "primary",
    url: ia("C124887752"),
  },
  "spur-prop-m": {
    short: "SPUR, “Proposition M and the downtown growth battle,” 1999",
    title: "SPUR, “Proposition M and the Downtown Growth Battle,” The Urbanist, July 1, 1999",
    kind: "secondary",
    url: "https://www.spur.org/publications/urbanist-article/1999-07-01/proposition-m-and-downtown-growth-battle",
  },
  "zoning-map-book-2000": {
    short: "Zoning map book introduction, Sept. 2000",
    title: "San Francisco Zoning Maps, introduction (American Legal Publishing / SF Planning), Sept. 2000",
    kind: "primary",
    where: `${OWNER_SCANS} (pipeline/data/raw/1980 Plus)`,
  },
  // ---- 2008–2026 ----
  "sfplanning-has-2020": {
    short: "SF Planning, “Regulation of Housing Development,” 2020",
    title: "SF Planning, Housing Affordability Strategies: “Regulation of Housing Development in San Francisco,” 2020",
    kind: "secondary",
    url: "https://sfplanning.s3.amazonaws.com/default/files/plans-and-programs/housing/affordability-strategy/HAS_Regulation_of_Housing_Development_in_SF_Final.pdf",
  },
  "sfplanning-market-octavia": {
    short: "SF Planning, Market & Octavia Area Plan",
    title: "SF Planning archive, Market & Octavia Area Plan (approved April 2008, effective May 30, 2008)",
    kind: "secondary",
    url: "https://sfgov.org/sfplanningarchive/market-octavia-area-plan",
  },
  "sfplanning-eastern-neighborhoods": {
    short: "Eastern Neighborhoods adoption package, 2008",
    title: "SF Planning, Eastern Neighborhoods adoption package, 2008",
    kind: "secondary",
    url: "https://archives.sfplanning.org/documents/1248-EN_ADOPTION_PACKAGE_VOLUME2a.pdf",
  },
  "ord-210-22-summary": {
    short: "Ord. 210-22 code summary (fourplexes)",
    title: "SF Planning code summary, 2021-010762PCA, Four-Unit Density Exception (Ord. 210-22)",
    kind: "primary",
    url: "https://sfplanning.org/sites/default/files/documents/legis/code-summaries/2021-010762PCA-4_Unit_Density_Exception.pdf",
  },
  "ord-248-23": {
    short: "Ord. 248-23, Housing Production",
    title: "Ord. 248-23, Housing Production (Board of Supervisors, Dec. 2023)",
    kind: "primary",
    url: "https://sfbos.org/sites/default/files/o0248-23.pdf",
  },
  "sfplanning-family-zoning": {
    short: "SF Planning, Family Zoning Plan",
    title: "SF Planning, “San Francisco Family Zoning Plan” (Board Files 250700/250701; effective Jan. 12, 2026)",
    kind: "secondary",
    url: "https://sfplanning.org/sf-family-zoning-plan",
  },
  "kqed-family-zoning": {
    short: "KQED on the Family Zoning Plan vote, Dec. 2025",
    title: "KQED News, report on the Board's 7–4 vote on the Family Zoning Plan, Dec. 2, 2025",
    kind: "secondary",
    note: "KQED article 12065708.",
  },
  "datasf-blocks": {
    short: "DataSF parcels and blocks",
    title: "DataSF, parcels and street centerlines (basis of the modern block outlines)",
    kind: "primary",
  },
};

/** Problems with source ids cited by steps and eras; empty means every id resolves. */
export function unresolvedSources(steps: Pick<Step, "id" | "sources">[], eras: Pick<Era, "id" | "sources">[]): string[] {
  const out: string[] = [];
  for (const s of steps) for (const id of s.sources) if (!SOURCES[id]) out.push(`step ${s.id}: unknown source "${id}"`);
  for (const e of eras) for (const id of e.sources) if (!SOURCES[id]) out.push(`era ${e.id}: unknown source "${id}"`);
  return out;
}

/** Source ids in first-cited order across the story, then the eras. */
export function citedSourceIds(steps: Pick<Step, "sources">[], eras: Pick<Era, "sources">[] = []): string[] {
  return Array.from(new Set([...steps.flatMap((s) => s.sources), ...eras.flatMap((e) => e.sources)])).filter((id) => SOURCES[id]);
}
