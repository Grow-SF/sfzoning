// DRAFT COPY. Placeholder steps for the scroll engine (Task S2); not reviewed by the owner.
// Facts come from docs/research/zoning-history.md; every `sources` entry points into
// docs/research/sources.md. The full story arc replaces these in a later task.
import type { Step } from "./schema";

export const STORY: Step[] = [
  {
    id: "1921-first-ordinance",
    era: "1921",
    view: "scan",
    camera: { center: [-122.4376, 37.7555], zoom: 12.2 },
    title: "In 1921, apartments were legal almost everywhere",
    body: [
      {
        kind: "p",
        text: "San Francisco’s first zoning ordinance split the city into six districts. It said what could be built, not how big: no height limits, no minimum lot sizes, no cap on homes per lot.",
      },
      {
        kind: "p",
        text: "Apartments and flats were allowed by right in Second Residential, Commercial and Light Industrial land. Only Heavy Industrial banned dwellings outright.",
      },
      { kind: "stat", text: "6 districts, 0 density limits" },
    ],
    sources: ["Ord. 5464 (N.S.), Building Zone Ordinance, Oct. 3, 1921 (archive.org sf-zoning-1921-10)"],
  },
  {
    id: "1948-lot-rules",
    era: "1948",
    view: "both",
    camera: { center: [-122.4335, 37.7805], zoom: 14.2 },
    title: "By 1948, the first lot-size and coverage rules",
    body: [
      {
        kind: "p",
        text: "The 1948 maps still use the 1921 six-district system. What changed was the fine print. Section 99, approved in November 1946, was the city’s first rule about how much of a lot a home could cover.",
      },
      {
        kind: "quote",
        text: "Dwellings may cover no more than 65 per cent of the lot.",
      },
      {
        kind: "p",
        text: "New subdivision lots had to match the neighbourhood or be at least 33 feet wide. Resubdivided lots had to be at least 2,500 square feet. Rear yards of 25 feet were required in new subdivisions and 15 feet elsewhere.",
      },
      {
        kind: "p",
        text: "In 1955 a further amendment required one off-street parking space for every new home, after complaints about apartments built without garages.",
      },
      {
        kind: "p",
        text: "First Residential land, where a second home needed 5,000 square feet of open lot, was already single-family in practice. Most standard lots were 2,500 square feet.",
      },
    ],
    sources: [
      "City Planning Code, amended as of July 1947, Sec. 99 (Ord. 4162)",
      "Planning Code to Dec. 1, 1957, Art. 4.5 (archive.org san-francisco-planning-code-1957)",
    ],
  },
  {
    id: "1960-density-caps",
    era: "1960",
    view: "scan",
    camera: { center: [-122.4795, 37.7595], zoom: 14.6, bearing: -8 },
    title: "1960: the first caps on homes per lot",
    body: [
      {
        kind: "p",
        text: "The modern Planning Code took effect on May 2, 1960. For the first time, every residential district carried a number: how much lot area each home needed.",
      },
      { kind: "stat", text: "R-1: one home per lot. R-3: about six on a 25×100 lot." },
      {
        kind: "p",
        text: "It also banned new housing in industrial zones, which the 1921 code had allowed in Light Industrial.",
      },
    ],
    sources: ["City Planning Code, effective May 2, 1960; Summary of Zoning Provisions, Mar. 1, 1960"],
  },
];
