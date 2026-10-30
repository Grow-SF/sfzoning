export type Era = {
  id: string;
  label: string;
  year: number;
  scanSeries: string[];
  districtSeries?: string;
  /** Ids into content/sources.ts (which mirrors docs/research/sources.md) */
  sources: string[];
};

// Height and special-use series are not shown as scans in the story.
export const ERAS: Era[] = [
  { id: "1921", label: "1921", year: 1921, scanSeries: ["1921"], sources: ["sf-zoning-1921-10", "use-of-property-1921-legend"] },
  { id: "1929", label: "1929", year: 1929, scanSeries: ["1929"], sources: ["charter-1929"] },
  { id: "1935", label: "1935", year: 1935, scanSeries: ["1935"], sources: ["sf-zoning-1935"] },
  { id: "1948", label: "1948", year: 1948, scanSeries: ["1948"], sources: ["rumsey-1948-atlas", "planning-code-1947"] },
  { id: "1960", label: "1960", year: 1960, scanSeries: ["1960"], sources: ["planning-code-1960"] },
  { id: "1964", label: "1964", year: 1964, scanSeries: ["1964"], sources: ["1964-sf-zoning-partial"] },
  { id: "1968", label: "1968", year: 1968, scanSeries: ["1968"], sources: ["map-legend-1968"] },
  { id: "1972", label: "1972", year: 1972, scanSeries: ["1972"], sources: ["map-legend-1972"] },
  { id: "1975", label: "1975", year: 1975, scanSeries: ["1975"], sources: ["map-legend-1975"] },
  { id: "1980", label: "1980", year: 1980, scanSeries: ["1980"], sources: ["map-legend-1980"] },
  { id: "1980-1990", label: "1980–90", year: 1985, scanSeries: ["1980-1990-zoning"], districtSeries: "1980-1990-zoning", sources: ["zoning-map-book-2000"] },
  { id: "1990-1996", label: "1990–96", year: 1993, scanSeries: ["1990-1996-zoning"], districtSeries: "1990-1996-zoning", sources: ["zoning-map-book-2000"] },
  { id: "1996-2000", label: "1996–2000", year: 1998, scanSeries: ["1996-2000-zoning"], districtSeries: "1996-2000-zoning", sources: ["zoning-map-book-2000"] },
  { id: "2000s", label: "2000s", year: 2000, scanSeries: ["2000-digital-zoning"], districtSeries: "2000-digital-zoning", sources: ["zoning-map-book-2000"] },
  { id: "today", label: "Today", year: 2026, scanSeries: [], sources: ["sfplanning-has-2020", "sfplanning-family-zoning"] },
];
