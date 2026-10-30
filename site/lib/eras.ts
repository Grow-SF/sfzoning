export type Era = {
  id: string;
  label: string;
  year: number;
  scanSeries: string[];
  districtSeries?: string;
  /** Pointers into docs/research/sources.md */
  sources: string[];
};

// Height and special-use series are not shown as scans in the story.
export const ERAS: Era[] = [
  { id: "1921", label: "1921", year: 1921, scanSeries: ["1921"], sources: ["1921 Building Zone Ordinance (Ord. 5464)"] },
  { id: "1929", label: "1929", year: 1929, scanSeries: ["1929"], sources: ["Charter Art. XVIII reprint of Ord. 5464"] },
  { id: "1935", label: "1935", year: 1935, scanSeries: ["1935"], sources: ["archive.org sf-zoning-1935"] },
  { id: "1948", label: "1948", year: 1948, scanSeries: ["1948"], sources: ["City Planning Code, amended as of July 1947 (Sec. 99)"] },
  { id: "1960", label: "1960", year: 1960, scanSeries: ["1960"], sources: ["City Planning Code, effective May 2, 1960"] },
  { id: "1964", label: "1964", year: 1964, scanSeries: ["1964"], sources: ["1964 zoning map legend; Art. 2.5 height districts"] },
  { id: "1968", label: "1968", year: 1968, scanSeries: ["1968"], sources: ["1968 zoning map legend (amendments to Oct 1, 1968)"] },
  { id: "1972", label: "1972", year: 1972, scanSeries: ["1972"], sources: ["1972 zoning map legend (amendments to Oct 10, 1970)"] },
  { id: "1975", label: "1975", year: 1975, scanSeries: ["1975"], sources: ["1975 zoning map legend (amendments to Feb 2, 1975)"] },
  { id: "1980", label: "1980", year: 1980, scanSeries: ["1980"], sources: ["1980 zoning map legend (RH/RM/RC families)"] },
  { id: "1980-1990", label: "1980–90", year: 1985, scanSeries: ["1980-1990-zoning"], districtSeries: "1980-1990-zoning", sources: ["Final EIR, residential rezoning, June 27, 1978"] },
  { id: "1990-1996", label: "1990–96", year: 1993, scanSeries: ["1990-1996-zoning"], districtSeries: "1990-1996-zoning", sources: ["Zoning map series 1990–96"] },
  { id: "1996-2000", label: "1996–2000", year: 1998, scanSeries: ["1996-2000-zoning"], districtSeries: "1996-2000-zoning", sources: ["Zoning map series 1996–2000"] },
  { id: "2000s", label: "2000s", year: 2000, scanSeries: ["2000-digital-zoning"], districtSeries: "2000-digital-zoning", sources: ["Sept. 2000 zoning map book introduction"] },
  { id: "today", label: "Today", year: 2026, scanSeries: [], sources: ["SF Planning current zoning"] },
];
