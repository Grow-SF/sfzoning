// DRAFT COPY — NOT REVIEWED BY THE OWNER. Do not publish as final.
//
// Facts come from docs/research/zoning-history.md (and zoning-districts.csv). Every
// `sources` entry is an id in content/sources.ts, which mirrors docs/research/sources.md.
// Numbers appear only where the research cites them. Map descriptions ("hatching",
// "RH-1 labels") were read off the georeferenced scans at each step's camera; the 1921 and
// 1948 symbology is from the "Map Designations for Use of Property" legend on both atlases.
//
// Cameras were chosen by screenshotting candidate neighbourhoods at z14+ in every era.
// Steps that repeat a camera (Sunset 1921 → 1948 → 1980; Noe Valley 1960 → 1972 → 1978)
// do so on purpose, so the change between eras is the only thing that moves.
//
// Views: "allowed" (district colours over the paper basemap) where the colours alone tell
// the story (today, the 1978 legacy), "both" (colours over the scan) where the scan's own
// labels and hatching matter, and "scan" for height and downtown floor-area steps that the
// use colours can't show. Eras whose district tiles aren't published yet fall back to the
// scan with a "colours coming" note (StoryMap), so these views are safe to set ahead of data.
import type { Camera, Step } from "./schema";

const CITY: Camera = { center: [-122.443, 37.758], zoom: 12.3 };
const OUTER_SUNSET: Camera = { center: [-122.49, 37.749], zoom: 14.2 };
const MISSION: Camera = { center: [-122.418, 37.758], zoom: 15 };
const NOE_VALLEY: Camera = { center: [-122.433, 37.752], zoom: 14.5 };
const SOMA: Camera = { center: [-122.405, 37.775], zoom: 14.5 };
const WESTERN_ADDITION: Camera = { center: [-122.433, 37.781], zoom: 14.5 };
const DOWNTOWN: Camera = { center: [-122.401, 37.79], zoom: 14.8 };
const FINANCIAL_DISTRICT: Camera = { center: [-122.398, 37.793], zoom: 15.2 };
const MISSION_WIDE: Camera = { center: [-122.415, 37.762], zoom: 14.5 };

export const STORY: Step[] = [
  {
    id: "today-opening",
    era: "today",
    view: "allowed",
    camera: CITY,
    title: "A city zoned, block by block",
    body: [
      {
        kind: "p",
        // DRAFT: describes the "allowed" colours shown at this step (lib/palette.ts).
        text: "This is San Francisco today, every lot coloured by what its zoning allows. Orange-red means one home per lot. The colours run through amber and pale yellow, then pale mint, green and teal as more homes are allowed, with or without shops below, ending in bright green where the most homes are allowed. Light and mid blue are commercial, brown is industry and grey is parks and public land.",
      },
      {
        kind: "p",
        text: "In 1921, apartments were legal across large parts of the city. By the end of the 1970s, most of it was zoned for houses. The city’s own zoning maps, aligned to today’s streets, show how that happened.",
      },
    ],
    sources: ["sf-zoning-1921-10", "sfplanning-has-2020"],
  },
  {
    id: "1921-sunset",
    era: "1921",
    view: "both",
    camera: OUTER_SUNSET,
    title: "1921: the Sunset was zoned for apartments",
    body: [
      {
        kind: "p",
        text: "You’re looking at the Outer Sunset on the city’s first zone maps. The bright green blocks (plain white on the original map) are Second Residential, where apartment houses, flats and boarding houses were legal by right, with no cap on how many homes a lot could hold.",
      },
      {
        kind: "p",
        text: "The orange-red at the southern edge of the Sunset (diagonal hatching on the original) is First Residential: single-family houses. A two-family house was allowed only if 5,000 square feet of open lot remained. A standard San Francisco lot is 25 by 100 feet, or 2,500 square feet, so in practice that meant one house.",
      },
      { kind: "stat", text: "Second Residential: no limit on homes per lot" },
    ],
    sources: ["sf-zoning-1921-10", "use-of-property-1921-legend"],
  },
  {
    id: "1921-mission",
    era: "1921",
    view: "both",
    camera: MISSION,
    title: "Six districts, and homes allowed in five",
    body: [
      {
        kind: "p",
        text: "In the Mission, the blue streets are Commercial (solid black on the original map). Commercial did not mean “no homes”: the ordinance said nothing allowed in the residential districts “shall be excluded from the Commercial District.” Light Industrial, in brown, allowed flats and tenements too.",
      },
      {
        kind: "p",
        text: "Only Heavy Industrial, the same brown on this map, banned dwellings, keeping a watchman’s quarters. And the whole ordinance had no height limits, no minimum lot sizes, no yard rules and no density limits. It regulated what a building was for, not how big it was.",
      },
      { kind: "stat", text: "6 districts, 0 density limits" },
    ],
    sources: ["sf-zoning-1921-10", "use-of-property-1921-legend"],
  },
  {
    id: "1948-sunset",
    era: "1948",
    view: "both",
    camera: OUTER_SUNSET,
    title: "By 1948, the hatching had spread",
    body: [
      {
        kind: "p",
        text: "Same blocks, 1948. The legend hasn’t changed, but most of the Outer Sunset has turned orange-red: First Residential (hatched on the original), one house per lot in practice. North of Golden Gate Park, most of the Richmond stayed bright green: Second Residential.",
      },
      {
        kind: "p",
        text: "The fine print changed too. Section 99, approved in November 1946, was the city’s first rule on lot size and coverage.",
      },
      { kind: "quote", text: "Dwellings may cover no more than 65 per cent of the lot." },
      {
        kind: "p",
        text: "Resubdivided lots had to be at least 2,500 square feet and 25 feet wide, and new subdivisions needed 25-foot rear yards. In 1955, after complaints about new apartments without garages, every new home needed an off-street parking space.",
      },
    ],
    sources: ["rumsey-1948-atlas", "planning-code-1947", "san-francisco-planning-code-1957", "draftenvironment1319sanf"],
  },
  {
    id: "1960-density-caps",
    era: "1960",
    view: "both",
    camera: NOE_VALLEY,
    title: "1960: the first caps on homes per lot",
    body: [
      {
        kind: "p",
        text: "Now the map speaks in codes. Around Noe and Eureka Valleys it’s a patchwork of R-1 (orange-red), R-2 (amber) and R-3 (mid green). The modern City Planning Code took effect on May 2, 1960, and for the first time every residential district came with a number: how much lot area each home needed.",
      },
      { kind: "stat", text: "On a 25×100 lot: R-1, one home · R-2, two · R-3, about six · R-4, about twelve" },
      {
        kind: "p",
        text: "The code also set heights, yards and coverage. R-1 houses were held to 35 feet; R-2 and R-3 to 40.",
      },
    ],
    sources: ["planning-code-1960"],
  },
  {
    id: "1960-not-a-downzoning",
    era: "1960",
    view: "both",
    camera: SOMA,
    title: "Not mainly a downzoning",
    body: [
      {
        kind: "p",
        text: "A common story says 1960 is when the city downzoned its neighbourhoods. The record says otherwise. The city’s own later account says the code cut commercial zoning. It also added the new density caps and barred homes from the brown south of Market: M-1 and M-2. The 1921 code had allowed housing in Light Industrial.",
      },
      {
        kind: "p",
        text: "The Planning Department later wrote that the 1960 districts",
      },
      {
        kind: "quote",
        text: "allowed the existing mixed pattern of residential development to continue … new development tended to move toward the greatest density that the zoning allowed.",
      },
      { kind: "p", text: "The backlash to that came in the 1970s." },
    ],
    sources: ["planning-code-1960", "draftenvironment1319sanf"],
  },
  {
    id: "1972-height-limits",
    era: "1972",
    view: "scan",
    camera: NOE_VALLEY,
    title: "1972: forty feet, almost everywhere",
    body: [
      {
        kind: "p",
        text: "Back in Noe Valley, a new kind of map. The labels now read “40-X”: a 40-foot height limit, with no extra bulk rule. Height had become its own citywide layer.",
      },
      {
        kind: "p",
        text: "In 1964 the city had capped heights in a few northern neighbourhoods, at limits as low as 20 to 35 feet in Anza Vista and as high as 105 on Russian Hill. Ordinance 234-72, approved Aug. 18, 1972, went further. In the Planning Department’s words, it",
      },
      {
        kind: "quote",
        text: "put height limits on all property in San Francisco, with the majority of residential areas limited to 40 feet.",
      },
      {
        kind: "p",
        text: "Voters had turned down stricter caps for downtown: Prop T (80 feet) lost 63–37 in 1971, and Prop P (160 feet) won only 43% in 1972.",
      },
    ],
    sources: ["draftenvironment1319sanf", "gocodebook-sec-252", "1964-sf-zoning-partial", "map-legend-1975", "spur-high-rise-hegira"],
  },
  {
    id: "1975-downzonings",
    era: "1975",
    view: "both",
    camera: WESTERN_ADDITION,
    title: "1970–75: the downzonings",
    body: [
      {
        kind: "p",
        text: "The 1975 maps still use the 1960 codes, like the R-3s in mid green and the R-4s in teal across the Western Addition. The meanings didn’t change; the amount of each did. Between 1970 and 1975, large areas were reclassified from R-3, R-4 and R-5 to R-1 and R-2.",
      },
      { kind: "stat", text: "1970–75: R-5 land −57% · R-4 −7% · R-2 +22.5%" },
      {
        kind: "p",
        text: "R-5 was the densest residential district, about 20 homes on a standard lot. R-2 allowed two.",
      },
    ],
    sources: ["draftenvironment1319sanf", "map-legend-1975", "planning-code-1960"],
  },
  {
    id: "1978-rezoning",
    era: "1980",
    view: "both",
    camera: NOE_VALLEY,
    title: "1978: the city counted what it gave up",
    body: [
      {
        kind: "p",
        text: "Noe Valley once more, on the map amended to November 1978. The R codes are gone. In their place are RH-1, RH-2 and RH-3 (orange-red, amber and pale yellow), “House Character” districts allowing one, two or three homes per lot, and RM (mint, green and teal), “Mixed House & Apartment Character.”",
      },
      {
        kind: "p",
        text: "The Board of Supervisors voted for the residential rezoning on Sept. 18, 1978. The city’s own Final Environmental Impact Report had said it would “reduce the allowable density in many neighborhoods, so that,",
      },
      { kind: "quote", text: "approximately 180,000 estimated fewer housing units could legally be built.”" },
      {
        kind: "p",
        text: "It also warned that “prices and rents may be bid upward,” and that low- and moderate-income households could be displaced.",
      },
    ],
    sources: ["finalenvironment2719sanf", "map-legend-1980", "sbuss-when-did-things-go-wrong"],
  },
  {
    id: "1978-legacy",
    era: "1980",
    view: "allowed",
    camera: OUTER_SUNSET,
    title: "RH-1, as far as the eye can see",
    body: [
      {
        kind: "p",
        text: "The Outer Sunset, 1980: RH-1 on nearly every block, one home per lot. The 1978 rules went past the labels. RH-1 was capped at 35 feet and RH-2 at 40, front setbacks had to match the neighbours, and anything over 40 feet in a residential district needed the Planning Commission’s approval.",
      },
      {
        kind: "p",
        text: "The house districts stuck. In 2020 SF Planning counted RH-1, RH-2 and RH-3 at about 70% of the city’s residential land, and 7% of its recent housing.",
      },
      { kind: "stat", text: "RH districts: ~70% of residential land, 7% of recent homes" },
    ],
    sources: ["zoning-map-book-2000", "map-legend-1980", "sfplanning-has-2020"],
  },
  {
    id: "1985-downtown-plan",
    era: "1980-1990",
    view: "scan",
    camera: DOWNTOWN,
    title: "1985: the Downtown Plan",
    body: [
      {
        kind: "p",
        text: "Downtown, the 1980s maps read C-3-O, C-3-R, C-3-G and C-3-S. The Downtown Plan, passed 6–5 in September 1985, cut how much floor space each could hold. In C-3-O the basic limit fell from 14 times the lot area to 9; in C-3-R and C-3-G, from 10 to 6.",
      },
      {
        kind: "p",
        text: "Builders could go up to 18 by buying development rights from other lots. The plan also protected 266 landmark buildings and capped new office space at 950,000 square feet a year, for three years.",
      },
      { kind: "stat", text: "C-3-O floor-area ratio: 14 → 9" },
    ],
    sources: ["C124887752", "spur-prop-m"],
  },
  {
    id: "1986-prop-m",
    era: "1980-1990",
    view: "scan",
    camera: FINANCIAL_DISTRICT,
    title: "1986: Prop M makes the cap permanent",
    body: [
      {
        kind: "p",
        text: "A year later voters went further. Prop M passed 51.4% to 48.6%. It set a permanent annual cap of 475,000 square feet on new large office space.",
      },
      {
        kind: "p",
        text: "It also wrote eight Priority Policies into the Planning Code, favouring the preservation of neighbourhood character and existing housing.",
      },
    ],
    sources: ["spur-prop-m"],
  },
  {
    id: "2008-area-plans",
    era: "2000s",
    view: "both",
    camera: MISSION_WIDE,
    title: "2008: growth goes east",
    body: [
      {
        kind: "p",
        text: "The Mission in the 2000 map book: a dense patchwork of house, apartment, commercial and industrial districts, with homes over shops in the same greens as apartments. In 2008 two area plans rewrote much of this side of the city.",
      },
      {
        kind: "p",
        text: "Market & Octavia created districts with no density limit by lot area: the building’s size, not a unit count, sets how many homes fit. Eastern Neighborhoods rezoned about 2,200 acres of the Mission, SoMa, Potrero Hill and the Central Waterfront.",
      },
      {
        kind: "p",
        text: "Most of the housing built since 2000 went up in these eastern plan areas, not on the RH west side.",
      },
    ],
    sources: ["zoning-map-book-2000", "sfplanning-market-octavia", "sfplanning-eastern-neighborhoods", "sfplanning-has-2020"],
  },
  {
    id: "today-reform",
    era: "today",
    view: "allowed",
    camera: CITY,
    title: "Today: turning the dial back",
    body: [
      {
        kind: "p",
        text: "Since November 2022, an RH lot may hold four homes, or six on a corner, under conditions. In December 2023 a Housing Production ordinance removed many conditional-use hearings and notice requirements.",
      },
      {
        kind: "p",
        text: "Then the Family Zoning Plan. The Board passed it 7–4 on Dec. 2, 2025, and it took effect on Jan. 12, 2026. It upzones about 96,000 parcels, mostly on the west and north sides, for roughly 36,000 more homes of capacity, with taller buildings along corridors such as Van Ness, Market and Geary.",
      },
      { kind: "stat", text: "~96,000 parcels upzoned · ~36,000 homes of new capacity" },
      {
        kind: "p",
        text: "Press called it the first significant upzoning of those neighbourhoods since 1978. Whether it undoes 1978 is a question for the next set of maps.",
      },
    ],
    sources: ["ord-210-22-summary", "ord-248-23", "sfplanning-family-zoning", "kqed-family-zoning"],
  },
];
