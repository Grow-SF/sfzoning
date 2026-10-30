# San Francisco zoning history: what was allowed, when, and why it matters for housing

Research notes for sfzoning.com scrollytelling copy. Facts marked **[primary]** were read from ordinance or code text (the owner's scans or archive.org). Facts marked **[secondary]** come from histories or press. Claims marked *(unverified)* still need a primary check. The district-by-district data is in `zoning-districts.csv`, and every source is listed in `sources.md`.

---

## 1921: The first zoning ordinance regulates use, not size

- **Ordinance No. 5464 (New Series)** "regulating and establishing the location of trades, industries and buildings" passed **Oct. 3, 1921**. **[primary]** (`pipeline/data/raw/1929/…pdf` p.2 reprints it with that date. The owner's 1921 scan is the original pamphlet.)
- It divided the city into six classes: **First Residential, Second Residential, Commercial, Light Industrial, Heavy Industrial and Unrestricted**, shown on 14 sectional "use of property zone maps." **[primary]** (1921 Sec. 2; index page.)
- **First Residential** allowed single-family dwellings, schools, churches, community club houses, nurseries/farms/greenhouses and accessory uses, such as a doctor's home office or a garage for up to four cars. One detail matters for the story: **a two-family dwelling was also allowed "providing that at least five thousand (5,000) square feet of open area shall remain upon the lot."** On a standard 25×100 lot (2,500 sq ft) that is impossible, so in practice First Residential meant single-family only. **[primary]** (1921 Sec. 3.) *The lot-size arithmetic is our inference.*
- **Second Residential** added tenement/apartment houses, flats, boarding houses, hotels, hospitals and the like. **Apartments were legal by right, with no density cap.** **[primary]** (Sec. 4.)
- The system was **cumulative**. Commercial and Light Industrial listed only what was *excluded*, and expressly said "no uses permitted by Sections 3 and 4 hereof shall be excluded from the Commercial District." Housing was therefore allowed in Commercial and Light Industrial too. **Heavy Industrial was the only district that banned dwellings**, keeping just a watchman's quarters. **[primary]** (Secs. 5–7.)
- The ordinance contained **no height limits, lot-size minimums, yard or coverage rules, or density limits**. **[primary]** (Read the whole text: Secs. 1–19.) SF Planning's 2020 white paper says the same: no height limits, lot-area minimums, rear-yard, open-space, parking or sign rules. **[secondary]** Erica Fischer, quoted by Collectors Weekly, notes that a corner-store lot could host an apartment building or a 20-story tower. **[secondary]**
- Front set-back lines were adopted street by street under a separate ordinance (later Ord. 5636), and only in residential districts. **[primary]** (1929 reprint.) SF Planning dates a "uniform setback ordinance" to 1922. **[secondary]**
- **Why it matters:** 1921 zoning already decided *where* apartments could go. First Residential was effectively single-family, mostly in outlying areas (the bottom edge of the Sunset, Balboa Park, the Excelsior, per sbuss.substack.com; *confirm from the georeferenced 1921 layer*). Everywhere else, apartments were legal at any size. Context: the Aug. 24, 1921 Board meeting included calls to use residential zoning to keep Japanese residents out of the Fillmore. **[secondary: Collectors Weekly]**

## 1928–29: A Planning Commission with teeth; same six districts

- **Charter Article XVIII**, approved by voters Nov. 6, 1928 and ratified Jan. 17, 1929, created a five-member City Planning Commission. It gave the Commission control over zoning changes and set-back lines, with a two-thirds Board vote needed to overrule it. **[primary]** (1929 PDF pp.1–2.)
- The 1929 reprint of Ord. 5464 keeps the same six districts and First Residential's 5,000 sq ft rule for two-family homes. Small amendments: Commercial also excludes warehouses and stone works, and the First Residential garage rule is reworded. **[primary]** The **1935 zone maps** use the same legend. **[primary]**
- **Height limits in the 1920s:** the 1977 Draft EIR says "during the 1920's scattered height limits were established in the northern part of the City under the Building Code, not the City Planning Code." **[primary-ish: Planning Dept. EIR]** SF Planning's 2020 white paper says height limits were added to residential areas in 1927, and Collectors Weekly says 40-ft limits were set for the Marina, Pacific Heights and Presidio Heights in 1928. **[secondary]** *(Ordinance not located.)*

## 1946–1948: Lot-size and coverage rules arrive

- The 1947/48 "City Planning Code" (Part II, Ch. II of the Municipal Code) is still the 1921 six-district system, with use lists lightly edited. **[primary]** (`1948/originals/planning-code.jpg`.)
- **Sec. 99**, added by Ordinance No. 4162 and approved Nov. 26, 1946, was the first lot-size and lot-coverage rule. **[primary]**
  - New subdivision lots must match the neighborhood pattern, or be at least 33 ft wide.
  - Resubdivided lots must be at least 2,500 sq ft and 25 ft wide.
  - Dwellings may cover no more than **65%** of the lot.
  - Rear yards of **25 ft** are required in new subdivisions and 15 ft elsewhere.
- **1955:** after complaints about new apartments without parking, an amendment required **one off-street parking space per new dwelling unit**. This appears as Article 4.5 in the Dec. 1957 code. **[primary: 1977 DEIR; 1957 code on archive.org]**
- **On apartment bans:** sbuss's earlier analysis says that by 1948 apartments were banned on about one-third of residential land, the First Residential area. *Reproduce that figure from the georeferenced 1948 map rather than citing it.*
- A 1949 "Second Draft of New Comprehensive Zoning Plan" (archive.org) started the 12-year rewrite that became the 1960 code.

## 1954: Discretionary review

- In a 1954 dispute over a motel next to Shriners Hospital in the Sunset, the City Attorney opined that the city had "supreme control" over permits and that the Commission could use "sound discretion." This became the basis for **Discretionary Review**, which lets any project be pulled into a hearing even when it complies with the code. **[secondary: SF Planning HAS white paper, citing City Attorney Opinion No. 845, May 26, 1954]**

## 1960: The modern Planning Code and the first density limits

- The comprehensive revision was adopted by the Board and took effect **May 2, 1960**. **[primary: 1960 map cover; 1977 DEIR]**
- It created **R-1-D, R-1, R-2, R-3, R-4, R-5, C-1, C-2, C-3, C-M, M-1, M-2 and P**. For the first time it set **density limits as lot area per dwelling unit**, along with coverage, yards, height and FAR. **[primary: March 1, 1960 "Summary of Zoning Provisions," `1960/originals/1960.pdf` p.2]**

| District | Units allowed | Height | Notes |
|---|---|---|---|
| R-1-D | 1 per lot | 35 ft | 4,000 sq ft min lot; side yards |
| R-1 | 1 per lot (or 1 per 3,000 sq ft) | 35 ft | 60% coverage |
| R-2 | 2 per lot (or 1 per 1,500 sq ft) | 40 ft | |
| R-3 | 1 per 400 sq ft (~6 on 25×100) | 40 ft | multi-family up to 3 stories |
| R-4 | 1 per 200 sq ft (~12) | FAR 4.8 | |
| R-5 | 1 per 125 sq ft (~20) | FAR 10 | 1-room apt = ¾ unit |
| C-1 / C-2 | same as nearest R, never below R-3 | FAR 3.6 | |
| C-3 | same rule | FAR 20 | |
| M-1 / M-2 | **no dwellings** | FAR 5 | |

- **Why it matters:**
  1. **Housing was banned from industrial zones.** The 1921 code had allowed it in Light Industrial; C-M also barred one- and two-family homes. **[primary]**
  2. Single-family and two-family zones got **numeric unit caps**.
  3. The Planning Department's 1977 Draft EIR says the 1960 districts still "allowed the existing mixed pattern of residential development to continue" and that "new development tended to move toward the greatest density that the zoning allowed." That backlash drove the 1970s downzonings. **[primary: DEIR p.4]**
- "The 1960 downzoning": the 1960 code was *not* mainly a residential downzoning. Its big shifts were reducing commercial zoning, banning housing in M districts and adding density caps. Large reclassifications from R-3/R-4/R-5 to R-1/R-2 came **1970–75**: R-5 acreage fell 57%, R-4 fell 7% and R-2 rose 22.5%. **[primary: DEIR Table 3-6]**

## 1964–1968: Height districts begin; downtown split

- **Ord. 35-64**, approved Feb. 21, 1964 and effective March 22, 1964, created Article 2.5 **Special Height Districts**, the first zoning height caps on residential neighborhoods. **[primary: 1964 maps 1SH/2SH; 1964 code excerpts on archive.org]**
  - 40 ft: Marina–Pacific Heights, Northern Waterfront–North Beach, Telegraph Hill.
  - 65 ft: North Beach, Union Street.
  - 84 ft: Northeastern Embarcadero.
  - 105 ft: Pacific Heights, both slopes of Russian Hill.
  - 20–35 ft: Anza Vista.
- An **R-3.5** district was added in 1964. **[primary: 1965 map legend; DEIR]** *Its standards were not found.*
- By the **Oct. 1, 1968** map, C-3 was split into **C-3-O, C-3-R, C-3-G, C-3-S**, and **R-4-C / R-5-C** mixed residential-commercial districts appeared. **[primary: map legends]** The 1966 Downtown Zoning Study recommended basic FARs of 14:1 (O), 10:1 (R, G) and 7:1 (S), down from 20:1. **[primary: study]** *Exact adopted 1968 values are unverified.*

## 1971–72: The Urban Design Plan and citywide height limits

- The **Urban Design Plan** was proposed in May 1971 after a study begun in 1968. The 1971 draft named 11 areas for taller residential buildings, and opponents attacked them as "Manhattanization." They were dropped after hearings. **[secondary: Collectors Weekly; SF Chronicle 2021; primary: May 25, 1972 staff report]**
- Voters rejected height-cap initiatives: **Prop T (1971)**, 80 ft downtown, lost 63–37; **Prop P (1972)**, 160 ft, lost with 43%. **[secondary: SPUR]**
- **Ord. 234-72**, approved **Aug. 18, 1972**, established **height and bulk districts** citywide (Sec. 252; maps under Board File 362-72-2). **[secondary: Planning Code history note via gocodebook]** The Planning Department's 1977 Draft EIR says it "put height limits on all property in San Francisco, with the majority of residential areas limited to 40 feet." **[primary]**
- **How to read a height/bulk label:** the 1975 map legend reads "Numbers are height limits in feet. Letter symbols refer to bulk limits in City Planning Code Sec. 270."
  - **40-X** = 40 ft, no bulk limit.
  - **65-A** = 65 ft; above 40 ft, a floor plate may be at most 110 ft long with a 125 ft diagonal.
  - **240-S** = 240 ft, downtown "S" bulk rules.
  - **OS** = open space.
  - **[primary: 1975 sheet 1H; 2000 map book bulk table]**
- **Why it matters:** height became a separate, citywide map layer. Combined with use-district density, it froze most neighborhoods at about 4 stories.

## 1978: The residential rezoning (RH/RM/RC)

- The Planning Department began a residential zoning study in **1973**. The Commission adopted a resolution of intention Dec. 8, 1977, and the Draft EIR came out Dec. 13, 1977. **[primary: DEIR]**
- The Board voted **Sept. 18, 1978** (sbuss.substack.com lists 8 yes and 3 no names from the Board Journal, July–Dec. 1978, archive.org `journaljulydecofproceed73sanfrich` p.1161; *tally not re-checked*). It was enacted as **Ord. 443-78, approved Oct. 6, 1978**. **[secondary: Planning Code history notes for Secs. 209.1–209.3]**
- It replaced R-1-D…R-5 with:
  - **House districts:** RH-1(D), RH-1, RH-1(S), RH-2, RH-3, allowing 1, 1, 1+small second unit, 2 and 3 units per lot.
  - **Mixed districts:** RM-1 to RM-4, at 1 unit per 800 / 600 / 400 / 200 sq ft.
  - **Residential-commercial districts:** RC-1 to RC-4.
  - The 1980 map legend calls them "House Character," "Mixed House & Apartment Character" and "Residential-Commercial Combined" districts. **[primary]**
- New rules **[primary: 2000 map book; DEIR]**:
  - Buildings over **40 ft** in any R district need Commission approval.
  - Front setbacks must match neighbors.
  - RH-2/3 and RM-1/2 need 45% rear yards.
  - RH-1 is capped at 35 ft and RH-2 at 40 ft (Sec. 261).
- The Final EIR (June 27, 1978) says the amendments "would reduce the allowable density in many neighborhoods, so that, approximately **180,000 estimated fewer housing units** could legally be built." It also warned that "prices and rents may be bid upward" and low- and moderate-income households displaced. **[primary: FEIR p.122]** The Draft EIR said the new controls "would likely intensify the pressures for high rents and housing prices." **[primary: DEIR p.59–60]**
- **Lasting effect:** in SF Planning's 2020 white paper, RH-1/2/3 cover **about 70% of residential land but produced 7% of recent housing**, with RH-1 alone at 41% of residential land. Many existing multi-unit buildings became nonconforming. **[secondary: SF Planning]**

## 1984–1986: Downtown fights (Prop K, Downtown Plan, Prop M)

- **Prop K (June 1984), the "Sunlight Ordinance"** (Planning Code Sec. 295): new buildings over 40 ft that would shade Recreation & Park property need joint Planning and Rec-Park approval. **[secondary]**
- **Downtown Plan, Ord. 414-85**, passed Sept. 10, 1985 on a 6–5 vote and effective Oct. 17, 1985. **[primary: ordinance on archive.org]**
  - Cut basic FARs to **C-3-O 9:1, C-3-R 6:1, C-3-G 6:1, C-3-O(SD) 6:1** (C-3-S 5:1 per the 2000 book), down from 14/10/10/7.
  - Allowed FAR up to 18:1 via transferable development rights.
  - Added an annual office cap of 950,000 sq ft with a 3-year sunset.
  - Protected 266 landmark buildings. **[secondary: SPUR]**
- **Prop M (Nov. 1986)** passed 51.4–48.6. It set a permanent annual cap of **475,000 sq ft** on new large office space and added the eight **Priority Policies** (Sec. 101.1), which favor preserving neighborhood character and existing housing. **[secondary: SPUR]**
- **Why it matters for housing:** Prop M's priority policies became a standard hook for opposing change in residential neighborhoods. The office cap pushed growth debates toward jobs-housing balance.

## 1987–1990: Neighborhood Commercial and South of Market

- **Neighborhood Commercial (Article 7)** controls took effect **April 1987**, replacing most C-1/C-2 corridors with NC-1, NC-2, NC-3, NC-S and named districts. Housing was allowed above shops at 1 unit per 800 (NC-1/2) or 600 (NC-3) sq ft. **[secondary: SF Planning; primary: 2000 map book table]**
- **South of Market use districts** (RED, SPD, RSD, SLR, SLI, SSO) appear in the 2000 map book. *Adoption around 1990 is inferred.*

## 2008: Area plans and form-based zoning

- **Market & Octavia Plan** was approved by the Board April 2008 and took effect May 30, 2008. It created **RTO** and **NCT** districts, with **no density limit by lot area**: unit counts are set by the building envelope. **[secondary: SF Planning archive]**
- **Eastern Neighborhoods** EIR was certified Aug. 7, 2008, with Board approval in Dec. 2008. It rezoned about 2,200 acres of the Mission, SoMa, Potrero and Central Waterfront into about 13 new districts (UMU, MUR, MUG, MUO, PDR-1/2, etc.). **[secondary: SF Planning]**
- **Why it matters:** most post-2000 housing was built in these eastern plan areas, not in the RH west side. **[secondary: SF Planning white paper]**

## 2016–2025: Chipping away, then the Family Zoning Plan

- **Fourplexes:** Ord. 210-22, effective Nov. 27, 2022, allows **4 units per lot, or 6 on corner lots**, in all RH districts under conditions. Added units are rent-controlled. **[primary: SF Planning code summary]**
- **Ord. 248-23 ("Housing Production," Mayor Breed, Dec. 2023)** removed many conditional-use and notice requirements and expanded the density exceptions. **[primary: ordinance title]**
- **Family Zoning Plan:** the Board passed it 7–4 on Dec. 2, 2025, and Mayor Lurie signed it Dec. 12, 2025. It took effect **Jan. 12, 2026**. **[secondary: SF Planning, KQED]**
  - Upzones about 96,000 parcels, mostly on the west and north sides, for about 36,000 units of new capacity.
  - Adds 2–4 extra stories along corridors and allows high-rises on select major streets (Van Ness, Market, Geary).
  - Creates a "Local Program" bonus height map and a new **RTO-C** district.
  - Press called it the first significant upzoning of these neighborhoods since 1978.
  - *Exact final heights and density treatment per parcel should be read from the adopted maps.*

---

### Key numbers for copy (all sourced above)
- 1921: 6 districts; apartments legal by right everywhere except First Residential and Heavy Industrial.
- 1946: first lot-coverage limit (65%).
- 1960: first density limits; housing banned in M zones.
- 1964: first zoning height caps (40–105 ft) in the northern neighborhoods.
- 1972: citywide height limits, mostly 40 ft in residential areas.
- 1978: RH/RM; the city's own EIR estimated 180,000 fewer legally buildable homes.
- 1985–86: downtown FARs cut (C-3-O 14 → 9) and an office cap.
- 2022–2026: fourplexes, then the Family Zoning Plan.
