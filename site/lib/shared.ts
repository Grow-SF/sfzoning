import { isCategory, type Category } from "./palette";
import { tileUrl, tilesHttpUrl } from "./tiles";

// The explorer's one set of shapes for every era (pipeline `sfzmaps districts --shared`). Each lot
// and district line is stored once with a number, and a small table says what that number looks
// like in each year, so switching year changes no geometry and fetches nothing.
// out/shared/districts.json names the current files, so publishing needs no site rebuild.

export const SHARED_SOURCE = "dist-shared";
const MANIFEST_VERSION = 2;
/** Set after a lost WebGL context, so a reload on this device takes the lighter per-era path. */
const LOST_CONTEXT_FLAG = "sfz-no-shared-tiles";
export const NO_VALUE = 255;

export type SharedManifest = {
  /** MapLibre source URL of the shapes. */
  url: string;
  table: string;
  cards: string;
  categories: Category[];
  lots: number;
  lineSets: number;
  /** Series, in the table's order. */
  eras: string[];
};

/** The shapes and the table, ready to draw. */
export type SharedDistricts = SharedManifest & {
  /** Each era's category number per lot, then each era's 0/1 per line set. */
  values: Uint8Array;
};

/** The pointer file's contents, or null for anything this version of the site can't use. */
export function parseManifest(json: unknown): SharedManifest | null {
  const m = json as Record<string, unknown> | null;
  if (m?.version !== MANIFEST_VERSION) return null;
  const { archive, table, cards, categories, lots, lineSets, eras } = m;
  if (typeof archive !== "string" || typeof table !== "string" || typeof cards !== "string") return null;
  if (!Array.isArray(categories) || !categories.every(isCategory)) return null;
  if (!Array.isArray(eras) || !eras.every((e) => typeof e === "string")) return null;
  if (typeof lots !== "number" || typeof lineSets !== "number") return null;
  return { url: tileUrl(archive), table, cards, categories, lots, lineSets, eras };
}

function storageFlag(write: boolean): boolean {
  // Storage can throw in a private window; losing the flag only loses the fallback.
  try {
    if (write) window.sessionStorage.setItem(LOST_CONTEXT_FLAG, "1");
    return window.sessionStorage.getItem(LOST_CONTEXT_FLAG) === "1";
  } catch {
    return false;
  }
}

export function noteLostContext() {
  storageFlag(true);
}

/** Bytes of a .gz file. A host may already have unpacked it in transit, so check rather than assume. */
export async function gunzipped(bytes: ArrayBuffer): Promise<Uint8Array> {
  const head = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength));
  if (head[0] !== 0x1f || head[1] !== 0x8b) return new Uint8Array(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function fetchGz(path: string): Promise<Uint8Array> {
  const r = await fetch(tilesHttpUrl(path));
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return gunzipped(await r.arrayBuffer());
}

/** The shared districts if all of them are really there and haven't cost this tab its GPU; else null. */
export async function loadSharedDistricts(): Promise<SharedDistricts | null> {
  if (storageFlag(false)) return null;
  try {
    const r = await fetch(tilesHttpUrl("shared/districts.json"));
    if (!r.ok) return null;
    const manifest = parseManifest(await r.json());
    if (!manifest) return null;
    const probe = await fetch(manifest.url.replace(/^pmtiles:\/\//, ""), { headers: { Range: "bytes=0-6" } });
    if (!probe.ok) return null;
    const values = await fetchGz(manifest.table);
    if (values.length !== manifest.eras.length * (manifest.lots + manifest.lineSets)) return null;
    return { ...manifest, values };
  } catch {
    return null;
  }
}

/** A lot number's category in an era, or null if that era's map doesn't cover it. */
export function lotCategory(shared: SharedDistricts, eraIndex: number, lot: number): Category | null {
  const n = shared.values[eraIndex * shared.lots + lot];
  return n === NO_VALUE || n === undefined ? null : (shared.categories[n] ?? null);
}

type EraCards = { code: (number | null)[]; height: (number | null)[]; confidence: (number | null)[] };
export type LotCards = { blklot: string[]; strings: string[]; eras: Record<string, EraCards> };

export type LotCard = { blklot: string; code?: string; height_code?: string; confidence?: number };

/** What a lot's card shows for one series. */
export function lotCard(cards: LotCards, series: string, lot: number): LotCard {
  const era = cards.eras[series];
  const text = (n: number | null | undefined) => (n == null ? undefined : cards.strings[n]);
  return { blklot: cards.blklot[lot], code: text(era?.code[lot]), height_code: text(era?.height[lot]), confidence: era?.confidence[lot] ?? undefined };
}

const cardsCache = new Map<string, Promise<LotCards>>();
/** Lot cards are only needed once someone clicks a lot, so they load then. A failure can be retried. */
export function loadLotCards(path: string): Promise<LotCards> {
  let p = cardsCache.get(path);
  if (!p) {
    p = fetchGz(path).then((b) => JSON.parse(new TextDecoder().decode(b)) as LotCards);
    p.catch(() => cardsCache.delete(path));
    cardsCache.set(path, p);
  }
  return p;
}
