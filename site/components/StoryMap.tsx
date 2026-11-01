"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type * as Ml from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";
import { ERAS } from "@/lib/eras";
import { BASEMAP, scanSource, tileUrl } from "@/lib/tiles";
import { INK_ZOOM, MAP_COLORS, inkByZoom } from "@/lib/theme";
import { ALLOWED_FILL_OPACITY, BOTH_SCAN_WEIGHT, districtFillOpacity } from "@/lib/palette";
import { loadLotCards, loadSharedDistricts, lotCard, lotCategory, SHARED_SOURCE, type SharedDistricts } from "@/lib/shared";
import { heldScans, type Hold } from "@/lib/scanHold";
import { addLookupLayer, LOTS_LAYER } from "@/lib/lookupLayer";
import { useDistrictAvailability, type Availability } from "@/lib/districts";
import Legend from "./Legend";
import { RUMSEY_CREDIT } from "./Credits";

export type MapState = {
  era: string;
  view: "scan" | "allowed" | "both";
  /** Omit to leave the camera where it is (the explorer, where the reader pans). */
  camera?: { center: [number, number]; zoom: number; bearing?: number };
  highlight?: Feature<Geometry> | FeatureCollection<Geometry>;
  /** Warm this era's tiles (the next step's) by drawing them at near-zero opacity. */
  prefetch?: { era: string; view: "scan" | "allowed" | "both" };
};

export type StoryMapHandle = { apply: (state: MapState) => void };

/** A clicked parcel: its tile properties plus the era it was clicked in. */
export type ParcelHit = {
  era: string;
  blklot?: string;
  code?: string;
  height_code?: string;
  category?: string;
  confidence?: string | number;
};

type Props = {
  initial: MapState;
  /** Gestures are off during the story and on in the explorer. */
  interactive?: boolean;
  /** Explorer only: a parcel was clicked (null: clicked off any parcel). */
  onParcelClick?: (hit: ParcelHit | null) => void;
  /** Outline this parcel (by blklot) in the current era. */
  selectedBlklot?: string | null;
  className?: string;
};

type Expr = Ml.ExpressionSpecification;
type Opacity = number | Expr;

/** Test builds only (NEXT_PUBLIC_E2E=1, set by playwright.config): window.__map(s), data-idle. */
const E2E = process.env.NEXT_PUBLIC_E2E === "1";
const FADE_MS = 700;
/** The story's step: the camera move and the colour fades take this long together. */
const PAN_MS = 1400;
const HOLD_CHECK_MS = 200;
/** Story only: a scan tile that arrives late fades in rather than snapping over the old scan. */
const TILE_FADE_MS = 300;
/** Held alongside scans: the old colours stay up until the new scan has arrived. */
const COLOURS = "colours";
/** Below this, tiles still load but nothing visible is drawn. */
const WARM = 0.001;
/** In "both" at city zoom the colours carry the view; the scan fades to a trace of itself. */
const BOTH_CITY_INK = 0.3;
const scanLayer = (series: string) => `scan-${series}`;
const allSeries = () => Array.from(new Set(ERAS.flatMap((e) => e.scanSeries)));

/** Resolves once the page has loaded and the main thread has a moment to spare. */
function afterFirstPaint(): Promise<void> {
  return new Promise((resolve) => {
    const idle = () =>
      "requestIdleCallback" in window ? window.requestIdleCallback(() => resolve(), { timeout: 1500 }) : setTimeout(resolve, 200);
    if (document.readyState === "complete") idle();
    else window.addEventListener("load", idle, { once: true });
  });
}

let ml: typeof Ml | null = null;
/**
 * maplibre-gl (290 kB gzipped) is fetched and started only after the page has loaded and
 * gone idle, so the hero's headline paints and hydrates without waiting on WebGL work.
 */
async function loadMaplibre(): Promise<typeof Ml> {
  if (ml) return ml;
  await afterFirstPaint();
  const [lib, { Protocol }] = await Promise.all([import("maplibre-gl"), import("pmtiles")]);
  lib.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
  lib.addProtocol("pmtiles", new Protocol().tile);
  ml = lib;
  return lib;
}

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Story camera padding, so each step's subject sits in the part of the map the reader can see:
 * below the era ruler, and on phones above the step card (≤ 55% of the screen, plus its gap).
 * On desktop the map has its own column beside the text, so only the ruler needs room.
 */
function storyPadding(map: MlMap): Ml.PaddingOptions {
  const h = map.getContainer().clientHeight;
  const desktop = window.matchMedia("(min-width: 900px)").matches;
  const top = 64;
  return desktop ? { top, bottom: 0, left: 0, right: 0 } : { top, bottom: Math.round(h * 0.55) + 16, left: 0, right: 0 };
}

function scanOpacity(s: MapState, series: string, avail: Availability): Opacity {
  const era = ERAS.find((e) => e.id === s.era);
  if (!era || !era.scanSeries.includes(series)) {
    const next = s.prefetch && ERAS.find((e) => e.id === s.prefetch!.era);
    return next && next.scanSeries.includes(series) && !(s.prefetch!.view === "allowed" && avail[next.id]) ? WARM : 0;
  }
  // "allowed" hides the scan only once the colours exist; until then the scan stands in.
  if (s.view === "allowed" && avail[era.id]) return 0;
  // In "both" the scan's solid black (1948 Commercial, say) would drown the colours; ease the
  // ink so labels and hatching show through. Pulled out to the city, the scan is a grey veil,
  // so it fades almost away and the colours carry the view.
  if (s.view === "both" && avail[era.id]) return inkByZoom(BOTH_SCAN_WEIGHT, BOTH_CITY_INK) as Expr;
  // "The map as drawn": the paper itself, at full strength at every zoom.
  return 1;
}

/** The scan series that `s` draws at all (current era or prefetch warm-up). */
export function scansShown(s: MapState, avail: Availability): Set<string> {
  return new Set(allSeries().filter((series) => scanOpacity(s, series, avail) !== 0));
}

/** The scans drawn for real: shown, but not merely warmed up. */
function scansWanted(s: MapState, avail: Availability): string[] {
  return allSeries().filter((series) => {
    const o = scanOpacity(s, series, avail);
    return o !== 0 && o !== WARM;
  });
}

/** Add a scan's source and layer, under the district layers, if it isn't there yet. */
function addScan(map: MlMap, series: string, fadeTiles: boolean) {
  const id = scanLayer(series);
  if (map.getLayer(id)) return;
  map.addSource(id, { type: "raster", url: scanSource(series), tileSize: 256 });
  const before = map.getStyle().layers.find((l) => l.id.startsWith("dist-") || l.id === "selected-line")?.id;
  map.addLayer(
    { id, type: "raster", source: id, layout: { visibility: "none" }, paint: { "raster-opacity": 0, "raster-fade-duration": fadeTiles ? TILE_FADE_MS : 0 } },
    before,
  );
}

const LINE_WIDTH = { lo: [11, 0.4], hi: [16, 1.4] } as const;
const lineOpacity = (view: MapState["view"]) => (view === "both" ? 0.9 : 0.6);

const between = (x: number, [x0, y0]: readonly [number, number], [x1, y1]: readonly [number, number]) =>
  y0 + (y1 - y0) * Math.min(1, Math.max(0, (x - x0) / (x1 - x0)));

/** Fill opacity per view at a zoom. In "both", the scan shows through more closely up. Same object each time, so an unchanged look stays unchanged. */
const FILL_AT_ZOOM: Record<MapState["view"], (zoom: number) => number> = {
  scan: () => 0,
  allowed: () => ALLOWED_FILL_OPACITY,
  both: (zoom) => between(zoom, [INK_ZOOM.lo + 0.5, ALLOWED_FILL_OPACITY], [INK_ZOOM.hi - 0.5, districtFillOpacity("both")]),
};

const StoryMap = forwardRef<StoryMapHandle, Props>(function StoryMap(
  { initial, interactive = false, onParcelClick, selectedBlklot = null, className },
  ref,
) {
  const container = useRef<HTMLDivElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const pending = useRef<MapState | null>(initial);
  const readyRef = useRef(false);
  const stateRef = useRef<MapState>(initial);
  const availRef = useRef<Availability>({});
  const hideTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  /** The one set of shapes for every era, and the table saying how each looks. */
  const sharedRef = useRef<SharedDistricts | null>(null);
  const lookupRef = useRef<ReturnType<typeof addLookupLayer> | null>(null);
  /** The lot number behind selectedBlklot, in the shared shapes. */
  const selectedLot = useRef<number | null>(null);
  const clicks = useRef(0);
  /** Story only: scans kept visible under a new one until it has arrived. */
  const holdRef = useRef<Hold | null>(null);
  const holdTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastWanted = useRef<string[]>([]);
  /** The era whose colours are on screen (null: none). */
  const lastEra = useRef<number | null>(null);
  const paintedEra = useRef(initial.era);
  const clickRef = useRef(onParcelClick);
  /** Sources that failed to load ("blocks", "scan-<series>", or "maplibre" itself). */
  const [failedSources, setFailedSources] = useState<ReadonlySet<string>>(() => new Set());
  const [state, setState] = useState<MapState>(initial);
  const avail = useDistrictAvailability();

  useEffect(() => {
    clickRef.current = onParcelClick;
  }, [onParcelClick]);

  /** An era's position in the shared table; undefined if it has no districts yet. */
  const sharedIndex = useCallback((eraId: string) => {
    const series = ERAS.find((e) => e.id === eraId)?.districtSeries;
    const i = series && sharedRef.current ? sharedRef.current.eras.indexOf(series) : -1;
    return i >= 0 ? i : undefined;
  }, []);

  const stopHold = useCallback(() => {
    if (holdTimer.current) clearInterval(holdTimer.current);
    holdTimer.current = null;
  }, []);

  /** Paint every layer for `s` without moving the camera. */
  const paint = useCallback((s: MapState) => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    // In the explorer a new year replaces the last at once, so the eye can compare the two. A fade
    // shows a blend of both that neither map ever drew. The story's fades are part of its pacing.
    const newYear = interactive && paintedEra.current !== s.era;
    paintedEra.current = s.era;
    const instant = reducedMotion() || newYear;
    const fade = { duration: instant ? 0 : interactive ? FADE_MS : PAN_MS, delay: 0 };
    // Only the current era's scan and the prefetch era's are visible (MapLibre loads tiles for
    // every visible layer, even at zero opacity). A scan's source is added the first time it is
    // needed, so eras the reader never reaches cost no requests at all.
    const shownIndex = s.view === "scan" ? undefined : sharedIndex(s.era);
    const shown = scansShown(s, availRef.current);
    const wanted = scansWanted(s, availRef.current);
    if (wanted.join() !== lastWanted.current.join()) {
      // The story keeps the scans it is leaving under the new one until that has arrived.
      const gone = lastWanted.current.filter((x) => !wanted.includes(x));
      // Going from colours to a scan alone, the colours stay until the scan is there: a hold protects
      // the picture, not just the scans.
      const keepColours = shownIndex === undefined && lastEra.current !== null;
      const held = Array.from(new Set([...(holdRef.current?.held ?? []), ...gone, ...(keepColours ? [COLOURS] : [])])).filter(
        (x) => !wanted.includes(x) && (x !== COLOURS || shownIndex === undefined),
      );
      lastWanted.current = wanted;
      stopHold();
      if (!interactive && !instant && wanted.length && held.length) {
        holdRef.current = { held, since: Date.now() };
        holdTimer.current = setInterval(() => {
          const m = mapRef.current;
          const since = holdRef.current?.since ?? 0;
          const next = m
            ? heldScans(
                holdRef.current,
                lastWanted.current,
                (series) => m.isSourceLoaded(scanLayer(series)),
                Date.now() - since >= PAN_MS,
                Date.now(),
              )
            : null;
          holdRef.current = next;
          if (!next) {
            stopHold();
            // Released scans fade and hide as any scan leaving the view does.
            paint(stateRef.current);
          }
        }, HOLD_CHECK_MS);
        // New scans sit above the held ones, still under the districts.
        const before = map.getStyle().layers.find((l) => l.id.startsWith("dist-") || l.id === "selected-line")?.id;
        for (const series of wanted) {
          addScan(map, series, !interactive);
          map.moveLayer(scanLayer(series), before);
        }
      } else holdRef.current = null;
    }
    const holding = (series: string) => holdRef.current?.held.includes(series) ?? false;
    for (const series of allSeries()) {
      const id = scanLayer(series);
      const key = `scan:${series}`;
      const timer = hideTimers.current.get(key);
      if (timer) clearTimeout(timer);
      hideTimers.current.delete(key);
      if (shown.has(series)) {
        addScan(map, series, !interactive);
        map.setLayoutProperty(id, "visibility", "visible");
      }
      if (!map.getLayer(id)) continue;
      if (holding(series)) continue; // keeps its opacity until the hold releases it
      map.setPaintProperty(id, "raster-opacity-transition", fade);
      map.setPaintProperty(id, "raster-opacity", scanOpacity(s, series, availRef.current));
      if (!shown.has(series) && map.getLayoutProperty(id, "visibility") !== "none") {
        // Hide once faded out, unless a fast scroll has made this scan wanted again.
        const hide = () => {
          const m = mapRef.current;
          if (!m || !m.getLayer(id) || scansShown(stateRef.current, availRef.current).has(series) || holding(series)) return;
          m.setLayoutProperty(id, "visibility", "none");
        };
        if (instant) hide();
        else hideTimers.current.set(key, setTimeout(hide, fade.duration + 50));
      }
    }
    if (holding(COLOURS)) return;
    lastEra.current = shownIndex ?? null;
    lookupRef.current?.show({ era: shownIndex ?? null, fillOpacity: FILL_AT_ZOOM[s.view], lineOpacity: lineOpacity(s.view) }, fade.duration);
  }, [interactive, sharedIndex, stopHold]);

  const applyNow = useCallback(
    (s: MapState) => {
      stateRef.current = s;
      setState(s);
      const map = mapRef.current;
      if (!map || !readyRef.current) {
        pending.current = s;
        return;
      }
      paint(s);
      const src = map.getSource("highlight") as Ml.GeoJSONSource | undefined;
      src?.setData(s.highlight ?? { type: "FeatureCollection", features: [] });
      if (s.camera) {
        const cam = {
          center: s.camera.center,
          zoom: s.camera.zoom,
          bearing: s.camera.bearing ?? 0,
          ...(interactive ? {} : { padding: storyPadding(map) }),
        };
        if (reducedMotion()) map.jumpTo(cam);
        else map.easeTo({ ...cam, duration: PAN_MS });
      }
    },
    [paint, interactive],
  );

  useImperativeHandle(ref, () => ({ apply: applyNow }), [applyNow]);

  /** The topmost lot this era colours at a point (the lowest number is the smallest lot). */
  const lotAt = useCallback((map: MlMap, point: Ml.PointLike, eraIndex: number) => {
    const shared = sharedRef.current!;
    const lots = map
      .queryRenderedFeatures(point, { layers: [LOTS_LAYER] })
      .map((f) => Number(f.properties?.i))
      .filter((lot) => lotCategory(shared, eraIndex, lot) !== null);
    return lots.length ? Math.min(...lots) : null;
  }, []);

  /** Before any tile loads: adding a layer later would rebuild every tile. */
  const addSharedDistricts = useCallback(() => {
    const map = mapRef.current;
    const shared = sharedRef.current;
    if (!map || !shared) return;
    map.addSource(SHARED_SOURCE, { type: "vector", url: shared.url });
    lookupRef.current = addLookupLayer(map, shared, SHARED_SOURCE, "selected-line", (zoom) => between(zoom, LINE_WIDTH.lo, LINE_WIDTH.hi), MAP_COLORS.districtEdge);
    map.on("mousemove", (e) => {
      const index = sharedIndex(stateRef.current.era);
      if (index === undefined || !clickRef.current) return;
      const hit = stateRef.current.view !== "scan" && lotAt(map, e.point, index) !== null;
      map.getCanvas().style.cursor = hit ? "pointer" : "";
      // About to click, probably: have the card's data ready.
      if (hit) loadLotCards(shared.cards).catch(() => {});
    });
  }, [sharedIndex, lotAt]);

  // The selected lot is outlined from its own GeoJSON source. (A filtered line layer on the
  // district source made MapLibre drop that lot's fill when the filter changed.)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const src = map.getSource("selected") as Ml.GeoJSONSource | undefined;
    const features =
      selectedBlklot && map.getSource(SHARED_SOURCE)
        ? map.querySourceFeatures(SHARED_SOURCE, { sourceLayer: "parcels", filter: ["==", ["get", "i"], selectedLot.current ?? -1] })
        : [];
    src?.setData({
      type: "FeatureCollection",
      features: features.map((f) => ({ type: "Feature", properties: {}, geometry: f.geometry })),
    });
  }, [selectedBlklot]);

  useEffect(() => {
    availRef.current = avail;
    paint(stateRef.current);
  }, [avail, paint]);

  useEffect(() => {
    if (!container.current) return;
    let cancelled = false;
    let cleanup = () => {};
    const timers = hideTimers.current;
    const setIdle = (v: boolean) => {
      // Test and screenshot hook: true once the map has drawn everything it is going to.
      if (E2E && wrap.current) wrap.current.dataset.idle = String(v);
    };

    Promise.all([loadMaplibre(), loadSharedDistricts()]).then(([maplibregl, shared]) => {
      if (cancelled || !container.current) return;
      sharedRef.current = shared;
      const cam = initial.camera ?? { center: [-122.443, 37.758] as [number, number], zoom: 12.3 };
      const map = new maplibregl.Map({
        container: container.current,
        style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": MAP_COLORS.water } }] },
        center: cam.center,
        zoom: cam.zoom,
        bearing: cam.bearing ?? 0,
        interactive,
        // On touch screens the explorer map takes two fingers to pan, so one finger still
        // scrolls the page past it instead of being trapped.
        cooperativeGestures: interactive && window.matchMedia("(pointer: coarse)").matches,
        attributionControl: false,
        maxZoom: 18,
        fadeDuration: 0,
      });
      if (!interactive) map.jumpTo({ ...cam, padding: storyPadding(map) });
      mapRef.current = map;
      const w = window as unknown as { __map?: MlMap; __maps?: MlMap[] };
      if (E2E) {
        if (!interactive) w.__map = map;
        (w.__maps ??= []).push(map);
      }

      map.on("idle", () => setIdle(true));
      map.on("movestart", () => setIdle(false));
      map.on("dataloading", () => setIdle(false));
      // Phone rotation or a resized window changes how much map the card leaves visible.
      map.on("resize", () => {
        if (!interactive) map.setPadding(storyPadding(map));
      });

      map.on("error", (e) => {
        const msg = String((e as unknown as { error?: Error }).error?.message ?? "");
        const sourceId = String((e as unknown as { sourceId?: string }).sourceId ?? "");
        // A district error never blanks the map, so it never raises the notice.
        if (sourceId.startsWith("dist-")) {
          console.warn("district tiles:", sourceId, msg);
          return;
        }
        // Only the basemap and scans matter to the notice; which scans matter (the current
        // era's) is decided at render. Anything else is logged, not shown.
        if (sourceId === "blocks" || sourceId.startsWith("scan-")) {
          setFailedSources((prev) => (prev.has(sourceId) ? prev : new Set(prev).add(sourceId)));
        } else {
          console.warn("map error:", sourceId || "(no source)", msg);
        }
      });

      map.on("load", () => {
        map.addSource("blocks", { type: "vector", url: tileUrl(BASEMAP) });
        // There is no land or street layer, only blocks. A wide street-coloured stroke under the
        // block fill paints the gaps between blocks as streets (~20 m wide at every zoom), while
        // open water stays blue.
        map.addLayer({
          id: "streets",
          type: "line",
          source: "blocks",
          "source-layer": "blocks",
          layout: { "line-join": "round" },
          paint: {
            "line-color": MAP_COLORS.street,
            "line-width": ["interpolate", ["exponential", 2], ["zoom"], 12, 2, 18, 128],
          },
        });
        map.addLayer({
          id: "blocks-fill",
          type: "fill",
          source: "blocks",
          "source-layer": "blocks",
          paint: { "fill-color": MAP_COLORS.block },
        });
        map.addLayer({
          id: "blocks-line",
          type: "line",
          source: "blocks",
          "source-layer": "blocks",
          paint: {
            "line-color": MAP_COLORS.blockEdge,
            "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.3, 16, 0.8],
          },
        });
        // Scan layers are added on demand by paint() (see addScan), between here and the districts.
        map.addSource("selected", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        map.addLayer({ id: "selected-line", type: "line", source: "selected", paint: { "line-color": MAP_COLORS.accent, "line-width": 3 } });
        map.addSource("highlight", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        map.addLayer({ id: "highlight-line", type: "line", source: "highlight", paint: { "line-color": MAP_COLORS.accent, "line-width": 3 } });
        readyRef.current = true;
        addSharedDistricts();
        if (pending.current) {
          const p = pending.current;
          pending.current = null;
          applyNow(p);
        }
      });

      map.on("click", (e) => {
        const cb = clickRef.current;
        if (!cb) return;
        const s = stateRef.current;
        const click = ++clicks.current;
        if (s.view === "scan") return cb(null);
        const shared = sharedRef.current;
        const index = sharedIndex(s.era);
        if (!shared || index === undefined) return cb(null);
        const lot = lotAt(map, e.point, index);
        if (lot === null) return cb(null);
        const category = lotCategory(shared, index, lot) ?? undefined;
        // A later click, or a slow download overtaken by one, wins.
        loadLotCards(shared.cards).then(
          (cards) => {
            if (click !== clicks.current) return;
            selectedLot.current = lot;
            clickRef.current?.({ era: s.era, category, ...lotCard(cards, shared.eras[index], lot) });
          },
          (err) => {
            console.warn("lot cards:", err);
            if (click === clicks.current) clickRef.current?.(null);
          },
        );
      });

      cleanup = () => {
        if (E2E) w.__maps = (w.__maps ?? []).filter((m) => m !== map);
        map.remove();
      };
    }, () => setFailedSources((prev) => new Set(prev).add("maplibre")));

    return () => {
      cancelled = true;
      readyRef.current = false;
      for (const t of timers.values()) clearTimeout(t);
      if (holdTimer.current) clearInterval(holdTimer.current);
      holdTimer.current = null;
      holdRef.current = null;
      cleanup();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const era = ERAS.find((e) => e.id === state.era);
  const showCredit = state.era === "1948";
  const colours = state.view !== "scan";
  const showLegend = colours && avail[state.era] === true;
  const comingSoon = colours && avail[state.era] === false;
  // The notice is for a map that can't show the step: no basemap, or no scan for this era.
  const failed =
    failedSources.has("maplibre") ||
    failedSources.has("blocks") ||
    (era?.scanSeries ?? []).some((series) => failedSources.has(scanLayer(series)));

  return (
    <div ref={wrap} className={`map-wrap${className ? ` ${className}` : ""}`} data-view={state.view} data-era={state.era}>
      {/* The story map is a picture; the explorer's holds a focusable canvas and controls, so
          it is a labelled region rather than role="img" (which would hide them). */}
      <div
        ref={container}
        className="map"
        role={interactive ? "region" : "img"}
        aria-label={interactive ? "Zoning map explorer" : "Historic zoning map of San Francisco"}
      />
      {showLegend && <Legend compact={!interactive} />}
      {comingSoon && (
        <p className="map-pending" data-testid="classification-coming">
          District colours for {era?.label ?? state.era} are coming.
          {era?.scanSeries.length ? " This is the original map." : ""}
        </p>
      )}
      {showCredit && <p className="credit">{RUMSEY_CREDIT}</p>}
      {failed && (
        <p className="map-notice" role="alert">
          The map images could not be loaded. The story below still works; reload to try again.
        </p>
      )}
    </div>
  );
});

export default StoryMap;
