"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type * as Ml from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";
import { ERAS } from "@/lib/eras";
import { BASEMAP, districtSource, scanSource, tileUrl } from "@/lib/tiles";
import { INK_ZOOM, MAP_COLORS, inkByZoom } from "@/lib/theme";
import { ALLOWED_FILL_OPACITY, BOTH_SCAN_WEIGHT, districtFillOpacity, fillColorExpression } from "@/lib/palette";
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
/** Below this, tiles still load but nothing visible is drawn. */
const WARM = 0.001;
/** In "both" at city zoom the colours carry the view; the scan fades to a trace of itself. */
const BOTH_CITY_INK = 0.3;
const scanLayer = (series: string) => `scan-${series}`;
const distSrc = (era: string) => `dist-${era}`;
const distFill = (era: string) => `dist-${era}-fill`;
const distLine = (era: string) => `dist-${era}-line`;
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
  return inkByZoom(1) as Expr;
}

/** The scan series that `s` draws at all (current era or prefetch warm-up). */
export function scansShown(s: MapState, avail: Availability): Set<string> {
  return new Set(allSeries().filter((series) => scanOpacity(s, series, avail) !== 0));
}

/** Add a scan's source and layer, under the district layers, if it isn't there yet. */
function addScan(map: MlMap, series: string) {
  const id = scanLayer(series);
  if (map.getLayer(id)) return;
  map.addSource(id, { type: "raster", url: scanSource(series), tileSize: 256 });
  const before = map.getStyle().layers.find((l) => l.id.startsWith("dist-") || l.id === "selected-line")?.id;
  map.addLayer(
    { id, type: "raster", source: id, layout: { visibility: "none" }, paint: { "raster-opacity": 0, "raster-fade-duration": 0 } },
    before,
  );
}

function fillOpacity(view: MapState["view"]): Opacity {
  if (view !== "both") return districtFillOpacity(view);
  // City zoom: "allowed" strength (the scan is mostly faded). Close up: let the scan through.
  return ["interpolate", ["linear"], ["zoom"], INK_ZOOM.lo + 0.5, ALLOWED_FILL_OPACITY, INK_ZOOM.hi - 0.5, districtFillOpacity("both")];
}

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
  const clickRef = useRef(onParcelClick);
  /** Sources that failed to load ("blocks", "scan-<series>", or "maplibre" itself). */
  const [failedSources, setFailedSources] = useState<ReadonlySet<string>>(() => new Set());
  const [state, setState] = useState<MapState>(initial);
  const avail = useDistrictAvailability();

  useEffect(() => {
    clickRef.current = onParcelClick;
  }, [onParcelClick]);

  /** Paint every layer for `s` without moving the camera. */
  const paint = useCallback((s: MapState) => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const instant = reducedMotion();
    const fade = { duration: instant ? 0 : FADE_MS, delay: 0 };
    // Only the current era's scan and the prefetch era's are visible (MapLibre loads tiles for
    // every visible layer, even at zero opacity). A scan's source is added the first time it is
    // needed, so eras the reader never reaches cost no requests at all.
    const shown = scansShown(s, availRef.current);
    for (const series of allSeries()) {
      const id = scanLayer(series);
      const key = `scan:${series}`;
      const timer = hideTimers.current.get(key);
      if (timer) clearTimeout(timer);
      hideTimers.current.delete(key);
      if (shown.has(series)) {
        addScan(map, series);
        map.setLayoutProperty(id, "visibility", "visible");
      }
      if (!map.getLayer(id)) continue;
      map.setPaintProperty(id, "raster-opacity-transition", fade);
      map.setPaintProperty(id, "raster-opacity", scanOpacity(s, series, availRef.current));
      if (!shown.has(series) && map.getLayoutProperty(id, "visibility") !== "none") {
        // Hide once faded out, unless a fast scroll has made this scan wanted again.
        const hide = () => {
          const m = mapRef.current;
          if (!m || !m.getLayer(id) || scansShown(stateRef.current, availRef.current).has(series)) return;
          m.setLayoutProperty(id, "visibility", "none");
        };
        if (instant) hide();
        else hideTimers.current.set(key, setTimeout(hide, FADE_MS + 50));
      }
    }
    for (const era of ERAS) {
      const fill = distFill(era.id);
      if (!map.getLayer(fill)) continue;
      const on = era.id === s.era && s.view !== "scan";
      const warm = !on && s.prefetch?.era === era.id && s.prefetch.view !== "scan";
      const timer = hideTimers.current.get(era.id);
      if (timer) clearTimeout(timer);
      hideTimers.current.delete(era.id);
      const layers = [fill, distLine(era.id)];
      if (on || warm) for (const l of layers) map.setLayoutProperty(l, "visibility", "visible");
      map.setPaintProperty(fill, "fill-opacity-transition", fade);
      map.setPaintProperty(fill, "fill-opacity", on ? fillOpacity(s.view) : warm ? WARM : 0);
      map.setPaintProperty(distLine(era.id), "line-opacity-transition", fade);
      map.setPaintProperty(distLine(era.id), "line-opacity", on ? (s.view === "both" ? 0.9 : 0.6) : 0);
      if (!on && !warm && map.getLayoutProperty(fill, "visibility") !== "none") {
        // Hide after the fade so off-screen eras stop loading tiles.
        const hide = () => {
          const cur = stateRef.current;
          if (!mapRef.current || (cur.era === era.id && cur.view !== "scan") || cur.prefetch?.era === era.id) return;
          for (const l of layers) if (mapRef.current.getLayer(l)) mapRef.current.setLayoutProperty(l, "visibility", "none");
        };
        if (instant) hide();
        else hideTimers.current.set(era.id, setTimeout(hide, FADE_MS + 50));
      }
    }
  }, []);

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
        else map.easeTo({ ...cam, duration: 1400 });
      }
    },
    [paint, interactive],
  );

  useImperativeHandle(ref, () => ({ apply: applyNow }), [applyNow]);

  // Add an era's district layers once its tiles are known to exist.
  const addDistricts = useCallback(
    (eraId: string) => {
      const map = mapRef.current;
      const era = ERAS.find((e) => e.id === eraId);
      if (!map || !readyRef.current || !era?.districtSeries || map.getSource(distSrc(eraId))) return;
      map.addSource(distSrc(eraId), { type: "vector", url: districtSource(era.districtSeries) });
      const hidden = { visibility: "none" as const };
      map.addLayer(
        {
          id: distFill(eraId),
          type: "fill",
          source: distSrc(eraId),
          "source-layer": "parcels",
          layout: hidden,
          paint: {
            "fill-color": fillColorExpression() as Expr,
            "fill-opacity": 0,
            "fill-outline-color": "rgba(0,0,0,0)",
          },
        },
        "selected-line",
      );
      map.addLayer(
        {
          id: distLine(eraId),
          type: "line",
          source: distSrc(eraId),
          "source-layer": "districts",
          layout: hidden,
          paint: {
            "line-color": MAP_COLORS.districtEdge,
            "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.4, 16, 1.4],
            "line-opacity": 0,
          },
        },
        "selected-line",
      );
      map.on("mouseenter", distFill(eraId), () => {
        if (clickRef.current) map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", distFill(eraId), () => {
        map.getCanvas().style.cursor = "";
      });
      paint(stateRef.current);
    },
    [paint],
  );

  // The selected lot is outlined from its own GeoJSON source. (A filtered line layer on the
  // district source made MapLibre drop that lot's fill when the filter changed.)
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !readyRef.current) return;
    const src = map.getSource("selected") as Ml.GeoJSONSource | undefined;
    const era = stateRef.current.era;
    const features =
      selectedBlklot && map.getSource(distSrc(era))
        ? map.querySourceFeatures(distSrc(era), { sourceLayer: "parcels", filter: ["==", ["get", "blklot"], selectedBlklot] })
        : [];
    src?.setData({
      type: "FeatureCollection",
      features: features.map((f) => ({ type: "Feature", properties: {}, geometry: f.geometry })),
    });
  }, [selectedBlklot]);

  useEffect(() => {
    availRef.current = avail;
    for (const [eraId, ok] of Object.entries(avail)) if (ok) addDistricts(eraId);
    paint(stateRef.current);
  }, [avail, addDistricts, paint]);

  useEffect(() => {
    if (!container.current) return;
    let cancelled = false;
    let cleanup = () => {};
    const timers = hideTimers.current;
    const setIdle = (v: boolean) => {
      // Test and screenshot hook: true once the map has drawn everything it is going to.
      if (E2E && wrap.current) wrap.current.dataset.idle = String(v);
    };

    loadMaplibre().then((maplibregl) => {
      if (cancelled || !container.current) return;
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
        // District tiles are optional and published one series at a time (a file can be
        // mid-write); a district error never blanks the map, so it never raises the notice.
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
        for (const [eraId, ok] of Object.entries(availRef.current)) if (ok) addDistricts(eraId);
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
        const layer = distFill(s.era);
        if (s.view === "scan" || !map.getLayer(layer)) return cb(null);
        const f = map.queryRenderedFeatures(e.point, { layers: [layer] })[0];
        if (!f) return cb(null);
        const p = f.properties as Record<string, string | number | undefined>;
        cb({
          era: s.era,
          blklot: p.blklot != null ? String(p.blklot) : undefined,
          code: p.code != null ? String(p.code) : undefined,
          height_code: p.height_code != null ? String(p.height_code) : undefined,
          category: p.category != null ? String(p.category) : undefined,
          confidence: p.confidence,
        });
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
