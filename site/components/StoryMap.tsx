"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import { Protocol } from "pmtiles";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";
import { ERAS } from "@/lib/eras";
import { BASEMAP, districtSource, scanSource, tileUrl } from "@/lib/tiles";
import { MAP_COLORS, scanInkWeight } from "@/lib/theme";
import { districtFillOpacity, fillColorExpression } from "@/lib/palette";
import { useDistrictAvailability, type Availability } from "@/lib/districts";
import Legend from "./Legend";
import { RUMSEY_CREDIT } from "./Credits";

export type MapState = {
  era: string;
  view: "scan" | "allowed" | "both";
  /** Omit to leave the camera where it is (the explorer, where the reader pans). */
  camera?: { center: [number, number]; zoom: number; bearing?: number };
  highlight?: Feature<Geometry> | FeatureCollection<Geometry>;
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

const FADE_MS = 700;
const BOTH_SCAN_WEIGHT = 0.6;
const scanLayer = (series: string) => `scan-${series}`;
const distSrc = (era: string) => `dist-${era}`;
const distFill = (era: string) => `dist-${era}-fill`;
const distLine = (era: string) => `dist-${era}-line`;
const allSeries = () => Array.from(new Set(ERAS.flatMap((e) => e.scanSeries)));

let protocolAdded = false;
function ensureProtocol() {
  if (protocolAdded) return;
  maplibregl.setWorkerUrl("/maplibre/maplibre-gl-worker.mjs");
  const protocol = new Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);
  protocolAdded = true;
}

function reducedMotion() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function scanOpacity(state: MapState, series: string, zoom: number, avail: Availability): number {
  const era = ERAS.find((e) => e.id === state.era);
  if (!era || !era.scanSeries.includes(series)) return 0;
  // "allowed" hides the scan only once the colours exist; until then the scan stands in.
  if (state.view === "allowed" && avail[era.id]) return 0;
  // In "both" the scan's solid black (1948 Commercial, say) would drown the colours; ease the
  // ink so labels and hatching show through the semi-transparent fills.
  const both = state.view === "both" && avail[era.id] ? BOTH_SCAN_WEIGHT : 1;
  return both * scanInkWeight(zoom);
}

const StoryMap = forwardRef<StoryMapHandle, Props>(function StoryMap(
  { initial, interactive = false, onParcelClick, selectedBlklot = null, className },
  ref,
) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const pending = useRef<MapState | null>(initial);
  const readyRef = useRef(false);
  const stateRef = useRef<MapState>(initial);
  const availRef = useRef<Availability>({});
  const hideTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const clickRef = useRef(onParcelClick);
  const [failed, setFailed] = useState(false);
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
    const zoom = s.camera?.zoom ?? map.getZoom();
    for (const series of allSeries()) {
      const id = scanLayer(series);
      if (!map.getLayer(id)) continue;
      map.setPaintProperty(id, "raster-opacity-transition", fade);
      map.setPaintProperty(id, "raster-opacity", scanOpacity(s, series, zoom, availRef.current));
    }
    for (const era of ERAS) {
      const fill = distFill(era.id);
      if (!map.getLayer(fill)) continue;
      const on = era.id === s.era && s.view !== "scan";
      const timer = hideTimers.current.get(era.id);
      if (timer) clearTimeout(timer);
      hideTimers.current.delete(era.id);
      const layers = [fill, distLine(era.id)];
      if (on) for (const l of layers) map.setLayoutProperty(l, "visibility", "visible");
      map.setPaintProperty(fill, "fill-opacity-transition", fade);
      map.setPaintProperty(fill, "fill-opacity", on ? districtFillOpacity(s.view) : 0);
      map.setPaintProperty(distLine(era.id), "line-opacity-transition", fade);
      map.setPaintProperty(distLine(era.id), "line-opacity", on ? (s.view === "both" ? 0.9 : 0.6) : 0);
      if (!on && map.getLayoutProperty(fill, "visibility") !== "none") {
        // Hide after the fade so off-screen eras stop loading tiles.
        const hide = () => {
          if (!mapRef.current || stateRef.current.era === era.id && stateRef.current.view !== "scan") return;
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
      const src = map.getSource("highlight") as maplibregl.GeoJSONSource | undefined;
      src?.setData(s.highlight ?? { type: "FeatureCollection", features: [] });
      if (s.camera) {
        const cam = { center: s.camera.center, zoom: s.camera.zoom, bearing: s.camera.bearing ?? 0 };
        if (reducedMotion()) map.jumpTo(cam);
        else map.easeTo({ ...cam, duration: 1400 });
      }
    },
    [paint],
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
            "fill-color": fillColorExpression() as maplibregl.ExpressionSpecification,
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
    const src = map.getSource("selected") as maplibregl.GeoJSONSource | undefined;
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
    ensureProtocol();
    const cam = initial.camera ?? { center: [-122.443, 37.758] as [number, number], zoom: 12.3 };
    const map = new maplibregl.Map({
      container: container.current,
      style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": MAP_COLORS.water } }] },
      center: cam.center,
      zoom: cam.zoom,
      bearing: cam.bearing ?? 0,
      interactive,
      attributionControl: false,
      maxZoom: 18,
      fadeDuration: 0,
    });
    mapRef.current = map;
    const w = window as unknown as { __map?: MlMap; __maps?: MlMap[] };
    if (!interactive) w.__map = map;
    (w.__maps ??= []).push(map);

    map.on("error", (e) => {
      const msg = String((e as unknown as { error?: Error }).error?.message ?? "");
      const sourceId = String((e as unknown as { sourceId?: string }).sourceId ?? "");
      // District tiles are optional and published one series at a time (a file can be
      // mid-write); a district error never blanks the map, so it never raises the notice.
      if (sourceId.startsWith("dist-")) {
        console.warn("district tiles:", sourceId, msg);
        return;
      }
      // Tile/source fetch failure (server down, 404). Keep the page; show a notice.
      if (/fetch|network|pmtiles|load|status|\b404\b/i.test(msg) || msg === "") setFailed(true);
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
      for (const series of allSeries()) {
        map.addSource(scanLayer(series), { type: "raster", url: scanSource(series), tileSize: 256 });
        map.addLayer({
          id: scanLayer(series),
          type: "raster",
          source: scanLayer(series),
          paint: { "raster-opacity": 0, "raster-fade-duration": 0 },
        });
      }
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

    // Scan ink eases with zoom; in the explorer the reader zooms, so repaint as they do.
    map.on("zoomend", () => {
      if (interactive) paint({ ...stateRef.current, camera: undefined });
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

    const timers = hideTimers.current;
    return () => {
      readyRef.current = false;
      for (const t of timers.values()) clearTimeout(t);
      w.__maps = (w.__maps ?? []).filter((m) => m !== map);
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const m = mapRef.current;
    if (!m) return;
    const fn = interactive ? "enable" : "disable";
    for (const h of ["scrollZoom", "boxZoom", "dragRotate", "dragPan", "keyboard", "doubleClickZoom", "touchZoomRotate", "touchPitch"] as const) {
      m[h][fn]();
    }
  }, [interactive]);

  const era = ERAS.find((e) => e.id === state.era);
  const showCredit = state.era === "1948";
  const colours = state.view !== "scan";
  const showLegend = colours && avail[state.era] === true;
  const comingSoon = colours && avail[state.era] === false;

  return (
    <div className={`map-wrap${className ? ` ${className}` : ""}`} data-view={state.view} data-era={state.era}>
      <div ref={container} className="map" aria-label="Historic zoning map of San Francisco" role="img" />
      {showLegend && <Legend />}
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
