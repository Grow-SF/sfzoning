"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { Map as MlMap } from "maplibre-gl";
import { Protocol } from "pmtiles";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";
import { ERAS } from "@/lib/eras";
import { BASEMAP, scanSource, tileUrl } from "@/lib/tiles";

export type MapState = {
  era: string;
  view: "scan" | "allowed" | "both";
  camera: { center: [number, number]; zoom: number; bearing?: number };
  highlight?: Feature<Geometry> | FeatureCollection<Geometry>;
};

export type StoryMapHandle = { apply: (state: MapState) => void };

type Props = {
  initial: MapState;
  /** Gestures are off during the story and on in the explorer. */
  interactive?: boolean;
};

const RUMSEY_CREDIT = "© Cartography Associates, David Rumsey Map Collection";
const FADE_MS = 700;
const layerId = (series: string) => `scan-${series}`;
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

function targetOpacity(state: MapState, series: string): number {
  const era = ERAS.find((e) => e.id === state.era);
  if (!era || !era.scanSeries.includes(series)) return 0;
  // District classification tiles arrive in step two; until then every view shows the scan.
  return state.view === "both" ? 0.7 : 1;
}

const StoryMap = forwardRef<StoryMapHandle, Props>(function StoryMap({ initial, interactive = false }, ref) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const pending = useRef<MapState | null>(initial);
  const readyRef = useRef(false);
  const [failed, setFailed] = useState(false);
  const [state, setState] = useState<MapState>(initial);

  const applyNow = useCallback((s: MapState) => {
    const map = mapRef.current;
    if (!map || !readyRef.current) {
      pending.current = s;
      return;
    }
    const instant = reducedMotion();
    for (const series of allSeries()) {
      const id = layerId(series);
      if (!map.getLayer(id)) continue;
      map.setPaintProperty(id, "raster-opacity-transition", { duration: instant ? 0 : FADE_MS, delay: 0 });
      map.setPaintProperty(id, "raster-opacity", targetOpacity(s, series));
    }
    const src = map.getSource("highlight") as maplibregl.GeoJSONSource | undefined;
    src?.setData(s.highlight ?? { type: "FeatureCollection", features: [] });
    const cam = { center: s.camera.center, zoom: s.camera.zoom, bearing: s.camera.bearing ?? 0 };
    if (instant) map.jumpTo(cam);
    else map.easeTo({ ...cam, duration: 1400 });
    setState(s);
  }, []);

  useImperativeHandle(ref, () => ({ apply: applyNow }), [applyNow]);

  useEffect(() => {
    if (!container.current) return;
    ensureProtocol();
    const map = new maplibregl.Map({
      container: container.current,
      style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#1c2229" } }] },
      center: initial.camera.center,
      zoom: initial.camera.zoom,
      bearing: initial.camera.bearing ?? 0,
      interactive,
      attributionControl: false,
      maxZoom: 18,
      fadeDuration: 0,
    });
    mapRef.current = map;
    (window as unknown as { __map?: MlMap }).__map = map;

    map.on("error", (e) => {
      const msg = String((e as unknown as { error?: Error }).error?.message ?? "");
      // Tile/source fetch failure (server down, 404). Keep the page; show a notice.
      if (/fetch|network|pmtiles|load|status|\b404\b/i.test(msg) || msg === "") setFailed(true);
    });

    map.on("load", () => {
      map.addSource("blocks", { type: "vector", url: tileUrl(BASEMAP) });
      map.addLayer({ id: "blocks-fill", type: "fill", source: "blocks", "source-layer": "blocks", paint: { "fill-color": "#232b33" } });
      map.addLayer({ id: "blocks-line", type: "line", source: "blocks", "source-layer": "blocks", paint: { "line-color": "#46535f", "line-width": 0.6 } });
      for (const series of allSeries()) {
        map.addSource(layerId(series), { type: "raster", url: scanSource(series), tileSize: 256 });
        map.addLayer({
          id: layerId(series),
          type: "raster",
          source: layerId(series),
          paint: { "raster-opacity": 0, "raster-fade-duration": 0 },
        });
      }
      map.addSource("highlight", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: "highlight-line", type: "line", source: "highlight", paint: { "line-color": "#ffb703", "line-width": 3 } });
      readyRef.current = true;
      if (pending.current) {
        const p = pending.current;
        pending.current = null;
        applyNow(p);
      }
    });

    return () => {
      readyRef.current = false;
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

  const showCredit = state.era === "1948";

  return (
    <div className="map-wrap">
      <div ref={container} className="map" aria-label="Historic zoning map of San Francisco" role="img" />
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
