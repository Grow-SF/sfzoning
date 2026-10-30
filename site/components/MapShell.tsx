"use client";

import { useEffect, useRef } from "react";
import StoryMap, { type MapState, type StoryMapHandle } from "./StoryMap";

const SF: [number, number] = [-122.4376, 37.7555];

const INITIAL: MapState = { era: "1921", view: "scan", camera: { center: SF, zoom: 11.6 } };

/** Dev hook: /?era=1948&zoom=13 jumps the shell to that state (used for screenshots). */
export default function MapShell() {
  const ref = useRef<StoryMapHandle>(null);
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const era = q.get("era");
    if (!era) return;
    const zoom = Number(q.get("zoom") ?? 13);
    ref.current?.apply({ era, view: "scan", camera: { center: SF, zoom } });
  }, []);
  return <StoryMap ref={ref} initial={INITIAL} />;
}
