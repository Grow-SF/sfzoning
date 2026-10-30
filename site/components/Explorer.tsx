"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import StoryMap, { type ParcelHit, type StoryMapHandle } from "./StoryMap";
import ParcelPopup from "./ParcelPopup";
import StatsChart from "./StatsChart";
import { ERAS } from "@/lib/eras";
import { loadRules, type Rule } from "@/lib/rules";
import type { View } from "@/content/schema";
import styles from "./explorer.module.css";

const VIEW_LABELS: Record<View, string> = { scan: "Original map", allowed: "What it allowed", both: "Both" };
const START = { center: [-122.443, 37.765] as [number, number], zoom: 12.4 };

export default function Explorer() {
  const mapRef = useRef<StoryMapHandle>(null);
  const [index, setIndex] = useState(() => ERAS.findIndex((e) => e.id === "1948"));
  const [view, setView] = useState<View>("both");
  const [hit, setHit] = useState<ParcelHit | null>(null);
  const [rules, setRules] = useState<Rule[]>([]);
  const sliderId = useId();
  const era = ERAS[index];
  const noScan = era.scanSeries.length === 0;
  // Today has no scan; its only honest view is the colours.
  const effectiveView: View = noScan ? "allowed" : view;

  // Mount the explorer's map only as the reader nears it: a second WebGL map loading tiles
  // under the story would compete with the story map for the GPU and the network.
  const frameRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = frameRef.current;
    if (!el || near) return;
    const io = new IntersectionObserver((entries) => entries.some((e) => e.isIntersecting) && setNear(true), {
      rootMargin: "100% 0px",
    });
    io.observe(el);
    return () => io.disconnect();
  }, [near]);

  useEffect(() => {
    loadRules().then(setRules);
  }, []);

  useEffect(() => {
    mapRef.current?.apply({ era: era.id, view: effectiveView });
  }, [era.id, effectiveView]);

  // A lot card belongs to the era it was opened in; moving the slider closes it.
  const shown = hit && hit.era === era.id && effectiveView !== "scan" ? hit : null;
  const onParcel = useCallback((h: ParcelHit | null) => setHit(h), []);
  const close = useCallback(() => setHit(null), []);

  return (
    <section className={styles.explorer} id="explore" aria-labelledby="explore-heading" data-testid="explorer">
      <header className={styles.intro}>
        <h2 id="explore-heading" className={styles.heading}>
          Explore every map
        </h2>
        <p className={styles.dek}>
          Slide through the years, switch between the original map and what it allowed, and tap a lot to read its
          rules.
        </p>
      </header>

      <div className={styles.controls}>
        <div className={styles.era}>
          <label htmlFor={sliderId} className={styles.eraLabel}>
            <span className={styles.srOnly}>Year</span>
            <output className={styles.eraValue} aria-live="polite" data-testid="explorer-era">
              {era.label}
            </output>
          </label>
          <div className={styles.slider}>
            <input
              id={sliderId}
              type="range"
              min={0}
              max={ERAS.length - 1}
              step={1}
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
              aria-valuetext={era.label}
              list={`${sliderId}-ticks`}
            />
            <datalist id={`${sliderId}-ticks`}>
              {ERAS.map((e, i) => (
                <option key={e.id} value={i} label={e.label} />
              ))}
            </datalist>
            <ol className={styles.ticks} aria-hidden>
              {ERAS.map((e, i) => (
                <li
                  key={e.id}
                  style={{ left: `${(i / (ERAS.length - 1)) * 100}%` }}
                  data-on={i === index || undefined}
                  data-major={i === 0 || i === ERAS.length - 1 || undefined}
                >
                  <span>{e.label}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>

        <fieldset className={styles.views}>
          <legend className={styles.srOnly}>Map view</legend>
          {(Object.keys(VIEW_LABELS) as View[]).map((v) => (
            <label key={v} data-on={effectiveView === v || undefined} data-disabled={(noScan && v !== "allowed") || undefined}>
              <input
                type="radio"
                name="explorer-view"
                value={v}
                checked={effectiveView === v}
                disabled={noScan && v !== "allowed"}
                onChange={() => setView(v)}
              />
              {VIEW_LABELS[v]}
            </label>
          ))}
        </fieldset>
      </div>

      <div className={styles.frame} ref={frameRef}>
        {near && (
          <StoryMap
            ref={mapRef}
            className="explorer-map"
            interactive
            initial={{ era: era.id, view: effectiveView, camera: START }}
            onParcelClick={onParcel}
            selectedBlklot={shown?.blklot ?? null}
          />
        )}
        {shown && <ParcelPopup hit={shown} era={era} rules={rules} onClose={close} />}
        {noScan && view !== "allowed" && (
          <p className={styles.frameNote}>There is no scanned map for today, only the current zoning.</p>
        )}
      </div>

      <StatsChart />
    </section>
  );
}
