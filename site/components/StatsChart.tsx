"use client";

import { useEffect, useRef, useState } from "react";
import { ERAS } from "@/lib/eras";
import { loadStats } from "@/lib/districts";
import { threePlusShare } from "@/lib/stats";
import styles from "./explorer.module.css";

type Point = { id: string; label: string; year: number; share: number; partial: boolean };

// An era whose read area is under this fraction of the best-covered era is drawn hollow
// and left out of the line: its share describes a few sheets, not the city.
const PARTIAL_BELOW = 0.6;

const H = 230;
const M = { top: 22, right: 18, bottom: 30, left: 40 };
const X0 = 1915;
const X1 = 2030;
const y = (share: number) => M.top + (1 - share) * (H - M.top - M.bottom);
const pct = (s: number) => `${Math.round(s * 100)}%`;

export default function StatsChart() {
  const [points, setPoints] = useState<Point[] | null>(null);
  // The SVG is drawn at its rendered width so type stays at its set size on phones.
  const figRef = useRef<HTMLElement>(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = figRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, [points]);
  const x = (yr: number) => M.left + ((yr - X0) / (X1 - X0)) * (W - M.left - M.right);

  useEffect(() => {
    let live = true;
    const eras = ERAS.filter((e) => e.districtSeries);
    Promise.all(eras.map((e) => loadStats(e.districtSeries!))).then((all) => {
      if (!live) return;
      const areas = all.map((s) => s?.mapped_area_sqft ?? 0);
      const maxArea = Math.max(0, ...areas);
      const pts: Point[] = [];
      all.forEach((s, i) => {
        if (!s) return;
        const share = threePlusShare(s);
        if (share === null) return;
        const e = eras[i];
        pts.push({ id: e.id, label: e.label, year: e.year, share, partial: maxArea > 0 && areas[i] < maxArea * PARTIAL_BELOW });
      });
      setPoints(pts);
    });
    return () => {
      live = false;
    };
  }, []);

  if (!points || points.length === 0) return null;

  const whole = points.filter((p) => !p.partial);
  const line = whole.map((p, i) => `${i ? "L" : "M"}${x(p.year).toFixed(1)},${y(p.share).toFixed(1)}`).join(" ");
  const anyPartial = points.some((p) => p.partial);
  const decades = W < 480 ? [1920, 1960, 2000] : [1920, 1940, 1960, 1980, 2000, 2020];
  // Direct-label selectively: skip a value label that would collide with the one before it
  // (the value stays in the dot's tooltip and the table). The latest era is always labelled.
  const labelled = new Set<string>();
  let lastX = -Infinity;
  points.forEach((p, i) => {
    const px = x(p.year);
    const last = i === points.length - 1;
    if (px - lastX >= 34 || last) {
      if (last && px - lastX < 34) labelled.delete(points[i - 1].id);
      labelled.add(p.id);
      lastX = px;
    }
  });

  return (
    <figure ref={figRef} className={styles.chart} data-testid="stats-chart">
      <figcaption>
        <h3 className={styles.chartTitle}>Where three or more homes were legal</h3>
        <p className={styles.chartDek}>
          Of the lots zoned for homes, the share whose district allowed three or more per lot, as read from each
          era&rsquo;s map.
        </p>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.chartSvg} role="img" aria-labelledby="stats-table-caption">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className={t === 0 ? styles.axis : styles.grid} />
            <text x={M.left - 8} y={y(t)} className={styles.yTick} textAnchor="end" dominantBaseline="middle">
              {pct(t)}
            </text>
          </g>
        ))}
        {decades.map((d) => (
          <text key={d} x={x(d)} y={H - M.bottom + 18} className={styles.xTick} textAnchor="middle">
            {d}
          </text>
        ))}
        {line && whole.length > 1 && <path d={line} className={styles.line} />}
        {points.map((p) => (
          <g key={p.id} className={styles.dot} tabIndex={0} aria-label={`${p.label}: ${pct(p.share)}${p.partial ? ", partial map" : ""}`}>
            <title>{`${p.label}: ${pct(p.share)}${p.partial ? " (partial map)" : ""}`}</title>
            <circle cx={x(p.year)} cy={y(p.share)} r={12} className={styles.hit} />
            <circle cx={x(p.year)} cy={y(p.share)} r={4.5} className={p.partial ? styles.hollow : styles.solid} />
            {labelled.has(p.id) && (
              <text x={x(p.year)} y={y(p.share) - 10} textAnchor="middle" className={styles.value}>
                {pct(p.share)}
              </text>
            )}
          </g>
        ))}
      </svg>
      {anyPartial && (
        <p className={styles.chartNote}>
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="3.8" className={styles.hollow} />
          </svg>{" "}
          Hollow dots: only part of that era&rsquo;s map has been read so far, so the share describes a few sheets, not the
          city.
        </p>
      )}
      <table className={styles.srOnly}>
        <caption id="stats-table-caption">Share of residential lots allowing three or more homes, by era</caption>
        <thead>
          <tr>
            <th scope="col">Era</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.id}>
              <th scope="row">{p.label}</th>
              <td>
                {pct(p.share)}
                {p.partial ? " (partial map)" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
