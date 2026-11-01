"use client";

import { useEffect, useState } from "react";
import { ERAS } from "./eras";
import { loadSharedDistricts } from "./shared";
import type { DistrictStats } from "./stats";
import { districtStatsUrl } from "./tiles";

/** Era id → true (has districts), false (none yet), absent (still checking). */
export type Availability = Record<string, boolean>;

/** The eras the shared districts cover. An era they don't shows its scan and says colours are coming. */
export function useDistrictAvailability(): Availability {
  const [avail, setAvail] = useState<Availability>({});
  useEffect(() => {
    let live = true;
    loadSharedDistricts().then((shared) => {
      if (!live) return;
      setAvail(Object.fromEntries(ERAS.map((e) => [e.id, !!e.districtSeries && !!shared?.eras.includes(e.districtSeries)])));
    });
    return () => {
      live = false;
    };
  }, []);
  return avail;
}

const statsCache = new Map<string, Promise<DistrictStats | null>>();

export function loadStats(series: string): Promise<DistrictStats | null> {
  let p = statsCache.get(series);
  if (!p) {
    p = fetch(districtStatsUrl(series))
      .then((r) => (r.ok ? (r.json() as Promise<DistrictStats>) : null))
      .catch(() => null);
    statsCache.set(series, p);
  }
  return p;
}
