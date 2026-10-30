"use client";

import { useEffect, useState } from "react";
import { ERAS } from "./eras";
import type { DistrictStats } from "./stats";
import { districtPath, districtStatsUrl, tilesHttpUrl } from "./tiles";

// Series are published one by one, so each era's district tiles are probed before the map
// adds them: a missing file never reaches MapLibre (no error events, no blank map), and the
// UI shows "classification coming" instead.

const probes = new Map<string, Promise<boolean>>();

/** True if out/<series>/districts/<series>-districts.pmtiles is being served. */
export function probeDistricts(series: string): Promise<boolean> {
  let p = probes.get(series);
  if (!p) {
    p = fetch(tilesHttpUrl(districtPath(series)), { headers: { Range: "bytes=0-6" } })
      .then((r) => r.ok)
      .catch(() => false);
    probes.set(series, p);
  }
  return p;
}

/** Era id → true (districts published), false (missing), absent (still checking). */
export type Availability = Record<string, boolean>;

export function useDistrictAvailability(): Availability {
  const [avail, setAvail] = useState<Availability>({});
  useEffect(() => {
    let live = true;
    for (const era of ERAS) {
      const probe = era.districtSeries ? probeDistricts(era.districtSeries) : Promise.resolve(false);
      probe.then((ok) => {
        if (live) setAvail((a) => ({ ...a, [era.id]: ok }));
      });
    }
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
