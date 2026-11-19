import record from "./download-releases.json";
import type { Release, ReleaseRecord } from "./releases.mjs";

const RECORD = record as ReleaseRecord;

/** Every release, oldest first. */
export const allReleases = (r: ReleaseRecord = RECORD): Release[] => r.releases;

/** One release by id; `"latest"` is the newest. */
export function findRelease(id: string, r: ReleaseRecord = RECORD): Release | undefined {
  const all = allReleases(r);
  return id === "latest" ? all.at(-1) : all.find((x) => x.id === id);
}

/** The release before `id` (or before the newest for `"latest"`); undefined for the first. */
export function previousRelease(id: string, r: ReleaseRecord = RECORD): Release | undefined {
  const all = allReleases(r);
  const i = id === "latest" ? all.length - 1 : all.findIndex((x) => x.id === id);
  return i > 0 ? all[i - 1] : undefined;
}

export function isLatest(id: string, r: ReleaseRecord = RECORD): boolean {
  return id === "latest" || allReleases(r).at(-1)?.id === id;
}
