/** Longest the old scan stays under a new one that has not arrived. */
export const HOLD_LIMIT_MS = 8000;

/** Held alongside scans: the old colours stay up until the new scan has arrived. */
export const COLOURS = "colours";

/** At most this many scans are held, so a fast scroll can't keep many of them fetching. */
const MAX_HELD_SCANS = 2;

/** Scans kept visible under the new one, and when the hold began. */
export type Hold = { held: string[]; since: number };

/**
 * What to hold on a step that changes the wanted scans: those already held, then the ones it leaves,
 * which are the screen's current picture; the oldest go past the limit. The colours stay only on a
 * step drawing a scan alone (`scanAlone`), if on screen now (`coloursDrawn`) or held already.
 */
export function nextHeld(held: string[], gone: string[], wanted: string[], scanAlone: boolean, coloursDrawn: boolean): string[] {
  const scans = [...held.filter((x) => x !== COLOURS && !gone.includes(x)), ...gone]
    .filter((x) => !wanted.includes(x))
    .slice(-MAX_HELD_SCANS);
  return scanAlone && (coloursDrawn || held.includes(COLOURS)) ? [...scans, COLOURS] : scans;
}

/**
 * The scans to keep under the wanted ones, or null when none are left: the wanted scans are loaded
 * and the camera has stopped, or the limit has passed. (The new scan has been fading in since the
 * step began, so there is no waiting for that.) A held scan that is wanted again stops being held.
 */
export function heldScans(
  hold: Hold | null,
  wanted: string[],
  loaded: (series: string) => boolean,
  stopped: boolean,
  now: number,
): Hold | null {
  if (!hold || now >= hold.since + HOLD_LIMIT_MS) return null;
  const held = hold.held.filter((s) => !wanted.includes(s));
  if (!held.length) return null;
  if (stopped && wanted.every(loaded)) return null;
  return { held, since: hold.since };
}
