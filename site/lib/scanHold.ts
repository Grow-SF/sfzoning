/** Longest the old scan stays under a new one that has not arrived. */
export const HOLD_LIMIT_MS = 8000;

/** Scans kept visible under the new one, and when the hold began. */
export type Hold = { held: string[]; since: number };

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
