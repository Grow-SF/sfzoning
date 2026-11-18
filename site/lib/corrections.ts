import { useSyncExternalStore } from "react";
import type { Rule } from "./rules";

// Readers' corrections of the codes the map reader got wrong. Kept in the browser until they're
// sent: the reader collects them across lots and eras, then opens a pre-filled GitHub issue (or
// downloads the same content as a file). The pipeline imports the issue's fenced block
// (`sfzmaps corrections`, phase two) into inputs/<series>/overrides.csv after checking each
// against the scan.

export const REPO = "Grow-SF/sfzoning";
export const FENCE = "sfzoning-corrections";
/** GitHub accepts long issue links, but browsers and proxies cut them; past this, download. */
export const MAX_URL = 8000;

export type Correction = {
  /** Era id (ERAS), e.g. "1980" */
  era: string;
  /** The district series the era's colours come from, e.g. "1980-1990-zoning" */
  series: string;
  blklot: string;
  /** The code the site shows for the lot ("" when none was read) */
  shown: string;
  /** The code the reader says the old map prints for the lot */
  proposed: string;
  note: string;
};

const KEY = "sfz-corrections";
const listeners = new Set<() => void>();
let current: Correction[] | null = null;
const EMPTY: Correction[] = [];

function read(): Correction[] {
  if (current) return current;
  try {
    const raw = JSON.parse(window.localStorage.getItem(KEY) || "[]");
    current = Array.isArray(raw) ? raw.filter(isCorrection) : [];
  } catch {
    current = [];
  }
  return current;
}

function isCorrection(c: unknown): c is Correction {
  const o = c as Correction;
  return !!o && typeof o.era === "string" && typeof o.blklot === "string" && typeof o.proposed === "string";
}

function write(next: Correction[]) {
  current = next;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Private windows and blocked storage: the list lasts for this page only.
  }
  listeners.forEach((l) => l());
}

export function getCorrections(): Correction[] {
  return typeof window === "undefined" ? EMPTY : read();
}

/** Add a correction; a second one for the same lot and era replaces the first. */
export function addCorrection(c: Correction) {
  write([...getCorrections().filter((x) => !(x.era === c.era && x.blklot === c.blklot)), c]);
}

export function removeCorrection(era: string, blklot: string) {
  write(getCorrections().filter((x) => !(x.era === era && x.blklot === blklot)));
}

export function clearCorrections() {
  write([]);
}

export function subscribeCorrections(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useCorrections(): Correction[] {
  return useSyncExternalStore(subscribeCorrections, getCorrections, () => EMPTY);
}

/** For tests: forget the cached list so the next read goes to storage. */
export function resetCorrections() {
  current = null;
}

/** The district codes in force in `year` (the rules the site knows), sorted, for the pick-list. */
export function codesForYear(rules: Rule[], year: number): string[] {
  const out = new Set<string>();
  for (const r of rules) {
    if (r.era_start > year || (r.era_end !== null && year >= r.era_end)) continue;
    for (const c of r.code.split("/")) if (c.trim()) out.add(c.trim());
  }
  return [...out].sort((a, b) => a.localeCompare(b, "en", { numeric: true }));
}

function lotLabel(b: string): string {
  return b.length >= 7 ? `${b.slice(0, 4)}/${b.slice(4)}` : b;
}

/** The issue's text: a table for people, then a fenced JSON block the pipeline reads. */
export function issueBody(list: Correction[]): string {
  const rows = list.map(
    (c) => `| ${c.era} | ${lotLabel(c.blklot)} | ${c.shown || "none"} | ${c.proposed} | ${c.note.replace(/\|/g, "/").replace(/\n/g, " ")} |`,
  );
  const data = list.map(({ era, series, blklot, shown, proposed, note }) => ({ era, series, blklot, shown, proposed, note }));
  return [
    "Corrections suggested on sfzoning.com: what the old map prints for these lots.",
    "",
    "| Era | Block/lot | Site shows | Map prints | Note |",
    "|---|---|---|---|---|",
    ...rows,
    "",
    "<!-- Read by the pipeline: please leave this block as it is. -->",
    "```" + FENCE,
    JSON.stringify(data, null, 1),
    "```",
  ].join("\n");
}

export function issueTitle(list: Correction[]): string {
  const eras = [...new Set(list.map((c) => c.era))];
  return `Map corrections: ${list.length} lot${list.length === 1 ? "" : "s"} (${eras.join(", ")})`;
}

/** A link that opens a pre-filled GitHub issue, or null when it would be too long to open. */
export function issueUrl(list: Correction[]): string | null {
  if (!list.length) return null;
  const q = new URLSearchParams({ title: issueTitle(list), body: issueBody(list), labels: "correction" });
  const url = `https://github.com/${REPO}/issues/new?${q.toString()}`;
  return url.length <= MAX_URL ? url : null;
}

/** The same content as a file to attach to (or paste into) an issue. */
export function correctionsFile(list: Correction[]): string {
  return `# ${issueTitle(list)}\n\n${issueBody(list)}\n`;
}
