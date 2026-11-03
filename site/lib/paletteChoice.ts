import { useSyncExternalStore } from "react";
import { PALETTES, type Category, type PaletteName } from "./palette";

// The reader's palette: standard, or the colour-blind one. One choice for every map on the page,
// remembered between visits.

const KEY = "sfz-palette";
const listeners = new Set<() => void>();
let current: PaletteName | null = null;

function read(): PaletteName {
  if (current) return current;
  try {
    current = window.localStorage.getItem(KEY) === "accessible" ? "accessible" : "standard";
  } catch {
    current = "standard";
  }
  return current;
}

export function getPaletteName(): PaletteName {
  return typeof window === "undefined" ? "standard" : read();
}

export function setPaletteName(name: PaletteName) {
  current = name;
  try {
    window.localStorage.setItem(KEY, name);
  } catch {
    // Private windows and blocked storage: the choice lasts for this page only.
  }
  listeners.forEach((l) => l());
}

export function subscribePalette(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The colours to draw with, following the reader's choice. */
export function usePalette(): { name: PaletteName; colors: Record<Category, string> } {
  const name = useSyncExternalStore(subscribePalette, getPaletteName, () => "standard" as const);
  return { name, colors: PALETTES[name] };
}

/** For tests: forget the cached choice so the next read goes to storage. */
export function resetPaletteChoice() {
  current = null;
}
