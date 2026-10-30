"use client";

import { useId, useState } from "react";
import { CATEGORY_COLORS, CATEGORY_LABELS, HOMES_RAMP, OTHER_USES } from "@/lib/palette";
import styles from "./legend.module.css";

const RAMP_TICKS: Record<string, string> = {
  single_family: "1",
  two_unit: "2",
  three_unit: "3",
  multi_family: "Apts",
};

type Props = {
  /**
   * Story map: on phones the key folds into a small chip (a strip of the colours) that opens
   * the full key on tap, so it doesn't sit over a third of the map. Desktop and the explorer
   * always show the full key.
   */
  compact?: boolean;
};

/** Key to the "what it allowed" colours: a homes-per-lot ramp, then the other uses. */
export default function Legend({ compact = false }: Props) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={styles.wrap} data-compact={compact || undefined} data-open={open || undefined} data-testid="legend">
      {compact && (
        <button
          type="button"
          className={styles.chip}
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen((o) => !o)}
        >
          <span className={styles.strip} aria-hidden>
            {[...HOMES_RAMP, ...OTHER_USES.filter((c) => c !== "unknown")].map((c) => (
              <span key={c} style={{ background: CATEGORY_COLORS[c] }} />
            ))}
          </span>
          {open ? "Hide key" : "Key"}
        </button>
      )}
      <figure id={id} className={styles.legend} aria-label="Map key">
        <figcaption className={styles.head}>Homes allowed per lot</figcaption>
        <ol className={styles.ramp}>
          {HOMES_RAMP.map((c) => (
            <li key={c} title={CATEGORY_LABELS[c]}>
              <span className={styles.bar} style={{ background: CATEGORY_COLORS[c] }} aria-hidden />
              <span className={styles.tick} aria-hidden>
                {RAMP_TICKS[c]}
              </span>
              <span className={styles.sr}>{CATEGORY_LABELS[c]}</span>
            </li>
          ))}
        </ol>
        <ul className={styles.uses}>
          {OTHER_USES.map((c) => (
            <li key={c} data-category={c}>
              <span
                className={c === "unknown" ? `${styles.swatch} ${styles.unknown}` : styles.swatch}
                style={{ background: CATEGORY_COLORS[c] }}
                aria-hidden
              />
              {CATEGORY_LABELS[c]}
            </li>
          ))}
        </ul>
      </figure>
    </div>
  );
}
