import { CATEGORY_COLORS, CATEGORY_LABELS, HOMES_RAMP, OTHER_USES } from "@/lib/palette";
import styles from "./legend.module.css";

const RAMP_TICKS: Record<string, string> = {
  single_family: "1",
  two_unit: "2",
  three_unit: "3",
  multi_family: "Apts",
};

/** Key to the "what it allowed" colours: a homes-per-lot ramp, then the other uses. */
export default function Legend() {
  return (
    <figure className={styles.legend} aria-label="Map key" data-testid="legend">
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
  );
}
