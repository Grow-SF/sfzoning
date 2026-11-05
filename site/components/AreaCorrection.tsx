"use client";

import { useId, useState } from "react";
import type { Era } from "@/lib/eras";
import type { Rule } from "@/lib/rules";
import { addCorrection, areaKey, codesForYear, roundRing } from "@/lib/corrections";
import styles from "./explorer.module.css";

type Props = {
  era: Era;
  rules: Rule[];
  drawing: boolean;
  ring: [number, number][] | null;
  onStart: () => void;
  onFinish: () => void;
  onDone: () => void;
};

/**
 * Under the explorer map: draw an area where the old map's district differs from what the site
 * shows (a district line through the middle of a lot, or several lots at once), then say what
 * the old map prints there. Becomes an area correction in the reader's list.
 */
export default function AreaCorrection({ era, rules, drawing, ring, onStart, onFinish, onDone }: Props) {
  const [code, setCode] = useState("");
  const [note, setNote] = useState("");
  const id = useId();
  if (!era.districtSeries) return null;
  const codes = codesForYear(rules, era.year);
  const proposed = code.trim().toUpperCase();

  if (!drawing && !ring) {
    return (
      <p className={styles.correctLine}>
        <button type="button" className={styles.linkButton} onClick={onStart} data-testid="draw-area">
          Is a district line in the wrong place? Draw the area the {era.label} map shows differently
        </button>
      </p>
    );
  }
  if (drawing && !ring) {
    return (
      <p className={styles.correctLine} data-testid="drawing-help">
        Click the map to place each corner of the area, following the old map&rsquo;s district line (it may cut
        through a lot). Double-click the last corner, or{" "}
        <button type="button" className={styles.linkButton} onClick={onFinish}>
          finish
        </button>
        .{" "}
        <button type="button" className={styles.linkButton} onClick={onDone}>
          Cancel
        </button>
      </p>
    );
  }
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!proposed || !ring) return;
    addCorrection({
      era: era.id,
      series: era.districtSeries!,
      blklot: areaKey(),
      shown: "",
      proposed,
      note: note.trim(),
      area: roundRing(ring),
    });
    setCode("");
    setNote("");
    onDone();
  };
  return (
    <form className={styles.correctForm} onSubmit={submit} data-testid="area-form">
      <p className={styles.correctHelp}>
        What district does the {era.label} map print inside the area you drew? Check the original map, not today&rsquo;s
        zoning.
      </p>
      <label htmlFor={`${id}-code`}>District inside the area</label>
      <input
        id={`${id}-code`}
        list={`${id}-codes`}
        value={code}
        onChange={(e) => setCode(e.target.value)}
        placeholder={codes[0] ? `e.g. ${codes[0]}` : "e.g. RH-1"}
        autoComplete="off"
        required
      />
      <datalist id={`${id}-codes`}>
        {codes.map((c) => (
          <option key={c} value={c} />
        ))}
      </datalist>
      <label htmlFor={`${id}-note`}>Note (optional)</label>
      <input id={`${id}-note`} value={note} onChange={(e) => setNote(e.target.value)} maxLength={200} />
      <div className={styles.correctButtons}>
        <button type="submit" disabled={!proposed}>
          Add to my corrections
        </button>
        <button type="button" className={styles.linkButton} onClick={onDone}>
          Cancel
        </button>
      </div>
    </form>
  );
}
