"use client";

import { useId, useState } from "react";
import type { Era } from "@/lib/eras";
import type { Rule } from "@/lib/rules";
import { addCorrection, codesForYear, useCorrections } from "@/lib/corrections";
import type { ParcelHit } from "./StoryMap";
import styles from "./explorer.module.css";

type Props = { hit: ParcelHit; era: Era; rules: Rule[] };

/** On the lot card: "the old map prints something else here", added to the reader's list. */
export default function CorrectionForm({ hit, era, rules }: Props) {
  const list = useCorrections();
  const [open, setOpen] = useState(false);
  const [code, setCode] = useState("");
  const [note, setNote] = useState("");
  const [added, setAdded] = useState(false);
  const id = useId();
  if (!hit.blklot || !era.districtSeries) return null;
  const queued = list.find((c) => c.era === era.id && c.blklot === hit.blklot);
  const codes = codesForYear(rules, era.year);
  const proposed = code.trim().toUpperCase();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!proposed) return;
    addCorrection({
      era: era.id,
      series: era.districtSeries!,
      blklot: hit.blklot!,
      shown: hit.code || "",
      proposed,
      note: note.trim(),
    });
    setAdded(true);
    setOpen(false);
  };

  if (!open) {
    return (
      <p className={styles.correctLine}>
        {added || queued ? (
          <span data-testid="correction-added">
            In your corrections: {queued?.proposed ?? proposed}.{" "}
          </span>
        ) : null}
        <button type="button" className={styles.linkButton} onClick={() => setOpen(true)} data-testid="suggest-correction">
          {queued ? "Change your correction" : "Does the old map show something else? Suggest a correction"}
        </button>
      </p>
    );
  }

  return (
    <form className={styles.correctForm} onSubmit={submit} data-testid="correction-form">
      <p className={styles.correctHelp}>
        What does the {era.label} map print for this lot? Check the original map (switch to it above), not
        today&rsquo;s zoning.
      </p>
      <label htmlFor={`${id}-code`}>District on the {era.label} map</label>
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
        <button type="button" className={styles.linkButton} onClick={() => setOpen(false)}>
          Cancel
        </button>
      </div>
    </form>
  );
}
