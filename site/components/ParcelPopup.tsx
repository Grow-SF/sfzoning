"use client";

import { useEffect, useRef } from "react";
import type { Era } from "@/lib/eras";
import { CATEGORY_COLORS, CATEGORY_LABELS, isCategory } from "@/lib/palette";
import { confidenceLabel, lookupRule, type Rule } from "@/lib/rules";
import type { ParcelHit } from "./StoryMap";
import styles from "./explorer.module.css";

type Props = { hit: ParcelHit; era: Era; rules: Rule[]; onClose: () => void };

function formatBlklot(b?: string) {
  if (!b) return null;
  return b.length >= 7 ? `Block ${b.slice(0, 4)}, lot ${b.slice(4)}` : `Lot ${b}`;
}

export default function ParcelPopup({ hit, era, rules, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const rule = lookupRule(rules, era.year, hit.code);
  const cat = isCategory(hit.category) ? hit.category : "unknown";

  useEffect(() => {
    closeRef.current?.focus({ preventScroll: true });
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [hit, onClose]);

  return (
    <aside className={styles.popup} role="dialog" aria-labelledby="parcel-code" data-testid="parcel-popup">
      <button ref={closeRef} type="button" className={styles.close} onClick={onClose} aria-label="Close lot details">
        ×
      </button>
      {/* The scroller carries no background (see StepCard): Chrome culls a WebGL canvas under an
          opaque composited scroller over the scroller's whole content height. */}
      <div className={styles.popupScroll}>
        <p className={styles.popupEra}>
          {era.label}
          {formatBlklot(hit.blklot) && <span className={styles.popupLot}>{formatBlklot(hit.blklot)}</span>}
        </p>
        <h3 id="parcel-code" className={styles.popupCode}>
          {hit.code || "No district read"}
        </h3>
        {rule?.name && <p className={styles.popupName}>{rule.name}</p>}
        <p className={styles.popupCat}>
          <span className={styles.popupSwatch} style={{ background: CATEGORY_COLORS[cat] }} aria-hidden />
          {CATEGORY_LABELS[cat]}
        </p>

        {rule ? (
          <dl className={styles.facts}>
            <dt>Homes per lot</dt>
            <dd>{rule.max_units_per_lot || "Not stated"}</dd>
            <dt>Height limit</dt>
            <dd>
              {rule.height_limit_ft || "Not stated"}
              {hit.height_code ? ` (height district ${hit.height_code})` : ""}
            </dd>
            <dt>What could be built</dt>
            <dd className={styles.uses}>{rule.allowed_uses || "Not stated"}</dd>
          </dl>
        ) : (
          <p className={styles.noRule}>
            {hit.code
              ? `We haven’t written a plain-language summary of ${hit.code} for ${era.label} yet.`
              : "The map reader couldn’t find a district label for this lot."}
          </p>
        )}

        <p className={styles.sure}>
          <span>Map reading: {confidenceLabel(hit.confidence)}</span>
          {rule && <span>Rule summary: {rule.confidence[0].toUpperCase() + rule.confidence.slice(1)}</span>}
        </p>
      </div>
    </aside>
  );
}
