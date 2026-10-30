"use client";

import { createRef, useEffect, useMemo, useRef } from "react";
import StoryMap, { type StoryMapHandle } from "./StoryMap";
import StepCard from "./StepCard";
import type { Step } from "@/content/schema";
import { ERAS } from "@/lib/eras";
import { useScrollSteps } from "@/lib/useScrollSteps";
import styles from "@/app/page.module.css";

type Props = { steps: Step[] };

export default function Story({ steps }: Props) {
  const mapRef = useRef<StoryMapHandle>(null);
  const refs = useMemo(() => steps.map(() => createRef<HTMLElement>()), [steps]);
  const { active, progress } = useScrollSteps(refs);
  const step = steps[active];
  const era = ERAS.find((e) => e.id === step.era);
  const classificationPending = step.view !== "scan" && !era?.districtSeries;

  useEffect(() => {
    mapRef.current?.apply({ era: step.era, view: step.view, camera: step.camera, highlight: step.highlight });
  }, [step]);

  const first = steps[0];
  return (
    <div className={styles.story} data-active-step={step.id}>
      <div className={styles.mapPane}>
        <StoryMap
          ref={mapRef}
          initial={{ era: first.era, view: first.view, camera: first.camera, highlight: first.highlight }}
        />
        <ol className={styles.ruler} aria-label="Eras">
          {ERAS.map((e) => (
            <li key={e.id} data-on={e.id === step.era || undefined} aria-current={e.id === step.era ? "step" : undefined}>
              <span>{e.label}</span>
            </li>
          ))}
        </ol>
        {classificationPending && (
          <p className={styles.pending}>District colours for {era?.label ?? step.era} are coming. This is the original map.</p>
        )}
      </div>

      <div className={styles.column}>
        <header className={styles.hero}>
          <h1 className={styles.headline}>How San Francisco was zoned</h1>
          <p className={styles.dek}>
            A century of rules about what could be built where, read straight off the city&rsquo;s own maps.
          </p>
          <p className={styles.cue}>Scroll to begin in 1921</p>
        </header>

        {steps.map((s, i) => (
          <section
            key={s.id}
            ref={refs[i]}
            className={styles.step}
            data-step={s.id}
            data-active={i === active || undefined}
            style={i === active ? ({ "--progress": progress } as React.CSSProperties) : undefined}
          >
            <StepCard step={s} era={ERAS.find((e) => e.id === s.era)} active={i === active} />
          </section>
        ))}
        <div className={styles.tail} aria-hidden />
      </div>
    </div>
  );
}
