import type { Step } from "@/content/schema";
import type { Era } from "@/lib/eras";
import { SOURCES } from "@/content/sources";
import styles from "@/app/page.module.css";

type Props = { step: Step; era?: Era; active: boolean };

export default function StepCard({ step, era, active }: Props) {
  return (
    <article className={styles.card} data-active={active || undefined} aria-labelledby={`${step.id}-title`}>
      {/* The scroller carries no background: Chrome culls a WebGL canvas under an opaque
          composited scroller over the scroller's whole content height, blanking the map. */}
      <div className={styles.cardScroll} tabIndex={0}>
      <p className={styles.year}>{era?.label ?? step.era}</p>
      <h2 id={`${step.id}-title`} className={styles.title}>
        {step.title}
      </h2>
      {step.body.map((b, i) => {
        const text = b.href ? <a href={b.href}>{b.text}</a> : b.text;
        if (b.kind === "quote")
          return (
            <blockquote key={i} className={styles.quote}>
              {text}
            </blockquote>
          );
        if (b.kind === "stat")
          return (
            <p key={i} className={styles.stat}>
              {text}
            </p>
          );
        return (
          <p key={i} className={styles.para}>
            {text}
          </p>
        );
      })}
      <footer className={styles.sources}>
        <span className={styles.sourcesLabel}>{step.sources.length > 1 ? "Sources" : "Source"}</span>
        <ul>
          {step.sources.map((id) => {
            const src = SOURCES[id];
            if (!src) return <li key={id}>{id}</li>;
            return (
              <li key={id} data-source={id}>
                {src.url ? (
                  <a href={src.url} target="_blank" rel="noreferrer">
                    {src.short}
                  </a>
                ) : (
                  <a href={`/about#src-${id}`}>{src.short}</a>
                )}
              </li>
            );
          })}
        </ul>
      </footer>
      </div>
    </article>
  );
}
