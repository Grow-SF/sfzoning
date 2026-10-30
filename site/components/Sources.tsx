import { SOURCES } from "@/content/sources";
import styles from "./endmatter.module.css";

type Props = { ids: string[]; heading?: string };

/** Numbered source list, primary sources first. Each item has an #src-<id> anchor. */
export default function Sources({ ids, heading = "Sources" }: Props) {
  const known = ids.filter((id) => SOURCES[id]);
  const groups = [
    { kind: "primary" as const, label: "Primary: ordinances, codes, maps and city reports" },
    { kind: "secondary" as const, label: "Secondary: histories, summaries and press" },
  ];
  return (
    <section className={styles.block} aria-labelledby="sources-heading">
      <h2 id="sources-heading" className={styles.heading}>
        {heading}
      </h2>
      {groups.map((g) => {
        const items = known.filter((id) => SOURCES[id].kind === g.kind);
        if (!items.length) return null;
        return (
          <div key={g.kind}>
            <h3 className={styles.subheading}>{g.label}</h3>
            <ol className={styles.sourceList}>
              {items.map((id) => {
                const s = SOURCES[id];
                return (
                  <li key={id} id={`src-${id}`} data-source={id}>
                    {s.url ? (
                      <a href={s.url} target="_blank" rel="noreferrer">
                        {s.title}
                      </a>
                    ) : (
                      s.title
                    )}
                    {s.where && <span className={styles.where}> {s.where}.</span>}
                    {s.note && <span className={styles.where}> {s.note}</span>}
                  </li>
                );
              })}
            </ol>
          </div>
        );
      })}
    </section>
  );
}
