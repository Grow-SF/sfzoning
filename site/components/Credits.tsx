import styles from "./endmatter.module.css";

export const RUMSEY_CREDIT = "David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries";

export default function Credits() {
  return (
    <section className={styles.block} aria-labelledby="credits-heading">
      <h2 id="credits-heading" className={styles.heading}>
        Credits
      </h2>
      <ul className={styles.credits}>
        <li>
          <strong>1948 zoning atlas:</strong> {RUMSEY_CREDIT}, licensed{" "}
          <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>.
        </li>
        <li>
          <strong>1921–2000 zoning maps:</strong> scanned from the San Francisco Planning Department’s map books on
          its large-format scanner; the 1921 ordinance is also held by the San Francisco Public Library (call no.
          917.946 Sa52b 1921).
        </li>
        <li>
          <strong>Ordinances, codes and reports:</strong> many are digitised by the San Francisco Public Library on the
          Internet Archive.
        </li>
        <li>
          <strong>Base map:</strong> today’s blocks, built from DataSF parcels and street centerlines.
        </li>
        <li>
          <strong>Map alignment:</strong> sfzmaps, this project’s own georeferencing pipeline. See{" "}
          <a href="/about">how the maps were made</a>.
        </li>
      </ul>
    </section>
  );
}
