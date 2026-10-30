import styles from "./endmatter.module.css";

export const RUMSEY_CREDIT = "David Rumsey Map Collection, David Rumsey Map Center, Stanford Libraries";
/**
 * Who made this site, shown in the credits. If you run a modified version, replace it with the
 * attribution NOTICE section 1 requires of works based on this code:
 * "Based on sfzoning.com by Steven Bacio for GrowSF (https://growsf.org)."
 */
export const ATTRIBUTION = "sfzoning.com by Steven Bacio for GrowSF (https://growsf.org).";
/** Where users of this site can get its source code (AGPL section 13). */
export const SOURCE_URL = "https://github.com/Grow-SF/sfzoning";

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
        <li>
          <strong>About and license:</strong> {ATTRIBUTION} © 2026 GrowSF. Code:{" "}
          <a href="https://www.gnu.org/licenses/agpl-3.0.html">AGPL-3.0</a>, with the{" "}
          <a href={SOURCE_URL}>source on GitHub</a>. Content and data:{" "}
          <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>. Provided as is,
          without warranty, and not an official zoning record. GrowSF™ is a trademark of GrowSF.
        </li>
      </ul>
    </section>
  );
}
