// DRAFT COPY — NOT REVIEWED BY THE OWNER. Methods text summarises pipeline/README.md,
// docs/superpowers/specs/2026-10-29-map-georeferencing-design.md and the final results
// table in docs/superpowers/notes/2026-10-29-tuning.md (full run, 2026-10-29).
import type { Metadata } from "next";
import Link from "next/link";
import Credits, { SOURCE_URL } from "@/components/Credits";
import DraftBanner from "@/components/DraftBanner";
import Sources from "@/components/Sources";
import { SOURCES, citedSourceIds } from "@/content/sources";
import { STORY } from "@/content/story";
import { ERAS } from "@/lib/eras";
import styles from "@/components/endmatter.module.css";

export const metadata: Metadata = {
  title: "How the maps were made · How San Francisco was zoned",
  description: "How a century of San Francisco zoning maps were aligned to today’s streets, how accurate they are, and the sources behind the story.",
  alternates: { canonical: "/about" },
  openGraph: {
    title: "How the maps were made",
    description: "How a century of San Francisco zoning maps were aligned to today’s streets, how accurate they are, and the sources behind the story.",
    url: "/about",
    // A segment's openGraph replaces the root's, so name the share image again.
    images: [{ url: "/opengraph-image.jpg", width: 1200, height: 630, alt: "Map of San Francisco with every lot coloured by how many homes today's zoning allows." }],
  },
};

// Every source the story and eras cite, then the rest of the registry.
const ALL_IDS = Array.from(new Set([...citedSourceIds(STORY, ERAS), ...Object.keys(SOURCES)]));

export default function About() {
  return (
    <main className={styles.page}>
      <DraftBanner />
      <p className={styles.kicker}>About this story</p>
      <h1 className={styles.title}>How the maps were made</h1>
      <p className={styles.lede}>
        Every map in this story is a page from one of San Francisco’s own zoning map books, 1921 to 2000, laid over
        today’s blocks so you can compare the same street across a century.
      </p>

      <p className={styles.status} data-testid="review-status">
        <strong>Status: draft.</strong> The story text has not yet been reviewed. Map alignment is automatic; sheets
        that miss the accuracy bar below are waiting for a person to check them. The “what it allowed” colours are
        published for the eras whose maps have been read so far; some eras are still pending.
      </p>

      <section className={styles.block}>
        <h2 className={styles.heading}>Where the maps come from</h2>
        <p>
          Most of the maps were scanned from the Planning Department’s map books on its large-format scanner. They come
          in series, one for each edition of the map book: 1921, 1929, 1935, 1948, 1960, 1964, 1968, 1972, 1975, 1980,
          and the zoning, height and special-use maps of 1980–90, 1990–96, 1996–2000 and 2000. The 1948 atlas images
          come from the David Rumsey Map Collection.
        </p>
      </section>

      <section className={styles.block}>
        <h2 className={styles.heading}>How they were lined up with today’s streets</h2>
        <p>
          Old map sheets don’t sit on a modern grid. Paper stretches, and every street was drawn by hand. Aligning a
          sheet by hand means clicking matching street corners on the scan and on a modern map, one at a time. We wrote
          a program to do it instead.
        </p>
        <p>
          <strong>First, find the sheet.</strong> An AI vision model reads the printed block numbers and street names
          on each page. The program matches them to today’s blocks and intersections from DataSF and places the sheet
          roughly, throwing out readings that disagree with the rest.
        </p>
        <p>
          <strong>Then, fit it closely.</strong> It traces the drawn street and block lines and matches hundreds to
          thousands of block corners to today’s block outlines, not the eight or so a person would click. It bends the
          sheet with a smooth warp that corrects stretched paper and drafting error over whole areas.
        </p>
        <p>
          <strong>Keep the history.</strong> Some blocks really have changed: streets were widened, shorelines filled,
          neighbourhoods like the Western Addition rebuilt. Where an old edge is consistently off from today’s in a
          way drafting error can’t explain, the program marks it as real change and leaves it where the old map put
          it, rather than pulling it onto the modern line.
        </p>
        <p>
          <strong>Finally, stitch and tile.</strong> Sheets are joined into one layer per series, the white paper is
          made transparent so today’s blocks show through, and the result is cut into web map tiles.
        </p>
      </section>

      <section className={styles.block}>
        <h2 className={styles.heading}>How accurate they are</h2>
        <p>
          Each sheet is tested on blocks it was not fitted to: the program leaves some blocks out, fits the rest, and
          measures how far the left-out corners land from today’s. The hand-drafted maps themselves are only good to
          about ±4 feet at a street corner, so that is roughly the floor.
        </p>
        <p>
          A sheet passes when half its corners land within 7 feet and 95% within 15 feet. Coarser scans get a
          proportionally looser bar: the 1980-and-later maps are office-copier scans at about 4 feet per pixel, and
          pass at about 10 and 22 feet. A sheet also fails if it looks misplaced or the fit didn’t settle.
        </p>
        <ul className={styles.figures} aria-label="Alignment results">
          <li>
            <strong>369</strong>map sheets
          </li>
          <li>
            <strong>334</strong>passed automatically
          </li>
          <li>
            <strong>35</strong>flagged for review
          </li>
          <li>
            <strong>4–6 ft</strong>typical median error
          </li>
        </ul>
        <p>
          Most map books reuse the same base sheets from edition to edition, so a sheet that can&rsquo;t be placed
          from its own labels is matched against the same sheet in another edition that was placed, and checked
          against the same bar. Of the 35 sheets still flagged, 26 couldn&rsquo;t be placed at all; they are missing
          from the maps until someone places them by hand. The rest appear, but may be off by more than the bar.
        </p>
        <p>
          The colors showing what each district allowed don&rsquo;t depend on this precision: each lot is colored by
          the district the historic map shows for it, using today&rsquo;s lot lines.
        </p>
      </section>

      <section className={styles.block} aria-labelledby="reuse-heading">
        <h2 id="reuse-heading" className={styles.heading}>
          Source code and reuse
        </h2>
        <p>
          The code for the map pipeline and this site is open source, licensed under the{" "}
          <a href="https://www.gnu.org/licenses/agpl-3.0.html">GNU AGPL v3.0</a> and published{" "}
          <a href={SOURCE_URL}>on GitHub</a>. The pipeline can align other cities&rsquo; historic maps too.
        </p>
        <p>
          The story, the research and the processed maps are licensed{" "}
          <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>: you can reuse and adapt
          them with credit, for non-commercial purposes, under the same license. That is not an open source license.
          The original zoning maps are, to the best of our knowledge, in the public domain.
        </p>
        <p>Corrections, fixes and adaptations are welcome; the repository explains how to contribute.</p>
      </section>

      <Sources ids={ALL_IDS} />
      <Credits />

      <p className={styles.block}>
        <Link className={styles.back} href="/">
          ← Back to the story
        </Link>
      </p>
    </main>
  );
}
