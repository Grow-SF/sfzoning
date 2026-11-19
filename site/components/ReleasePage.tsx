// DRAFT COPY — NOT REVIEWED BY THE OWNER. Lede, banner, notes and button wording are first drafts.
import Link from "next/link";
import { Fragment } from "react";
import SiteFooter from "@/components/SiteFooter";
import { DATA_CREDIT, RUMSEY_CREDIT } from "@/lib/credits";
import { fileUrl, formatBytes, type Manifest, type Version } from "@/lib/downloadRows";
import historyJson from "@/lib/download-versions.json";
import manifestJson from "@/lib/downloads.json";
import { allReleases } from "@/lib/releaseData";
import type { Fmt, ReleaseFile } from "@/lib/releaseFiles";
import { releaseView, type Row } from "@/lib/releaseView";
import { zipNames, type Release } from "@/lib/releases.mjs";
import ReleaseFilter from "./ReleaseFilter";
import styles from "./releases.module.css";

const history = historyJson as unknown as { versions: Version[] };
const manifest = manifestJson as unknown as Manifest;

const FMT_LABEL: Record<Fmt, string> = { geojson: "GeoJSON", gpkg: "GeoPackage", shp: "Shapefile", cog: "GeoTIFF" };
const TITLE = "A History of San Francisco’s Zoning";

/** A citation with the work's title in italics (the text itself is plain). */
function Cited({ text }: { text: string }) {
  const at = text.indexOf(TITLE);
  if (at < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <i>{TITLE}</i>
      {text.slice(at + TITLE.length)}
    </>
  );
}

const fileCount = (n: number) => `${n} file${n === 1 ? "" : "s"}`;
const rowCount = (r: Row) => (r.type === "file" ? 1 : r.count);
const rowBytes = (r: Row) => (r.type === "file" ? r.file.bytes : r.bytes);

function FmtTag({ fmt }: { fmt: Fmt }) {
  return <span className={`${styles.fmt} ${styles[fmt]}`}>{FMT_LABEL[fmt]}</span>;
}

function HashCell({ sha256 }: { sha256: string }) {
  return (
    <td className={styles.hash}>
      <button type="button" className={styles.copy} data-copy={sha256} title={`Copy full SHA-256: ${sha256}`}>
        {sha256.slice(0, 8)}
      </button>
    </td>
  );
}

/** The cells every file and sheet shares: format, dated name, size, date, hash. */
function FileCells({ file, href }: { file: ReleaseFile; href: string }) {
  return (
    <>
      <td>
        <FmtTag fmt={file.fmt} />
      </td>
      <td className={styles.file}>
        <a href={href} title={file.name}>
          {file.name}
        </a>
      </td>
      <td className={styles.num}>{formatBytes(file.bytes)}</td>
      <td className={styles.date}>{file.date}</td>
      <HashCell sha256={file.sha256} />
    </>
  );
}

const dataAttrs = (fmt: Fmt, kind: string, era: string, bytes: number, n: number) => ({
  "data-fmt": fmt, "data-kind": kind, "data-era": era, "data-bytes": bytes, "data-n": n,
});

export default function ReleasePage({ release }: { release: Release }) {
  const view = releaseView(release, history, manifest);
  const ids = allReleases().map((r) => r.id).reverse();
  const newest = ids[0];
  const zips = release.zips;
  const zipBase = `/data/releases/${view.id}/`;
  const names = zipNames(view.id);
  return (
    <>
      <main className={styles.page}>
        {!view.latest && (
          <p className={styles.banner}>
            This is an older release, kept so that work citing it can find the same files. The latest is{" "}
            <a href="/data">release {newest}</a>.
          </p>
        )}
        <header className={styles.head}>
          <div>
            <p className={styles.kicker}>Open data</p>
            <h1 className={styles.title}>Download the data</h1>
            <p className={styles.lede}>
              San Francisco’s zoning from 1921 to today: today’s lots tagged with each historic district, the districts
              dissolved from them, and the map scans warped to today’s streets. A reconstruction read from scans, not an
              official record — <a href="/about">read how it was made</a>.
            </p>
            <div className={styles.cite}>
              <div className={styles.citebar}>
                <span className={styles.lbl}>Cite this release</span>
                <div className={styles.seg} role="group" aria-label="Citation style" data-seg>
                  {(["apa", "chicago", "bibtex"] as const).map((s) => (
                    <button key={s} type="button" data-style={s} aria-pressed={s === "chicago"}>
                      {s === "apa" ? "APA" : s === "chicago" ? "Chicago" : "BibTeX"}
                    </button>
                  ))}
                </div>
                <a href={`${zipBase}CITATION.cff`}>CITATION.cff</a>
              </div>
              <div className={styles.citebox}>
                <p className={styles.ref} data-ref="apa" data-text={view.citations.apa} hidden>
                  <Cited text={view.citations.apa} />
                </p>
                <p className={styles.ref} data-ref="chicago" data-text={view.citations.chicago}>
                  <Cited text={view.citations.chicago} />
                </p>
                <pre className={styles.ref} data-ref="bibtex" data-text={view.citations.bibtex} hidden>
                  {view.citations.bibtex}
                </pre>
                <button type="button" className={styles.copy} data-copy={view.citations.chicago} data-citecopy>
                  copy
                </button>
              </div>
            </div>
          </div>
          <aside className={styles.relcard}>
            <div className={styles.lbl}>Release</div>
            <div className={styles.reldate}>
              {/* Plain links, so picking a release never navigates by surprise and works without JavaScript. */}
              <details className={styles.picker}>
                <summary aria-label={`Release ${view.id}; choose another release`}>
                  {view.id}
                  <span className={styles.caretd} aria-hidden>
                    ▾
                  </span>
                </summary>
                <ul>
                  {ids.map((id) => (
                    <li key={id}>
                      <a href={id === newest ? "/data" : `/data/releases/${id}`} aria-current={id === view.id ? "page" : undefined}>
                        {id}
                        {id === newest && " (latest)"}
                      </a>
                    </li>
                  ))}
                </ul>
              </details>
              {view.latest && <span className={styles.tag}>Latest</span>}
            </div>
            <div className={styles.relstats}>
              {fileCount(view.totals.files)} · {formatBytes(view.totals.bytes)} · {view.totals.editions} editions · CC
              BY-NC-SA 4.0
            </div>
            <div className={styles.rellinks}>
              {view.previousId && (
                <a href={`/data/releases/${view.id}/changes`}>Changes since {view.previousId}</a>
              )}
              <Link href="/data/releases">All releases</Link>
              <a href={`${zipBase}manifest.json`}>manifest.json</a>
            </div>
          </aside>
        </header>

        <ReleaseFilter
          id={view.id}
          files={view.totals.files}
          bytes={view.totals.bytes}
          editions={view.editions.map((e) => ({ era: e.era, label: e.label }))}
          zips={{
            all: { href: zipBase + names.all, bytes: zips.all.bytes },
            geojson: { href: zipBase + names.geojson, bytes: zips.geojson.bytes },
            gpkg: { href: zipBase + names.gpkg, bytes: zips.gpkg.bytes },
            shp: { href: zipBase + names.shp, bytes: zips.shp.bytes },
          }}
        />

        <nav className={styles.jump} aria-label="Editions">
          <span className={styles.lbl}>Jump to</span>
          {view.editions.map((e) => (
            <a key={e.era} href={`#e-${e.era}`} data-jump={e.era}>
              {e.label}
            </a>
          ))}
        </nav>

        <div className={styles.wrap}>
          <table className={styles.table}>
            <colgroup>
              <col className={styles.cl} />
              <col className={styles.cf} />
              <col />
              <col className={styles.cs} />
              <col className={`${styles.cd} ${styles.date}`} />
              <col className={`${styles.ch} ${styles.hash}`} />
              <col className={styles.ca} />
            </colgroup>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Format</th>
                <th>File · dated name, never changes</th>
                <th className={styles.num}>Size</th>
                <th className={styles.date}>Updated</th>
                <th className={styles.hash}>SHA-256</th>
                <th aria-label="Links" />
              </tr>
            </thead>
            {view.editions.map((e) => {
              let prev = "";
              const n = e.rows.reduce((s, r) => s + rowCount(r), 0);
              const b = e.rows.reduce((s, r) => s + rowBytes(r), 0);
              return (
                <tbody key={e.era} data-edition={e.era}>
                  <tr className={styles.era}>
                    <th colSpan={7} id={`e-${e.era}`}>
                      <span className={styles.eral}>{e.label}</span>
                      <span className={styles.eram} data-eram>
                        {fileCount(n)} · {formatBytes(b)}
                      </span>
                    </th>
                  </tr>
                  {e.rows.map((r) => {
                    if (r.type === "file") {
                      const f = r.file;
                      const shown = r.label === prev ? "" : r.label;
                      prev = r.label;
                      return (
                        <tr key={f.name} data-row data-ln={r.label} {...dataAttrs(f.fmt, f.kind, f.era, f.bytes, 1)}>
                          <td className={styles.layer}>{shown}</td>
                          <FileCells file={f} href={fileUrl(f.series, f.name)} />
                          <td className={styles.acts}>
                            <a className={styles.lat} href={r.latestUrl}>
                              ↓ latest
                            </a>
                            <a href={r.historyUrl}>history</a>
                          </td>
                        </tr>
                      );
                    }
                    prev = r.label;
                    const g = `g-${r.series}`;
                    return (
                      <Fragment key={g}>
                        <tr data-row data-group={g} data-ln={r.label} {...dataAttrs("cog", "sheets", r.era, r.bytes, r.count)}>
                          <td className={styles.layer}>{r.label}</td>
                          <td>
                            <FmtTag fmt="cog" />
                          </td>
                          <td className={styles.file}>
                            <button type="button" className={styles.tog} aria-expanded={false} data-tog={g}>
                              <span className={styles.caret}>▸</span> {r.count} sheets
                            </button>
                            {r.zipUrl && (
                              <a className={styles.zipl} href={r.zipUrl}>
                                all as zip
                              </a>
                            )}
                            <a className={styles.zipl} href={r.listUrl}>
                              {r.zipUrl ? "sheet list (CSV)" : "sheet list (urls.txt)"}
                            </a>
                          </td>
                          <td className={styles.num}>{formatBytes(r.bytes)}</td>
                          <td className={styles.date}>{r.date}</td>
                          <td className={styles.hash} />
                          <td className={styles.acts} />
                        </tr>
                        {r.sheets.map((s) => (
                          <tr key={s.name} className={`${styles.child} ${styles.shut}`} data-child={g}>
                            <td className={styles.layer}>{s.label}</td>
                            <FileCells file={s} href={fileUrl(s.series, s.name)} />
                            <td className={styles.acts}>
                              <a className={styles.lat} href={s.latestUrl}>
                                ↓ latest
                              </a>
                              <a href={s.historyUrl}>history</a>
                            </td>
                          </tr>
                        ))}
                      </Fragment>
                    );
                  })}
                </tbody>
              );
            })}
          </table>
          {/* Without JavaScript the sheet rows cannot be opened, so show them all. */}
          <noscript>
            <style>{`.${styles.shut}{display:table-row}`}</style>
          </noscript>
        </div>
        <p className={styles.empty} data-empty hidden>
          Nothing matches. Turn a filter back on.
        </p>

        <section className={styles.notes}>
          <div>
            <h3>Formats</h3>
            <p>
              Lots and districts as GeoJSON, GeoPackage and Shapefile in WGS84 (EPSG:4326); the GeoPackage also holds
              California State Plane III (feet). Each zip carries a README and LICENSE. Scans are Cloud-Optimized
              GeoTIFFs: tinted, border rules and binding shadows erased, nothing traced.
            </p>
          </div>
          <div>
            <h3>Links</h3>
            <p>
              A file name is its dated link and always returns exactly that file. <b>Latest</b> always returns the
              newest version; <b>history</b> lists every version. Single sheets carry no README; the sheet list gives
              each one’s license and credit.
            </p>
          </div>
          <div>
            <h3>License and credit</h3>
            <p>
              <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>. Credit: “{DATA_CREDIT}”.
              Lot lines from DataSF parcels. 1948 images from the {RUMSEY_CREDIT},{" "}
              <a href="https://creativecommons.org/licenses/by-nc-sa/3.0/">CC BY-NC-SA 3.0</a>.
            </p>
          </div>
        </section>
      </main>
      <SiteFooter />
    </>
  );
}
