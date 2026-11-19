"use client";
// DRAFT COPY — NOT REVIEWED BY THE OWNER. Panel labels and the zip note are first drafts.
//
// The filter and download panel, and the page's small behaviours (citation switch, sheet
// expanders, copy buttons). It renders only the panel; the server-rendered
// table is changed in place through its data- attributes, so React never re-renders the rows.
// No download is built here: links are plain hrefs to server routes.
import { useEffect, useRef } from "react";
import { formatBytes } from "@/lib/downloadRows";
import type { Fmt, Kind } from "@/lib/releaseFiles";
import { chipFilter, filterQuery, formatZip, matches, parseFilter, type Filter } from "@/lib/releaseFilter";
import styles from "./releases.module.css";

type Zip = { href: string; bytes: number };
type Props = {
  id: string;
  files: number;
  bytes: number;
  editions: { era: string; label: string }[];
  zips: { all: Zip; geojson: Zip; gpkg: Zip; shp: Zip };
};

const FORMATS: [Fmt, string][] = [["geojson", "GeoJSON"], ["gpkg", "GeoPackage"], ["shp", "Shapefile"], ["cog", "GeoTIFF"]];
const LAYERS: [Kind, string][] = [["lots", "Lots"], ["districts", "Districts"], ["scans", "Citywide scans"], ["sheets", "Scan sheets"]];
const ZIP_NAME = { geojson: "GeoJSON files", gpkg: "GeoPackages", shp: "Shapefiles" };
const SITE = "https://sfzoning.com";

const fileCount = (n: number) => `${n} file${n === 1 ? "" : "s"}`;
const command = (urls: string) => `curl -s '${urls}' | wget -nc -i -`;

function Chips({ name, options }: { name: string; options: [string, string][] }) {
  return options.map(([v, label]) => (
    <label key={v} className={styles.chip}>
      <input type="checkbox" name={name} value={v} defaultChecked />
      <span>{label}</span>
    </label>
  ));
}

export default function ReleaseFilter({ id, files, bytes, editions, zips }: Props) {
  const form = useRef<HTMLFormElement>(null);
  const base = `/data/releases/${id}/`;

  useEffect(() => {
    const el = form.current;
    if (!el) return;
    const $ = <T extends Element>(sel: string) => el.querySelector<T>(sel)!;
    const rows = [...document.querySelectorAll<HTMLTableRowElement>("tr[data-row]")];
    const bodies = [...document.querySelectorAll<HTMLTableSectionElement>("tbody[data-edition]")];

    const present = editions.map((e) => e.era);
    const read = (): Filter => {
      const fd = new FormData(el);
      const all = (k: string) => fd.getAll(k) as string[];
      return chipFilter({ format: all("format"), layer: all("layer"), edition: all("edition") }, present);
    };

    const update = () => {
      const f = read();
      let n = 0;
      let b = 0;
      for (const r of rows) {
        const d = r.dataset;
        const on = matches({ fmt: d.fmt as Fmt, kind: d.kind as Kind, era: d.era! }, f);
        r.hidden = !on;
        if (on) {
          n += Number(d.n);
          b += Number(d.bytes);
        }
        if (d.group) document.querySelectorAll<HTMLElement>(`tr[data-child="${d.group}"]`).forEach((c) => (c.hidden = !on));
      }
      let any = false;
      for (const t of bodies) {
        const shown = rows.filter((r) => !r.hidden && t.contains(r));
        t.hidden = shown.length === 0;
        any ||= !t.hidden;
        const jump = document.querySelector<HTMLElement>(`a[data-jump="${t.dataset.edition}"]`);
        if (jump) jump.hidden = t.hidden;
        let en = 0;
        let eb = 0;
        let prev = "";
        for (const r of shown) {
          en += Number(r.dataset.n);
          eb += Number(r.dataset.bytes);
          // Name the layer once per run of the same layer among the visible rows.
          const ln = r.dataset.ln ?? "";
          r.cells[0].textContent = ln === prev && !r.dataset.group ? "" : ln;
          prev = ln;
        }
        const m = t.querySelector("[data-eram]");
        if (m) m.textContent = `${fileCount(en)} · ${formatBytes(eb)}`;
      }
      document.querySelector<HTMLElement>("[data-empty]")!.hidden = any;

      const q = filterQuery(f);
      const none = n === 0;
      $<HTMLElement>("[data-count]").firstChild!.textContent = `${fileCount(n)} · ${formatBytes(b)}`;
      $<HTMLElement>("[data-count] small").textContent = q ? "Your selection" : "Everything in this release";

      const fz = formatZip(f);
      const fzLink = $<HTMLAnchorElement>("[data-fmtzip]");
      fzLink.hidden = !fz;
      $<HTMLElement>("[data-zipnote]").hidden = !!fz;
      if (fz) {
        fzLink.href = zips[fz].href;
        fzLink.textContent = `↓ Download all ${ZIP_NAME[fz]} as zip (${formatBytes(zips[fz].bytes)})`;
      }

      for (const [sel, file] of [["[data-urls]", "urls.txt"], ["[data-sums]", "SHA256SUMS"]] as const) {
        const a = $<HTMLAnchorElement>(sel);
        a.href = base + file + q;
        a.setAttribute("aria-disabled", String(none));
      }
      const cmd = none ? "Nothing selected" : command(SITE + base + "urls.txt" + q);
      $<HTMLElement>("[data-cmd]").textContent = cmd;
      $<HTMLElement>("[data-cmdcopy]").dataset.copy = none ? "" : cmd;

      history.replaceState(history.state, "", location.pathname + q + location.hash);
    };

    // A shared link's filter: ignored when invalid (everything stays on).
    const params = new URLSearchParams(location.search);
    const parsed = parseFilter(params);
    if (!("error" in parsed)) {
      const f = parsed.filter;
      el.querySelectorAll<HTMLInputElement>("input[type=checkbox]").forEach((c) => {
        c.checked = (f[c.name as keyof Filter] as Set<string>).has(c.value);
      });
    }
    update();

    const onSet = (e: Event) => {
      const btn = (e.target as Element).closest<HTMLElement>("[data-set]");
      if (!btn) return;
      el.querySelectorAll<HTMLInputElement>(`input[name="${btn.dataset.set}"]`).forEach((c) => (c.checked = btn.dataset.v === "1"));
      update();
    };

    const onClick = (e: MouseEvent) => {
      const t = e.target as Element;
      const copy = t.closest<HTMLButtonElement>("[data-copy]");
      if (copy && copy.dataset.copy) {
        navigator.clipboard?.writeText(copy.dataset.copy);
        const text = copy.textContent;
        copy.classList.add(styles.done);
        copy.textContent = "copied";
        setTimeout(() => {
          copy.classList.remove(styles.done);
          copy.textContent = text;
        }, 900);
      }
      const tog = t.closest<HTMLButtonElement>("[data-tog]");
      if (tog) {
        const tr = tog.closest("tr")!;
        const open = tr.classList.toggle(styles.open);
        tog.setAttribute("aria-expanded", String(open));
        document.querySelectorAll(`tr[data-child="${tog.dataset.tog}"]`).forEach((r) => r.classList.toggle(styles.shut, !open));
      }
      const seg = t.closest<HTMLButtonElement>("[data-seg] button");
      if (seg) {
        const style = seg.dataset.style;
        document.querySelectorAll("[data-seg] button").forEach((x) => x.setAttribute("aria-pressed", String(x === seg)));
        let text = "";
        document.querySelectorAll<HTMLElement>("[data-ref]").forEach((r) => {
          r.hidden = r.dataset.ref !== style;
          if (!r.hidden) text = r.dataset.text ?? "";
        });
        document.querySelector<HTMLElement>("[data-citecopy]")!.dataset.copy = text;
      }
    };

    el.addEventListener("change", update);
    el.addEventListener("click", onSet);
    document.addEventListener("click", onClick);
    return () => {
      el.removeEventListener("change", update);
      el.removeEventListener("click", onSet);
      document.removeEventListener("click", onClick);
    };
  }, [base, zips, editions]);

  const allCmd = command(SITE + base + "urls.txt");
  return (
    <form ref={form} className={styles.panel} onSubmit={(e) => e.preventDefault()} aria-label="Filter and download">
      <div className={styles.filters}>
        <div className={styles.frow}>
          <span className={styles.lbl}>Format</span>
          <div className={styles.chips}>
            <Chips name="format" options={FORMATS} />
          </div>
        </div>
        <div className={styles.frow}>
          <span className={styles.lbl}>Layer</span>
          <div className={styles.chips}>
            <Chips name="layer" options={LAYERS} />
          </div>
        </div>
        <div className={styles.frow}>
          <span className={styles.lbl}>Edition</span>
          <div className={styles.chips}>
            <Chips name="edition" options={editions.map((e) => [e.era, e.label])} />
            <button type="button" className={styles.mini} data-set="edition" data-v="1">
              all
            </button>
            <button type="button" className={styles.mini} data-set="edition" data-v="0">
              none
            </button>
          </div>
        </div>
      </div>
      <div className={styles.get}>
        <div className={styles.count} data-count aria-live="polite">
          {`${fileCount(files)} · ${formatBytes(bytes)}`}
          <small>Everything in this release</small>
        </div>
        <a className={styles.btn} href={zips.all.href}>
          ↓ Download the whole release as zip ({formatBytes(zips.all.bytes)})
        </a>
        <a className={`${styles.btn} ${styles.btn2}`} href={zips.shp.href} data-fmtzip hidden>
          ↓ Download all {ZIP_NAME.shp} as zip ({formatBytes(zips.shp.bytes)})
        </a>
        <p className={styles.zipnote} data-zipnote>
          Zips exist for the whole release and for each vector format; the command below fetches any selection.
        </p>
        <div className={styles.alts}>
          <a href={`${base}urls.txt`} data-urls>
            urls.txt
          </a>
          <a href={`${base}SHA256SUMS`} data-sums>
            SHA256SUMS
          </a>
        </div>
        <div className={styles.cmd}>
          <span data-cmd>{allCmd}</span>
          <button type="button" className={styles.copy} data-copy={allCmd} data-cmdcopy>
            copy
          </button>
        </div>
      </div>
    </form>
  );
}
