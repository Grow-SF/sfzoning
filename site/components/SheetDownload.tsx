// DRAFT COPY — NOT REVIEWED BY THE OWNER. Button, progress and error wording are first drafts.
"use client";

import { useRef, useState } from "react";
import { formatBytes, type Manifest } from "@/lib/downloadRows";
import { saveSheet, saveSheetsZip, type Progress, type Saved } from "@/lib/sheetZip";
import styles from "./downloads.module.css";

type Props = { series: string; sheets: Manifest["sheets"][string]; license?: string };
type State =
  | { kind: "idle" }
  | { kind: "running"; progress?: Progress }
  | { kind: "done"; saved: Saved }
  | { kind: "error"; message: string };

/** Progress redraws at most this often (ms); a 0.7 GB zip sends tens of thousands of chunks. */
const PROGRESS_MS = 250;
const isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

/** The sheets of one series: all of them as a zip built in the browser, or one at a time. */
export default function SheetDownload({ series, sheets, license }: Props) {
  const [state, setState] = useState<State>({ kind: "idle" });
  const [one, setOne] = useState<{ name: string; error?: string } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const total = sheets.items.reduce((n, i) => n + i.bytes, 0);

  async function all() {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setState({ kind: "running" });
    let last = 0;
    let sheetsDone = -1;
    const onProgress = (progress: Progress) => {
      const now = performance.now();
      // Always redraw when a sheet finishes; otherwise a few times a second.
      if (now - last < PROGRESS_MS && progress.sheetsDone === sheetsDone) return;
      last = now;
      sheetsDone = progress.sheetsDone;
      setState({ kind: "running", progress });
    };
    try {
      const saved = await saveSheetsZip({ sheets: { [series]: sheets }, license }, series, {
        signal: ctrl.signal,
        onProgress,
      });
      setState({ kind: "done", saved });
    } catch (e) {
      setState(isAbort(e) || ctrl.signal.aborted ? { kind: "idle" } : { kind: "error", message: (e as Error).message });
    } finally {
      abort.current = null;
    }
  }

  async function single(item: Props["sheets"]["items"][number]) {
    setOne({ name: item.name });
    try {
      await saveSheet(item);
      setOne(null);
    } catch (e) {
      setOne({ name: item.name, error: (e as Error).message });
    }
  }

  const p = state.kind === "running" ? state.progress : undefined;
  return (
    <span className={styles.item}>
      {state.kind === "running" ? (
        <span className={styles.progress} role="status">
          {p
            ? `${p.sheetsDone} of ${p.sheets} sheets · ${formatBytes(p.bytes)} of ${formatBytes(p.totalBytes)}`
            : "Starting…"}{" "}
          <button type="button" className={styles.button} onClick={() => abort.current?.abort()}>
            Cancel
          </button>
        </span>
      ) : license ? (
        <button type="button" className={styles.button} onClick={all}>
          All {sheets.items.length} sheets ({formatBytes(total)}, zip)
        </button>
      ) : (
        // No license text to put in the zip: offer the sheets one at a time only.
        <span>
          {sheets.items.length} sheets ({formatBytes(total)})
        </span>
      )}
      {state.kind === "done" && (
        <span className={styles.meta} role="status">
          {state.saved === "file"
            ? `Saved ${series}-sheets.zip.`
            : `Download of ${series}-sheets.zip handed to your browser.`}
        </span>
      )}
      {state.kind === "error" && (
        <span className={styles.error} role="alert">
          {state.message}
        </span>
      )}
      <span className={styles.meta}>
        {sheets.updated} · <a href={`/data/sheets/${series}`}>sheet list (CSV)</a>
      </span>
      <details className={styles.sheets}>
        <summary>One sheet at a time</summary>
        <ul className={styles.sheetList}>
          {sheets.items.map((i) => (
            <li key={i.sheet}>
              <button type="button" className={styles.link} onClick={() => single(i)} disabled={one?.name === i.name && !one.error}>
                {i.name}
              </button>{" "}
              <span className={styles.size}>{formatBytes(i.bytes)}</span>
              {one?.name === i.name && !one.error && <span className={styles.size}> · downloading…</span>}
              {one?.name === i.name && one.error && <span className={styles.error} role="alert">{one.error}</span>}
            </li>
          ))}
        </ul>
      </details>
    </span>
  );
}
