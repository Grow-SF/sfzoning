"use client";

import { useState } from "react";
import {
  clearCorrections,
  correctionsFile,
  issueBody,
  issueUrl,
  removeCorrection,
  REPO,
  useCorrections,
} from "@/lib/corrections";
import styles from "./explorer.module.css";

/** Under the explorer map, once the reader has queued corrections: review them and send. */
export default function CorrectionsTray() {
  const list = useCorrections();
  const [copied, setCopied] = useState(false);
  if (!list.length) return null;
  const url = issueUrl(list);

  const download = () => {
    const blob = new Blob([correctionsFile(list)], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sfzoning-corrections.md";
    a.click();
    URL.revokeObjectURL(a.href);
  };
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(issueBody(list));
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <section className={styles.tray} aria-labelledby="tray-title" data-testid="corrections-tray">
      <h3 id="tray-title" className={styles.trayTitle}>
        Your corrections ({list.length})
      </h3>
      <ul className={styles.trayList}>
        {list.map((c) => (
          <li key={`${c.era}-${c.blklot}`}>
            <span>
              {c.era}, block {c.blklot.slice(0, 4)} lot {c.blklot.slice(4)}: {c.shown || "none"} &rarr;{" "}
              <strong>{c.proposed}</strong>
              {c.note ? ` (${c.note})` : ""}
            </span>
            <button
              type="button"
              className={styles.linkButton}
              onClick={() => removeCorrection(c.era, c.blklot)}
              aria-label={`Remove the correction for ${c.era}, lot ${c.blklot}`}
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
      <div className={styles.trayButtons}>
        {url ? (
          <a className={styles.trayPrimary} href={url} target="_blank" rel="noopener noreferrer" data-testid="create-issue">
            Create issue on GitHub
          </a>
        ) : (
          <span className={styles.trayNote}>
            Too many to fit in a link: download the file and attach it to a{" "}
            <a href={`https://github.com/${REPO}/issues/new?labels=correction`} target="_blank" rel="noopener noreferrer">
              new issue
            </a>
            .
          </span>
        )}
        <button type="button" onClick={download}>
          Download file
        </button>
        <button type="button" onClick={copy}>
          {copied ? "Copied" : "Copy text"}
        </button>
        <button type="button" className={styles.linkButton} onClick={clearCorrections}>
          Clear all
        </button>
      </div>
      <p className={styles.trayNote}>
        Creating an issue needs a GitHub account. We check each correction against the scan before it changes the map.
        Your list stays in this browser until you clear it.
      </p>
    </section>
  );
}
