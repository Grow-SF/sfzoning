"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useId, useState } from "react";
import { menuItems } from "@/lib/menuItems";
import styles from "./siteMenu.module.css";

export default function SiteMenu({ hasDownloads }: { hasDownloads: boolean }) {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const listId = useId();
  const items = menuItems(hasDownloads);

  // Links from before the explorer moved to /maps pointed at /#explore.
  useEffect(() => {
    if (pathname === "/" && window.location.hash === "#explore") router.replace("/maps");
  }, [pathname, router]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <header className={styles.menu}>
      <nav className={styles.bar} aria-label="Site">
        <Link href="/" className={styles.title} onClick={() => setOpen(false)}>
          A History of San Francisco’s Zoning
        </Link>
        <button
          type="button"
          className={styles.toggle}
          aria-expanded={open}
          aria-controls={listId}
          aria-label="Menu"
          onClick={() => setOpen((o) => !o)}
        >
          <span aria-hidden>{open ? "×" : "☰"}</span>
        </button>
        <ul id={listId} className={styles.list} data-open={open || undefined}>
          {items.map((i) => (
            <li key={i.href}>
              <Link href={i.href} aria-current={pathname === i.href ? "page" : undefined} onClick={() => setOpen(false)}>
                {i.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
