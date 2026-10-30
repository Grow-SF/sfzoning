import type { Metadata } from "next";

/**
 * The site is a draft until a build sets NEXT_PUBLIC_DRAFT to exactly "false".
 * Anything else (unset, "", "0", "no") keeps the banner up and search engines out, so a
 * mistyped flag fails safe. Read at build time: Next inlines NEXT_PUBLIC_* values.
 */
export function isDraft(flag: string | undefined = process.env.NEXT_PUBLIC_DRAFT): boolean {
  return flag !== "false";
}

/** Metadata to merge into the root layout's: noindex/nofollow while the copy is a draft. */
export function draftMetadata(flag: string | undefined = process.env.NEXT_PUBLIC_DRAFT): Pick<Metadata, "robots"> {
  return isDraft(flag) ? { robots: { index: false, follow: false } } : {};
}
