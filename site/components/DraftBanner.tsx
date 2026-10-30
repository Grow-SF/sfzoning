import { isDraft } from "@/lib/draft";

/** Shown on every page until NEXT_PUBLIC_DRAFT is "false" (see lib/draft.ts). */
export default function DraftBanner() {
  if (!isDraft()) return null;
  return (
    <p className="draft-banner" data-testid="draft-banner">
      Draft copy, not reviewed. Do not publish.
    </p>
  );
}
