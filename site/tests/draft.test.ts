import { describe, expect, it } from "vitest";
import { draftMetadata, isDraft } from "@/lib/draft";

describe("draft flag", () => {
  it("is on unless NEXT_PUBLIC_DRAFT is exactly 'false'", () => {
    expect(isDraft(undefined)).toBe(true);
    expect(isDraft("")).toBe(true);
    expect(isDraft("true")).toBe(true);
    expect(isDraft("0")).toBe(true);
    expect(isDraft("False")).toBe(true);
    expect(isDraft("false")).toBe(false);
  });
  it("keeps drafts out of search engines", () => {
    expect(draftMetadata(undefined)).toEqual({ robots: { index: false, follow: false } });
    expect(draftMetadata("false")).toEqual({});
  });
});
