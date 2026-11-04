import { describe, expect, it } from "vitest";
import { menuItems } from "../lib/menuItems";

describe("menuItems", () => {
  it("lists Download the Data only when there are downloads, in a fixed order", () => {
    expect(menuItems(false).map((i) => i.href)).toEqual(["/", "/maps", "/about"]);
    expect(menuItems(true)).toEqual([
      { href: "/", label: "Story" },
      { href: "/maps", label: "All Maps" },
      { href: "/data", label: "Download the Data" },
      { href: "/about", label: "About" },
    ]);
  });
});
