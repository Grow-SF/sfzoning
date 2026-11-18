export type MenuItem = { href: string; label: string };

/** The top menu's links; /data appears only once the manifest lists files. */
export function menuItems(hasDownloads: boolean): MenuItem[] {
  return [
    { href: "/", label: "Story" },
    { href: "/maps", label: "All Maps" },
    ...(hasDownloads ? [{ href: "/data", label: "Download the Data" }] : []),
    { href: "/about", label: "About" },
  ];
}
