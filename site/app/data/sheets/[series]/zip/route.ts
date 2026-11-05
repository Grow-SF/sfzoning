import type { Manifest } from "@/lib/downloadRows";
import manifestJson from "@/lib/downloads.json";
import { sheetsZipResponse } from "@/lib/sheetZip";

const manifest = manifestJson as unknown as Manifest;

export const dynamic = "force-dynamic";
// A whole edition (up to ~0.7 GB) streams through this function; give slow connections time.
export const maxDuration = 800;

/** `<series>-sheets.zip`: every sheet of the series, zipped while it streams. */
export async function GET(req: Request, { params }: { params: Promise<{ series: string }> }) {
  const { series } = await params;
  return sheetsZipResponse(manifest, series, fetch, req.signal);
}
