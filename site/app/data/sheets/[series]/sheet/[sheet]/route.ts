import type { Manifest } from "@/lib/downloadRows";
import manifestJson from "@/lib/downloads.json";
import { sheetResponse } from "@/lib/sheetZip";

const manifest = manifestJson as unknown as Manifest;

export const dynamic = "force-dynamic";
export const maxDuration = 300;

/** One sheet's GeoTIFF, saved under its readable name. */
export async function GET(req: Request, { params }: { params: Promise<{ series: string; sheet: string }> }) {
  const { series, sheet } = await params;
  return sheetResponse(manifest, series, sheet, fetch, req.signal);
}
