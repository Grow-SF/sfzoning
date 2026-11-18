import { sheetsCsv, type Manifest } from "@/lib/downloadRows";
import manifestJson from "@/lib/downloads.json";

const manifest = manifestJson as unknown as Manifest;

export const dynamic = "force-static";
export const dynamicParams = false;

export function generateStaticParams() {
  return Object.keys(manifest.sheets).map((series) => ({ series }));
}

/** One CSV per series listing every sheet GeoTIFF: name, url, size, md5 and date. */
export async function GET(_req: Request, { params }: { params: Promise<{ series: string }> }) {
  const { series } = await params;
  return new Response(sheetsCsv(manifest, series), {
    headers: { "content-type": "text/csv; charset=utf-8" },
  });
}
