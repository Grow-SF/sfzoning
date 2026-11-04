import { latestUrl, type Manifest } from "@/lib/downloadRows";
import manifest from "@/lib/downloads.json";

/** /data/latest/<series>/<file>: a stable link that redirects to the file's current url. */
export async function GET(_req: Request, { params }: { params: Promise<{ series: string; file: string }> }) {
  const { series, file } = await params;
  const url = latestUrl(manifest as unknown as Manifest, series, file);
  if (!url) return new Response("Not found", { status: 404 });
  return new Response(null, { status: 307, headers: { location: url } });
}
