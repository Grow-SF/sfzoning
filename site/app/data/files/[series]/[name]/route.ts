import type { Manifest } from "@/lib/downloadRows";
import manifest from "@/lib/downloads.json";
import history from "@/lib/download-versions.json";
import { filesResponse } from "@/lib/filesRoute";
import type { VersionHistory } from "@/lib/versions";

export const dynamic = "force-dynamic";
// The sheet zip (a whole edition, up to ~0.7 GB) streams through this function.
export const maxDuration = 800;

/** Versioned names redirect to Blob; `latest` names redirect to the current version or stream the sheet zip. */
export async function GET(req: Request, { params }: { params: Promise<{ series: string; name: string }> }) {
  const { series, name } = await params;
  return filesResponse(manifest as unknown as Manifest, series, name, fetch, req.signal, history as VersionHistory);
}
