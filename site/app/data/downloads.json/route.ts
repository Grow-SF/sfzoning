import type { Manifest } from "@/lib/downloadRows";
import manifest from "@/lib/downloads.json";

export const dynamic = "force-static";

/** The download manifest, for scripts that want every file and its checksum. */
export function GET() {
  return Response.json(manifest as unknown as Manifest);
}
