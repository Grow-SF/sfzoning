import { downloadsJson, type Manifest } from "@/lib/downloadRows";
import manifest from "@/lib/downloads.json";

export const dynamic = "force-static";

/** The download manifest, for scripts that want every file, its checksum and its versioned and latest links. */
export function GET() {
  return Response.json(downloadsJson(manifest as unknown as Manifest));
}
