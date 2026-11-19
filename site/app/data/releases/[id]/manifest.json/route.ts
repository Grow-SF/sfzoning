import { releaseTextResponse } from "@/lib/releaseRoutes";

export const dynamic = "force-dynamic";

export async function GET(req: Request, { params }: { params: Promise<{ id: string }> }) {
  return releaseTextResponse("manifest", (await params).id, new URL(req.url));
}
