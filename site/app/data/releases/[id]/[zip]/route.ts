import { releaseZipResponse } from "@/lib/releaseRoutes";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string; zip: string }> }) {
  const { id, zip } = await params;
  return releaseZipResponse(id, zip);
}
