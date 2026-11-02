import { createHash } from "node:crypto";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { putOne } from "../scripts/blob-put.mjs";

function file(body: string) {
  const dir = mkdtempSync(path.join(tmpdir(), "blobput-"));
  const p = path.join(dir, "f");
  writeFileSync(p, body);
  return { p, md5: createHash("md5").update(body).digest("hex") };
}

const dvcName = (md5: string) => `dvc/files/md5/${md5.slice(0, 2)}/${md5.slice(2)}`;

describe("blob-put", () => {
  it("never sends a file whose bytes don't match its name, or a name outside the cache layout", async () => {
    const f = file("hello");
    const sent: string[] = [];
    const put = async (name: string) => {
      sent.push(name);
    };
    const wrong = "0".repeat(32);
    await expect(putOne({ file: f.p, name: dvcName(wrong), alg: "md5", hash: wrong }, { put })).rejects.toThrow(/doesn't hash/);
    await expect(putOne({ file: f.p, name: "shared/districts.json", alg: "md5", hash: f.md5 }, { put })).rejects.toThrow(/content-addressed/);
    await expect(putOne({ file: f.p, name: "dvc/files/md5/../../x", alg: "md5", hash: f.md5 }, { put })).rejects.toThrow(/content-addressed/);
    expect(sent).toEqual([]);
  });
});
