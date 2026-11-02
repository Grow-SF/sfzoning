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
  const md5 = createHash("md5").update(body).digest("hex");
  return { p, md5, size: Buffer.byteLength(body) };
}

/** A fake store: sizes by name; HEAD answers from it, put adds to it. */
function store(initial: Record<string, number> = {}) {
  const sizes = { ...initial };
  const puts: string[] = [];
  const fetchImpl = async (url: string) => {
    const name = new URL(url).pathname.slice(1);
    expect(new URL(url).searchParams.has("check")).toBe(true); // never a cacheable real-URL miss
    return sizes[name] === undefined
      ? new Response(null, { status: 404 })
      : new Response(null, { status: 200, headers: { "content-length": String(sizes[name]) } });
  };
  const put = async (name: string, _body: unknown, opts: { allowOverwrite: boolean }) => {
    expect(opts.allowOverwrite).toBe(false);
    puts.push(name);
    sizes[name] = sizes.__next ?? 0;
  };
  return { sizes, puts, fetchImpl, put };
}

const dvcName = (md5: string) => `dvc/files/md5/${md5.slice(0, 2)}/${md5.slice(2)}`;
const noWait = async () => {};

describe("blob-put", () => {
  it("uploads a missing object and confirms its size", async () => {
    const f = file("hello");
    const s = store({ __next: f.size });
    const name = dvcName(f.md5);
    expect(await putOne({ file: f.p, name, alg: "md5", hash: f.md5 }, { ...s, wait: noWait })).toBe("uploaded");
    expect(s.puts).toEqual([name]);
  });

  it("skips an object already there, and never overwrites one that differs", async () => {
    const f = file("hello");
    const name = dvcName(f.md5);
    const same = store({ [name]: f.size });
    expect(await putOne({ file: f.p, name, alg: "md5", hash: f.md5 }, same)).toBe("present");
    expect(same.puts).toEqual([]);
    const other = store({ [name]: 999 });
    await expect(putOne({ file: f.p, name, alg: "md5", hash: f.md5 }, other)).rejects.toThrow(/already published/);
    expect(other.puts).toEqual([]);
  });

  it("refuses a file that doesn't hash to its name, and names it doesn't accept", async () => {
    const f = file("hello");
    const s = store();
    const wrong = "0".repeat(32);
    await expect(putOne({ file: f.p, name: dvcName(wrong), alg: "md5", hash: wrong }, s)).rejects.toThrow(/doesn't hash/);
    await expect(putOne({ file: f.p, name: "shared/districts.json", alg: "md5", hash: f.md5 }, s)).rejects.toThrow(/content-addressed/);
    await expect(putOne({ file: f.p, name: "dvc/files/md5/../../x", alg: "md5", hash: f.md5 }, s)).rejects.toThrow(/content-addressed/);
    expect(s.puts).toEqual([]);
  });

  it("treats a failed put whose bytes landed anyway (a retry) as uploaded", async () => {
    const f = file("hello");
    const name = dvcName(f.md5);
    const s = store();
    const put = async () => {
      s.sizes[name] = f.size;
      throw new Error("This blob already exists");
    };
    expect(await putOne({ file: f.p, name, alg: "md5", hash: f.md5 }, { ...s, put, wait: noWait })).toBe("uploaded");
  });
});
