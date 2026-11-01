import type { PMTiles, Protocol, RangeResponse, Source } from "pmtiles";

/** All archives together keep at most this many bytes of what they have read. */
export const CACHE_BYTES = 32 * 1024 * 1024;

/**
 * Recent range reads, shared by every archive: keyed on archive, offset and length, least recently
 * used evicted first once the total passes `limit`.
 */
export class ByteCache {
  private entries = new Map<string, RangeResponse>();
  private size = 0;
  constructor(readonly limit = CACHE_BYTES) {}

  get(key: string): RangeResponse | undefined {
    const hit = this.entries.get(key);
    if (hit) {
      this.entries.delete(key);
      this.entries.set(key, hit); // now the most recent
    }
    return hit;
  }

  set(key: string, value: RangeResponse) {
    const old = this.entries.get(key);
    if (old) this.size -= old.data.byteLength;
    this.entries.delete(key);
    if (value.data.byteLength > this.limit) return;
    this.entries.set(key, value);
    this.size += value.data.byteLength;
    for (const [k, v] of this.entries) {
      if (this.size <= this.limit) break;
      this.entries.delete(k);
      this.size -= v.data.byteLength;
    }
  }
}

/**
 * `source` with its reads kept in `cache`. Two reads of a range make one upstream request, even
 * while the first is in flight; a failed read is not kept, so the next one tries again. The browser's
 * own HTTP cache is not relied on (it re-fetches ranges, and pmtiles turns it off for Chromium on
 * Windows). Each caller gets its own copy of the bytes, since MapLibre may transfer them away.
 */
export function cachedSource(source: Source, cache: ByteCache): Source {
  const inflight = new Map<string, Promise<RangeResponse>>();
  return {
    getKey: () => source.getKey(),
    async getBytes(offset, length, signal, etag) {
      const key = `${source.getKey()}#${offset}+${length}`;
      let got = cache.get(key);
      if (!got) {
        let pending = inflight.get(key);
        if (!pending) {
          // Not tied to one caller's signal: others may be waiting on the same read.
          pending = source.getBytes(offset, length, undefined, etag).then(
            (r) => (cache.set(key, r), inflight.delete(key), r),
            (e) => (inflight.delete(key), Promise.reject(e)),
          );
          inflight.set(key, pending);
        }
        got = await pending;
      }
      if (signal?.aborted) throw new DOMException("aborted", "AbortError");
      return { ...got, data: got.data.slice(0) };
    },
  };
}

/** The classes of the pmtiles module, which the site loads on demand. */
export type Pm = { PMTiles: typeof PMTiles; FetchSource: new (url: string) => Source };

const shared = new ByteCache();

/**
 * Put the archive at `url` (plain http URL) into `protocol`, reading through the shared cache, unless
 * it is there already. Do it before MapLibre first asks for the archive, which would otherwise
 * make its own uncached one.
 */
export function registerArchive(protocol: Protocol, url: string, pm: Pm): PMTiles {
  let archive = protocol.get(url);
  if (!archive) {
    archive = new pm.PMTiles(cachedSource(new pm.FetchSource(url), shared));
    protocol.add(archive);
  }
  return archive;
}
