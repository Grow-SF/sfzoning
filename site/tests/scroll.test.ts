// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, createElement, createRef, useEffect, type RefObject } from "react";
import { createRoot, type Root } from "react-dom/client";
import { useScrollSteps } from "@/lib/useScrollSteps";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

type Cb = (entries: Partial<IntersectionObserverEntry>[]) => void;
let observers: { cb: Cb; opts?: IntersectionObserverInit; els: Element[] }[] = [];

class MockIO {
  rec: (typeof observers)[number];
  constructor(cb: Cb, opts?: IntersectionObserverInit) {
    this.rec = { cb, opts, els: [] };
    observers.push(this.rec);
  }
  observe(el: Element) {
    this.rec.els.push(el);
  }
  unobserve() {}
  disconnect() {
    this.rec.els = [];
  }
  takeRecords() {
    return [];
  }
}

let frames: FrameRequestCallback[] = [];
const flushFrame = () =>
  act(() => {
    const f = frames;
    frames = [];
    f.forEach((cb) => cb(0));
  });

let root: Root;
let host: HTMLDivElement;
const reports: number[] = [];

function Harness({ refs }: { refs: RefObject<HTMLElement | null>[] }) {
  const { active } = useScrollSteps(refs);
  useEffect(() => {
    reports.push(active);
  }, [active]);
  return createElement(
    "div",
    null,
    refs.map((r, i) => createElement("section", { key: i, ref: r as RefObject<HTMLElement> })),
  );
}

beforeEach(() => {
  observers = [];
  frames = [];
  reports.length = 0;
  vi.stubGlobal("IntersectionObserver", MockIO);
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    frames.push(cb);
    return frames.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

const hit = (el: Element, on = true) => ({ target: el, isIntersecting: on }) as Partial<IntersectionObserverEntry>;

describe("useScrollSteps", () => {
  it("observes every step at the viewport middle", () => {
    const refs = [createRef<HTMLElement>(), createRef<HTMLElement>(), createRef<HTMLElement>()];
    act(() => root.render(createElement(Harness, { refs })));
    expect(observers).toHaveLength(1);
    expect(observers[0].els).toHaveLength(3);
    expect(observers[0].opts?.rootMargin).toBe("-50% 0px -50% 0px");
  });

  it("reports only the last of several changes in one frame", () => {
    const refs = [0, 1, 2, 3].map(() => createRef<HTMLElement>());
    act(() => root.render(createElement(Harness, { refs })));
    const io = observers[0];
    const [a, b, c, d] = io.els;
    act(() => {
      io.cb([hit(a, false), hit(b)]);
      io.cb([hit(b, false), hit(c)]);
      io.cb([hit(c, false), hit(d)]);
    });
    expect(reports).toEqual([0]); // nothing reported before the frame
    flushFrame();
    expect(reports).toEqual([0, 3]); // 1 and 2 were never reported
  });

  it("takes the last intersecting entry within one batch", () => {
    const refs = [0, 1, 2].map(() => createRef<HTMLElement>());
    act(() => root.render(createElement(Harness, { refs })));
    const io = observers[0];
    act(() => io.cb([hit(io.els[1]), hit(io.els[2])]));
    flushFrame();
    expect(reports.at(-1)).toBe(2);
  });
});
