// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, createElement, createRef, useEffect, type RefObject } from "react";
import { createRoot, type Root } from "react-dom/client";
import { activeIndex, useScrollSteps } from "@/lib/useScrollSteps";

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("activeIndex", () => {
  it("is the last card whose top has crossed the line", () => {
    expect(activeIndex([100, 500, 900], 590)).toBe(1);
    expect(activeIndex([-800, -100, 400], 590)).toBe(2);
  });
  it("is the first step before any card has crossed", () => {
    expect(activeIndex([700, 1500], 590)).toBe(0);
  });
  it("holds a step between cards (no gap where nothing is active)", () => {
    // Card 1 has scrolled wholly above the line, card 2 has not reached it.
    expect(activeIndex([-2000, -600, 800], 590)).toBe(1);
  });
});

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
let tops: number[] = [];

function Harness({ refs }: { refs: RefObject<HTMLElement | null>[] }) {
  const { active } = useScrollSteps(refs, { offset: 0.7 });
  useEffect(() => {
    reports.push(active);
  }, [active]);
  return createElement(
    "div",
    null,
    refs.map((r, i) => createElement("article", { key: i, ref: r as RefObject<HTMLElement>, "data-i": i })),
  );
}

beforeEach(() => {
  frames = [];
  reports.length = 0;
  tops = [];
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    frames.push(cb);
    return frames.length;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    const top = tops[Number(this.dataset.i)] ?? 0;
    return { top, bottom: top + 300, left: 0, right: 0, width: 0, height: 300, x: 0, y: top, toJSON() {} } as DOMRect;
  });
  Object.defineProperty(window, "innerHeight", { value: 844, configurable: true });
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useScrollSteps", () => {
  it("switches when a card's top crosses 70% of the viewport, not before", () => {
    const refs = [0, 1, 2].map(() => createRef<HTMLElement>());
    tops = [200, 1200, 2200];
    act(() => root.render(createElement(Harness, { refs })));
    flushFrame();
    expect(reports.at(-1)).toBe(0);

    tops = [-400, 600, 1600]; // card 1 is on screen but its top is below 590 (70% of 844)
    act(() => void window.dispatchEvent(new Event("scroll")));
    flushFrame();
    expect(reports.at(-1)).toBe(0);

    tops = [-420, 580, 1580];
    act(() => void window.dispatchEvent(new Event("scroll")));
    flushFrame();
    expect(reports.at(-1)).toBe(1);
  });

  it("reports only where a fast fling landed, once per frame", () => {
    const refs = [0, 1, 2, 3].map(() => createRef<HTMLElement>());
    tops = [100, 1100, 2100, 3100];
    act(() => root.render(createElement(Harness, { refs })));
    flushFrame();
    act(() => {
      for (const t of [[-900, 100, 1100, 2100], [-1900, -900, 100, 1100], [-2900, -1900, -900, 100]]) {
        tops = t;
        window.dispatchEvent(new Event("scroll"));
      }
    });
    expect(reports).toEqual([0]); // nothing reported before the frame
    flushFrame();
    expect(reports).toEqual([0, 3]); // 1 and 2 were never reported
  });
});
