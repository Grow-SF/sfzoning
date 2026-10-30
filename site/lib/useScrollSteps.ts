"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

export type ScrollStepsOptions = {
  /** Where the trigger line sits, as a fraction of viewport height from the top. Default 0.5. */
  offset?: number;
};

export type ScrollStepsState = { active: number; progress: number };

/**
 * Scrollama-style step tracker. A step is active while it crosses the trigger line
 * (the viewport middle by default). Observer callbacks are coalesced into one report per
 * animation frame, so a fast fling across several steps reports only where it landed.
 * `progress` (0..1) is how far the trigger line has travelled through the active step.
 */
export function useScrollSteps(
  refs: RefObject<HTMLElement | null>[],
  opts: ScrollStepsOptions = {},
): ScrollStepsState {
  const offset = opts.offset ?? 0.5;
  const [active, setActive] = useState(0);
  const [progress, setProgress] = useState(0);
  const activeRef = useRef(0);

  useEffect(() => {
    const els = refs.map((r) => r.current).filter((el): el is HTMLElement => !!el);
    if (!els.length || typeof IntersectionObserver === "undefined") return;

    let pending: number | null = null;
    let frame = 0;
    const flush = () => {
      frame = 0;
      if (pending === null) return;
      activeRef.current = pending;
      setActive(pending);
      pending = null;
    };

    const top = -Math.round(offset * 100);
    const bottom = -Math.round((1 - offset) * 100);
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const i = els.indexOf(e.target as HTMLElement);
          if (i >= 0) pending = i; // later entries win
        }
        if (pending !== null && !frame) frame = requestAnimationFrame(flush);
      },
      { rootMargin: `${top}% 0px ${bottom}% 0px`, threshold: 0 },
    );
    els.forEach((el) => io.observe(el));

    let pFrame = 0;
    const measure = () => {
      pFrame = 0;
      const el = els[activeRef.current];
      if (!el) return;
      const r = el.getBoundingClientRect();
      const line = window.innerHeight * offset;
      const p = r.height ? (line - r.top) / r.height : 0;
      setProgress(Math.min(1, Math.max(0, p)));
    };
    const onScroll = () => {
      if (!pFrame) pFrame = requestAnimationFrame(measure);
    };
    window.addEventListener("scroll", onScroll, { passive: true });

    return () => {
      io.disconnect();
      window.removeEventListener("scroll", onScroll);
      if (frame) cancelAnimationFrame(frame);
      if (pFrame) cancelAnimationFrame(pFrame);
    };
    // refs are stable per mount; re-run only when the step count changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refs.length, offset]);

  return { active, progress };
}
