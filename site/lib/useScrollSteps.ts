"use client";

import { useEffect, useRef, useState, type RefObject } from "react";

export type ScrollStepsOptions = {
  /** Where the trigger line sits, as a fraction of viewport height from the top. Default 0.7. */
  offset?: number;
};

export type ScrollStepsState = { active: number };

/**
 * The active step is the last one whose element's top has crossed the trigger line, or the
 * first step before any has. Pass the step *cards*, not their sections: on phones a card sits
 * at the bottom of a full-screen section, so keying off the card means the map changes as the
 * card itself comes up into view, not a screen earlier.
 *
 * Unlike an "intersecting the line" rule, this has no gaps: a short card that never straddles
 * the line, or a jump that lands between cards, still resolves to the step the reader is on.
 */
export function activeIndex(tops: number[], line: number): number {
  let active = 0;
  tops.forEach((top, i) => {
    if (top <= line) active = i;
  });
  return active;
}

/**
 * Tracks the active step. Scroll and resize events are coalesced into one measurement per
 * animation frame, so a fast fling across several steps reports only where it landed.
 */
export function useScrollSteps(
  refs: RefObject<HTMLElement | null>[],
  opts: ScrollStepsOptions = {},
): ScrollStepsState {
  const offset = opts.offset ?? 0.7;
  const [active, setActive] = useState(0);
  const activeRef = useRef(0);

  useEffect(() => {
    const els = refs.map((r) => r.current).filter((el): el is HTMLElement => !!el);
    if (!els.length) return;

    let frame = 0;
    const measure = () => {
      frame = 0;
      const next = activeIndex(
        els.map((el) => el.getBoundingClientRect().top),
        window.innerHeight * offset,
      );
      if (next !== activeRef.current) {
        activeRef.current = next;
        setActive(next);
      }
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    schedule(); // a reload can restore a scrolled position

    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
      if (frame) cancelAnimationFrame(frame);
    };
    // refs are stable per mount; re-run only when the step count changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refs.length, offset]);

  return { active };
}
