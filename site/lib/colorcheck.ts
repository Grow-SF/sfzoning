// Colour-difference helpers for checking the map palette: OKLab ΔE (×100) under normal
// vision and simulated protanopia / deuteranopia (Machado, Oliveira & Fernandes 2009,
// severity 1.0, applied in linear sRGB). Used by tests/palette.test.ts and nothing at runtime.

export type RGB = [number, number, number]; // 0..1 sRGB (gamma-encoded)
export type Vision = "normal" | "protan" | "deutan";

export function hexToRgb(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

const toLin = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

const MACHADO: Record<Exclude<Vision, "normal">, number[][]> = {
  protan: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998],
  ],
  deutan: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.01182, 0.04294, 0.968881],
  ],
};

/** Alpha-composite `top` at opacity `a` over `base` (gamma space, as browsers and WebGL do). */
export function over(top: RGB, a: number, base: RGB): RGB {
  return [0, 1, 2].map((i) => top[i] * a + base[i] * (1 - a)) as RGB;
}

export function oklab(rgb: RGB, vision: Vision = "normal"): [number, number, number] {
  let [r, g, b] = rgb.map(toLin);
  if (vision !== "normal") {
    const m = MACHADO[vision];
    const v = [r, g, b];
    [r, g, b] = m.map((row) => Math.min(1, Math.max(0, row[0] * v[0] + row[1] * v[1] + row[2] * v[2])));
  }
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

/** OKLab distance ×100 between two colours as seen with the given vision. */
export function deltaE(a: RGB, b: RGB, vision: Vision = "normal"): number {
  const x = oklab(a, vision);
  const y = oklab(b, vision);
  return 100 * Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]);
}
