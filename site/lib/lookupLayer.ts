import type * as Ml from "maplibre-gl";
import { CATEGORY_COLORS } from "./palette";
import { NO_VALUE, type SharedDistricts } from "./shared";

// Draws the shared districts with one copy of each shape and one small table, instead of one
// MapLibre layer per year, each of which holds its own per-vertex colours (and, for lines, its own
// geometry). MapLibre still loads, cuts and uploads the tiles: two layers it never draws hold them
// and carry each shape's number in their colour attribute. This layer draws those same buffers.
//
// It reads MapLibre internals (tile managers, buckets, vertex layouts), which is why maplibre-gl's
// version is pinned exactly. Recheck this file by eye when upgrading.

export const LOTS_LAYER = "dist-shared-lots";
export const LINES_LAYER = "dist-shared-lines";
const LAYER = "dist-shared-draw";
const TABLE_WIDTH = 2048;
const TEXTURE_UNIT = 7;
const EXTENT = 8192;
/** MapLibre's tile size for vector sources: its tiles are 512 px at their own zoom. */
const TILE_PX = 512;
/** A channel holds under 200, so the half added below can never push it past 255. */
const BASE = 200;

/**
 * A MapLibre colour holding a shape's number. Channels are stored rounded down after a trip through
 * 0..1, so each carries an extra half to land on the intended whole number.
 */
export function numberAsColor(property: string): unknown[] {
  const n = ["get", property];
  const channel = (x: unknown) => ["+", 0.5, x];
  return [
    "rgb",
    channel(["floor", ["/", n, BASE * BASE]]),
    channel(["%", ["floor", ["/", n, BASE]], BASE]),
    channel(["%", n, BASE]),
  ];
}

const PICK_NUMBER = `
  int x = int(a_color.x), y = int(a_color.y);
  int id = (x / 256) * ${BASE * BASE} + (x % 256) * ${BASE} + (y / 256);`;

const lookup = (row: string) => `int(texelFetch(u_table, ivec2(id % ${TABLE_WIDTH}, id / ${TABLE_WIDTH} + ${row}), 0).r * 255.0 + 0.5)`;

const FILL_VS = `#version 300 es
precision highp float; precision highp int;
in vec2 a_pos; in vec2 a_color;
uniform mat4 u_matrix; uniform highp sampler2D u_table; uniform vec4 u_palette[16];
uniform int u_row; uniform int u_prev; uniform float u_opacity; uniform float u_prev_opacity;
out vec4 v_color;
vec4 colour(int c) { return c == ${NO_VALUE} ? vec4(0.0) : u_palette[c]; }
void main() {${PICK_NUMBER}
  v_color = colour(${lookup("u_row")}) * u_opacity + colour(${lookup("u_prev")}) * u_prev_opacity;
  gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0);
}`;

const FILL_FS = `#version 300 es
precision highp float;
in vec4 v_color; out vec4 color;
void main() { if (v_color.a == 0.0) discard; color = v_color; }`;

// MapLibre's line vertex shader and antialiasing, for one width and no offset or gap.
const LINE_VS = `#version 300 es
precision highp float; precision highp int;
in ivec2 a_pos_normal; in uvec4 a_data; in vec2 a_color;
uniform mat4 u_matrix; uniform highp sampler2D u_table; uniform vec4 u_edge;
uniform int u_row; uniform int u_prev; uniform float u_opacity; uniform float u_prev_opacity;
uniform float u_ratio; uniform float u_width; uniform float u_dpr;
out vec4 v_color; out vec2 v_normal; out float v_outset; out vec2 v_tile;
void main() {${PICK_NUMBER}
  float o = float(${lookup("u_row")}) * u_opacity + float(${lookup("u_prev")}) * u_prev_opacity;
  v_color = u_edge * o;
  vec2 pos = vec2(a_pos_normal >> 1);
  vec2 normal = vec2(a_pos_normal & 1);
  normal.y = normal.y * 2.0 - 1.0;
  v_normal = normal;
  v_outset = u_width / 2.0 + 1.0 / u_dpr / 2.0;
  vec2 extrude = vec2(ivec2(a_data.xy) - 128);
  vec2 p = pos + v_outset * extrude * 0.015873016 / u_ratio;
  v_tile = p;
  gl_Position = o == 0.0 ? vec4(-2.0, -2.0, -2.0, 1.0) : u_matrix * vec4(p, 0.0, 1.0);
}`;

const LINE_FS = `#version 300 es
precision highp float;
in vec4 v_color; in vec2 v_normal; in float v_outset; in vec2 v_tile;
uniform float u_dpr; out vec4 color;
void main() {
  // Each tile holds a margin beyond its edge, which is its neighbour's to draw.
  if (v_tile.x < 0.0 || v_tile.y < 0.0 || v_tile.x > ${EXTENT}.0 || v_tile.y > ${EXTENT}.0) discard;
  float blur = 1.0 / u_dpr;
  float dist = length(v_normal) * v_outset;
  color = v_color * clamp((v_outset - dist) / blur, 0.0, 1.0);
}`;

export type LookupLook = {
  /** Era position in the table, or null to draw nothing. */
  era: number | null;
  fillOpacity: (zoom: number) => number;
  lineOpacity: number;
};

// The parts of MapLibre this file reads that it doesn't publish.
type Segment = { vertexOffset: number; primitiveOffset: number; primitiveLength: number };
type Bucket = {
  layoutVertexBuffer?: { buffer: WebGLBuffer };
  indexBuffer?: { buffer: WebGLBuffer };
  segments: { get(): Segment[] };
  programConfigurations: { get(id: string): { binders: Record<string, { paintVertexBuffer?: { buffer: WebGLBuffer } }> } };
};
type Coord = { overscaledZ: number; wrap: number; canonical: { x: number; y: number; z: number } };
type Tile = { getBucket(layer: unknown): Bucket | undefined };
type Inner = {
  style: {
    tileManagers: Record<string, { getVisibleCoordinates(): Coord[]; getTile(c: Coord): Tile }>;
    _layers: Record<string, unknown>;
  };
};

function compile(gl: WebGL2RenderingContext, vs: string, fs: string, attributes: string[]) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    gl.attachShader(program, s);
  }
  attributes.forEach((a, i) => gl.bindAttribLocation(program, i, a));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) ?? "program");
  const locations = new Map<string, WebGLUniformLocation | null>();
  return { program, at: (name: string) => {
    if (!locations.has(name)) locations.set(name, gl.getUniformLocation(program, name));
    return locations.get(name)!;
  } };
}

const rgba = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).concat(1);

/** Adds the holders and the drawing layer below `before`. Returns how to change what's shown. */
export function addLookupLayer(map: Ml.Map, shared: SharedDistricts, source: string, before: string, lineWidth: (zoom: number) => number, edge: string) {
  const inner = map as unknown as Inner;
  const lotRows = Math.ceil(shared.lots / TABLE_WIDTH);
  const lineRows = Math.ceil(shared.lineSets / TABLE_WIDTH);
  const eraCount = shared.eras.length;
  const lotRow = (era: number) => era * lotRows;
  const lineRow = (era: number) => eraCount * lotRows + era * lineRows;

  map.addLayer(
    { id: LOTS_LAYER, type: "fill", source, "source-layer": "parcels", paint: { "fill-opacity": 0, "fill-color": numberAsColor("i") as Ml.ExpressionSpecification } },
    before,
  );
  map.addLayer(
    { id: LINES_LAYER, type: "line", source, "source-layer": "districts", paint: { "line-opacity": 0, "line-color": numberAsColor("s") as Ml.ExpressionSpecification } },
    before,
  );

  const palette = new Float32Array(16 * 4);
  // The table's numbers are the pointer's category order, so at most 16 categories.
  shared.categories.forEach((c, i) => palette.set(rgba(CATEGORY_COLORS[c]), i * 4));
  const edgeColor = rgba(edge);

  let gl: WebGL2RenderingContext | null = null;
  let fill: ReturnType<typeof compile>;
  let line: ReturnType<typeof compile>;
  let table: WebGLTexture;

  function setup(context: WebGL2RenderingContext) {
    gl = context;
    fill = compile(gl, FILL_VS, FILL_FS, ["a_pos", "a_color"]);
    line = compile(gl, LINE_VS, LINE_FS, ["a_pos_normal", "a_data", "a_color"]);
    const height = eraCount * (lotRows + lineRows);
    const texels = new Uint8Array(TABLE_WIDTH * height);
    for (let e = 0; e < eraCount; e++) {
      const lots = shared.values.subarray(e * shared.lots, (e + 1) * shared.lots);
      texels.set(lots, lotRow(e) * TABLE_WIDTH);
      const lineStart = eraCount * shared.lots + e * shared.lineSets;
      texels.set(shared.values.subarray(lineStart, lineStart + shared.lineSets), lineRow(e) * TABLE_WIDTH);
    }
    table = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + TEXTURE_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, table);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, TABLE_WIDTH, height, 0, gl.RED, gl.UNSIGNED_BYTE, texels);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // What's on screen: the current look fading in over what it replaced.
  let look: LookupLook = { era: null, fillOpacity: () => 0, lineOpacity: 0 };
  let previous: LookupLook = look;
  let startedAt = 0;
  let duration = 0;

  function drawBuckets(g: WebGL2RenderingContext, args: Ml.CustomRenderMethodInput, matrixAt: WebGLUniformLocation | null, layerId: string, colorProperty: string, bind: (s: Segment, vertices: WebGLBuffer, numbers: WebGLBuffer) => void, perTile?: (c: Coord) => void) {
    const manager = inner.style.tileManagers[source];
    const holder = inner.style._layers[layerId];
    // Closest zoom first: a parent tile standing in for a child never draws over it.
    for (const coord of manager.getVisibleCoordinates().sort((a, b) => b.overscaledZ - a.overscaledZ)) {
      const bucket = manager.getTile(coord).getBucket(holder);
      const numbers = bucket?.programConfigurations.get(layerId).binders[colorProperty]?.paintVertexBuffer;
      if (!bucket?.layoutVertexBuffer || !bucket.indexBuffer || !numbers) continue;
      const matrix = args.getProjectionData({ tileID: { canonical: coord.canonical, wrap: coord.wrap } as never, aligned: false, applyGlobeMatrix: false }).mainMatrix;
      g.uniformMatrix4fv(matrixAt, false, matrix);
      perTile?.(coord);
      g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, bucket.indexBuffer.buffer);
      for (const s of bucket.segments.get()) {
        bind(s, bucket.layoutVertexBuffer.buffer, numbers.buffer);
        g.drawElements(g.TRIANGLES, s.primitiveLength * 3, g.UNSIGNED_SHORT, s.primitiveOffset * 6);
      }
    }
  }

  const layer: Ml.CustomLayerInterface = {
    id: LAYER,
    type: "custom",
    renderingMode: "2d",
    render(context, args) {
      const g = context as WebGL2RenderingContext;
      // A restored context has lost everything created on the old one.
      if (gl !== g || !g.isProgram(fill.program)) setup(g);
      const zoom = map.getZoom();
      const t = duration > 0 ? Math.min(1, (performance.now() - startedAt) / duration) : 1;
      const eraOr0 = (l: LookupLook) => l.era ?? 0;
      const fillOn = look.era === null ? 0 : look.fillOpacity(zoom) * t;
      const fillOff = previous.era === null ? 0 : previous.fillOpacity(zoom) * (1 - t);
      const lineOn = look.era === null ? 0 : look.lineOpacity * t;
      const lineOff = previous.era === null ? 0 : previous.lineOpacity * (1 - t);

      if (fillOn > 0 || fillOff > 0 || lineOn > 0 || lineOff > 0) {
        g.bindVertexArray(null);
        g.activeTexture(g.TEXTURE0 + TEXTURE_UNIT);
        g.bindTexture(g.TEXTURE_2D, table);
        g.enableVertexAttribArray(0);
        g.enableVertexAttribArray(1);

        // Each pixel takes one lot, once: the first to reach it. So lots that overlap, and the
        // margins of neighbouring tiles, don't darken by blending over each other.
        g.useProgram(fill.program);
        g.uniform1i(fill.at("u_table"), TEXTURE_UNIT);
        g.uniform4fv(fill.at("u_palette"), palette);
        g.uniform1i(fill.at("u_row"), lotRow(eraOr0(look)));
        g.uniform1i(fill.at("u_prev"), lotRow(eraOr0(previous)));
        g.uniform1f(fill.at("u_opacity"), fillOn);
        g.uniform1f(fill.at("u_prev_opacity"), fillOff);
        g.enable(g.STENCIL_TEST);
        g.stencilMask(255);
        g.clearStencil(0);
        g.clear(g.STENCIL_BUFFER_BIT);
        g.stencilFunc(g.EQUAL, 0, 255);
        g.stencilOp(g.KEEP, g.KEEP, g.INCR);
        drawBuckets(g, args, fill.at("u_matrix"), LOTS_LAYER, "fill-color", (s, vertices, numbers) => {
          g.bindBuffer(g.ARRAY_BUFFER, vertices);
          g.vertexAttribPointer(0, 2, g.SHORT, false, 4, s.vertexOffset * 4);
          g.bindBuffer(g.ARRAY_BUFFER, numbers);
          g.vertexAttribPointer(1, 2, g.FLOAT, false, 8, s.vertexOffset * 8);
        });
        g.disable(g.STENCIL_TEST);

        g.enableVertexAttribArray(2);
        g.useProgram(line.program);
        g.uniform1i(line.at("u_table"), TEXTURE_UNIT);
        g.uniform4fv(line.at("u_edge"), edgeColor);
        g.uniform1i(line.at("u_row"), lineRow(eraOr0(look)));
        g.uniform1i(line.at("u_prev"), lineRow(eraOr0(previous)));
        g.uniform1f(line.at("u_opacity"), lineOn);
        g.uniform1f(line.at("u_prev_opacity"), lineOff);
        g.uniform1f(line.at("u_width"), lineWidth(zoom));
        g.uniform1f(line.at("u_dpr"), window.devicePixelRatio || 1);
        drawBuckets(g, args, line.at("u_matrix"), LINES_LAYER, "line-color", (s, vertices, numbers) => {
          // MapLibre's line vertex: position and normal, then four data bytes, then an offset we skip.
          g.bindBuffer(g.ARRAY_BUFFER, vertices);
          g.vertexAttribIPointer(0, 2, g.SHORT, 10, s.vertexOffset * 10);
          g.vertexAttribIPointer(1, 4, g.UNSIGNED_BYTE, 10, s.vertexOffset * 10 + 4);
          g.bindBuffer(g.ARRAY_BUFFER, numbers);
          g.vertexAttribPointer(2, 2, g.FLOAT, false, 8, s.vertexOffset * 8);
        }, (coord) => {
          g.uniform1f(line.at("u_ratio"), (TILE_PX * 2 ** (zoom - coord.canonical.z)) / EXTENT);
        });
        g.disableVertexAttribArray(0);
        g.disableVertexAttribArray(1);
        g.disableVertexAttribArray(2);
        g.bindBuffer(g.ELEMENT_ARRAY_BUFFER, null);
      }
      if (t < 1) map.triggerRepaint();
    },
  };
  map.addLayer(layer, before);

  return {
    /** Show this look, fading from what's there over `ms`. */
    show(next: LookupLook, ms: number) {
      if (next.era === look.era && next.fillOpacity === look.fillOpacity && next.lineOpacity === look.lineOpacity) return;
      previous = look;
      look = next;
      startedAt = performance.now();
      duration = ms;
      map.triggerRepaint();
    },
  };
}
