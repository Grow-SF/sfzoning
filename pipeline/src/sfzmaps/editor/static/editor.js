/* sfzmaps control-point editor. Talks to the FastAPI app in editor/server.py (same origin).
 *
 * Left pane: the scan (Leaflet, L.CRS.Simple over the Deep Zoom tile endpoint).
 * Right pane: modern blocks (MapLibre; GeoJSON from /blocks.geojson) plus the preview overlay.
 * World coordinates are EPSG:2227 (US ft); the map pane works in lon/lat, converted by proj4.
 */
"use strict";

const TOL_GREEN = 7, TOL_AMBER = 15;           // residual colours; same as the pass rule
const PASS_P50 = 7, PASS_P95 = 15;
const DEBOUNCE_MS = 400;
const COL = { green: "#2e9e4f", amber: "#e0a21b", red: "#d6342c", blue: "#2f7de1", grey: "#8b8f96", none: "#4b5563" };

proj4.defs("EPSG:2227", "+proj=lcc +lat_0=36.5 +lon_0=-120.5 +lat_1=38.4333333333333 +lat_2=37.0666666666667 " +
  "+x_0=2000000.0001016 +y_0=500000.0001016 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs");
const toLL = (x, y) => proj4("EPSG:2227", "EPSG:4326", [x, y]);
const fromLL = (lng, lat) => proj4("EPSG:4326", "EPSG:2227", [lng, lat]);

const $ = (id) => document.getElementById(id);
const S = {                       // editor state
  series: null, sheet: null, data: null,
  gcps: [], areas: [], baseSig: "", history: [],
  sel: null, mode: null, pending: null, drawPts: [],
  baseScore: null, score: null, previewSeq: 0, previewTimer: null, busy: 0,
  queue: [], zmax: 0, scale: 1, worst: [], worstIdx: -1, saving: false, autofill: false,
};
let scanMap, scanLayer, scanMarkers = [], scanMatchLayer, scanPending, scanWorst;
let map, mapReady = null, mapMarkers = [];

/* ---------- small helpers ---------- */
const clone = (o) => JSON.parse(JSON.stringify(o));
function toast(msg, err) {
  const t = $("toast");
  t.textContent = msg; t.className = err ? "err" : ""; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, err ? 7000 : 2500);
}
function modal(title, body, buttons) {
  return new Promise((resolve) => {
    $("modal-title").textContent = title; $("modal-body").textContent = body;
    const box = $("modal-btns"); box.innerHTML = "";
    buttons.forEach((b) => {
      const el = document.createElement("button");
      el.textContent = b.label; if (b.primary) el.className = "primary";
      el.onclick = () => { $("modal-back").hidden = true; resolve(b.value); };
      box.appendChild(el);
    });
    $("modal-back").hidden = false;
    box.querySelector("button.primary, button").focus();
  });
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let d = r.statusText;
    try { const j = await r.json(); d = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* keep */ }
    throw new Error(d);
  }
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const fmt = (v) => (v == null || !isFinite(v) ? "–" : v.toFixed(1));
function residColor(r, g) {
  if (g && !g.enabled) return null;
  if (r == null) return COL.none;
  return r < TOL_GREEN ? COL.green : r < TOL_AMBER ? COL.amber : COL.red;
}
function matchColor(m) {
  if (m.label === "real_change") return COL.blue;
  if (m.label === "no_equivalent") return COL.grey;
  return residColor(m.dist_ft);
}
const cleanGcps = () => S.gcps.map((g) => ({ px: g.px, py: g.py, wx: g.wx, wy: g.wy, source: g.source, enabled: g.enabled }));
const areasFC = () => ({ type: "FeatureCollection", features: clone(S.areas) });
const signature = () => JSON.stringify([cleanGcps(), S.areas]);
const isDirty = () => S.data && signature() !== S.baseSig;

/* ---------- scan pane (Leaflet) ---------- */
const ll2px = (ll) => [ll.lng * S.scale, -ll.lat * S.scale];
const px2ll = (px, py) => L.latLng(-py / S.scale, px / S.scale);

const DZLayer = L.TileLayer.extend({
  createTile(coords, done) {
    return L.TileLayer.prototype.createTile.call(this, coords, (err, tile) => {
      // Deep Zoom edge tiles are smaller than 256 px, not padded: size them by their pixels.
      if (!err && tile.naturalWidth) { tile.style.width = tile.naturalWidth + "px"; tile.style.height = tile.naturalHeight + "px"; }
      done(err, tile);
    });
  },
});

function initScan() {
  scanMap = L.map("scan", { crs: L.CRS.Simple, minZoom: -4, maxZoom: 14, zoomSnap: 0.25, zoomControl: true,
    attributionControl: false, doubleClickZoom: false });
  scanMap.on("click", onScanClick);
}

function loadScan(d) {
  S.zmax = d.image_max_zoom; S.scale = Math.pow(2, S.zmax);
  if (scanLayer) scanMap.removeLayer(scanLayer);
  const { width: w, height: h } = d.image_size;
  const bounds = L.latLngBounds(px2ll(0, h), px2ll(w, 0));
  scanLayer = new DZLayer(d.image_url, { tileSize: d.image_tile_size, minNativeZoom: 0, maxNativeZoom: S.zmax,
    minZoom: -4, maxZoom: S.zmax + 3, bounds, noWrap: true, keepBuffer: 4 }).addTo(scanMap);
  scanMap.setMinZoom(S.zmax - 5); scanMap.setMaxZoom(S.zmax + 3);
  scanMap.setMaxBounds(bounds.pad(0.5));
  scanMap.fitBounds(bounds);
  if (scanMatchLayer) scanMap.removeLayer(scanMatchLayer);
  const canvas = L.canvas({ padding: 0.3 });
  scanMatchLayer = L.layerGroup().addTo(scanMap);
  (d.matches || []).forEach((m) => {
    if (m.px == null) return;
    L.circleMarker(px2ll(m.px, m.py), { renderer: canvas, radius: 3, weight: 0, fillOpacity: 0.85,
      fillColor: matchColor(m), interactive: false }).addTo(scanMatchLayer);
  });
}

/* ---------- map pane (MapLibre) ---------- */
function initMap() {
  map = new maplibregl.Map({
    container: "map", attributionControl: false, doubleClickZoom: false,
    style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#f7f6f2" } }] },
    center: [-122.43, 37.76], zoom: 12,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  mapReady = new Promise((res) => map.on("load", () => {
    const empty = { type: "FeatureCollection", features: [] };
    map.addSource("streets", { type: "geojson", data: empty });
    map.addSource("blocks", { type: "geojson", data: empty });
    map.addSource("matches", { type: "geojson", data: empty });
    map.addSource("areas", { type: "geojson", data: empty });
    map.addSource("draw", { type: "geojson", data: empty });
    map.addSource("worst", { type: "geojson", data: empty });
    map.addLayer({ id: "streets", type: "line", source: "streets", paint: { "line-color": "#b9b6ad", "line-width": 1 } });
    map.addLayer({ id: "blocks-fill", type: "fill", source: "blocks", paint: { "fill-color": "#e8e5da", "fill-opacity": 0.5 } });
    map.addLayer({ id: "blocks-line", type: "line", source: "blocks", paint: { "line-color": "#555", "line-width": 1.2 } });
    // preview overlay is inserted by setOverlay (above blocks fill, below outlines is not needed)
    map.addLayer({ id: "areas-fill", type: "fill", source: "areas", paint: { "fill-color": COL.blue, "fill-opacity": 0.2 } });
    map.addLayer({ id: "areas-line", type: "line", source: "areas", paint: { "line-color": COL.blue, "line-width": 2, "line-dasharray": [2, 1] } });
    map.addLayer({ id: "draw-fill", type: "fill", source: "draw", filter: ["==", "$type", "Polygon"], paint: { "fill-color": COL.amber, "fill-opacity": 0.2 } });
    map.addLayer({ id: "draw-line", type: "line", source: "draw", paint: { "line-color": COL.amber, "line-width": 2 } });
    map.addLayer({ id: "draw-pts", type: "circle", source: "draw", filter: ["==", "$type", "Point"], paint: { "circle-radius": 4, "circle-color": COL.amber } });
    map.addLayer({ id: "matches", type: "circle", source: "matches", paint: {
      "circle-radius": 3, "circle-color": ["get", "c"], "circle-opacity": 0.9 } });
    map.addLayer({ id: "worst", type: "circle", source: "worst", paint: {
      "circle-radius": 34, "circle-color": "#ffffff", "circle-opacity": 0, "circle-stroke-color": COL.red, "circle-stroke-width": 3 } });
    map.on("click", onMapClick);
    map.on("dblclick", (e) => { if (S.mode === "area") { e.preventDefault(); finishArea(true); } });
    res();
  }));
}

function setSource(id, data) { const s = map.getSource(id); if (s) s.setData(data); }

async function loadMapData(d) {
  await mapReady;
  const base = `/api/sheet/${S.series}/${S.sheet}`;
  setSource("blocks", { type: "FeatureCollection", features: [] });
  setSource("streets", { type: "FeatureCollection", features: [] });
  api(`${base}/blocks.geojson`).then((fc) => { if (S.sheet === d.sheet_id) setSource("blocks", fc); }).catch((e) => toast("blocks: " + e.message, true));
  api(`${base}/blocks.geojson?layer=streets`).then((fc) => { if (S.sheet === d.sheet_id) setSource("streets", fc); }).catch(() => {});
  setSource("matches", { type: "FeatureCollection", features: (d.matches || []).filter((m) => m.wx != null).map((m) => ({
    type: "Feature", properties: { c: matchColor(m) }, geometry: { type: "Point", coordinates: toLL(m.wx, m.wy) } })) });
  setOverlay(null);
  fitMap();
}

function fitMap() {
  const pts = S.gcps.map((g) => toLL(g.wx, g.wy));
  if (pts.length) {
    const b = new maplibregl.LngLatBounds(pts[0], pts[0]);
    pts.forEach((p) => b.extend(p));
    map.fitBounds(b, { padding: 60, duration: 0, maxZoom: 18 });
  } else if (S.data.world_bounds_wgs84) {
    const w = S.data.world_bounds_wgs84; map.fitBounds([[w[0], w[1]], [w[2], w[3]]], { padding: 40, duration: 0 });
  }
}

function setOverlay(res) {
  if (!map.getStyle()) return;
  if (!res) { if (map.getLayer("preview")) map.setLayoutProperty("preview", "visibility", "none"); return; }
  const [x0, y0, x1, y1] = res.preview_bounds;
  const coordinates = [toLL(x0, y1), toLL(x1, y1), toLL(x1, y0), toLL(x0, y0)];   // TL TR BR BL
  const src = map.getSource("preview");
  if (src) src.updateImage({ url: res.preview_png_url, coordinates });
  else {
    map.addSource("preview", { type: "image", url: res.preview_png_url, coordinates });
    map.addLayer({ id: "preview", type: "raster", source: "preview", paint: { "raster-opacity": $("ov-op").value / 100, "raster-fade-duration": 0 } }, "areas-fill");
  }
  map.setLayoutProperty("preview", "visibility", $("ov-on").checked ? "visible" : "none");
}

/* ---------- control-point markers ---------- */
function markerEl(i) {
  const el = document.createElement("div");
  el.className = "gcp"; el.textContent = String(i + 1);
  return el;
}
function rebuildMarkers() {
  scanMarkers.forEach((m) => scanMap.removeLayer(m)); scanMarkers = [];
  mapMarkers.forEach((m) => m.remove()); mapMarkers = [];
  S.gcps.forEach((g, i) => {
    const sm = L.marker(px2ll(g.px, g.py), { draggable: true, icon: L.divIcon({ className: "gcpicon", html: '<div class="gcp">' + (i + 1) + "</div>", iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(scanMap);
    sm.on("click", (e) => { L.DomEvent.stopPropagation(e); setSel(i); });
    sm.on("dragstart", () => setSel(i));
    sm.on("dragend", () => { pushHistory(); const [px, py] = ll2px(sm.getLatLng()); g.px = px; g.py = py; g._r = null; edited(); });
    scanMarkers.push(sm);
    const el = markerEl(i);
    const mm = new maplibregl.Marker({ element: el, draggable: true }).setLngLat(toLL(g.wx, g.wy)).addTo(map);
    el.addEventListener("mousedown", () => setSel(i));
    mm.on("dragend", () => { pushHistory(); const ll = mm.getLngLat(); const [wx, wy] = fromLL(ll.lng, ll.lat); g.wx = wx; g.wy = wy; g._r = null; edited(); });
    mapMarkers.push(mm);
  });
  refreshStyles();
}
function gcpEls(i) {
  const a = scanMarkers[i] && scanMarkers[i].getElement().firstChild;
  const b = mapMarkers[i] && mapMarkers[i].getElement();
  return [a, b].filter(Boolean);
}
function refreshStyles() {
  S.gcps.forEach((g, i) => {
    const c = residColor(g._r, g);
    gcpEls(i).forEach((el) => {
      el.style.setProperty("--c", c || "#fff");
      el.classList.toggle("off", !g.enabled);
      el.classList.toggle("sel", S.sel === i);
      el.title = `#${i + 1} ${g.source}${g._r != null ? ", " + g._r.toFixed(1) + " ft" : ""}${g.enabled ? "" : " (disabled)"}`;
    });
    if (scanMarkers[i]) scanMarkers[i].setZIndexOffset(S.sel === i ? 1000 : 0);
  });
  renderTable();
  $("btn-dis").textContent = S.sel != null && S.gcps[S.sel] && !S.gcps[S.sel].enabled ? "Enable (D)" : "Disable (D)";
}
function renderTable() {
  const tb = document.querySelector("#gcp-table tbody"); tb.innerHTML = "";
  S.gcps.forEach((g, i) => {
    const tr = document.createElement("tr");
    tr.className = (S.sel === i ? "sel " : "") + (g.enabled ? "" : "off");
    const c = residColor(g._r, g) || "#fff";
    tr.innerHTML = `<td><span class="dot" style="background:${c};${g.enabled ? "" : "border:1px dashed #888"}"></span>${i + 1}</td><td>${g.source}</td><td>${fmt(g._r)}</td>`;
    tr.onclick = () => setSel(i);
    tb.appendChild(tr);
  });
  const ul = $("area-list"); ul.innerHTML = "";
  S.areas.forEach((a, i) => {
    const li = document.createElement("li"); li.textContent = "Area " + (i + 1);
    const b = document.createElement("button"); b.textContent = "×"; b.title = "Remove this area";
    b.onclick = () => { pushHistory(); S.areas.splice(i, 1); edited(); };
    li.appendChild(b); ul.appendChild(li);
  });
}
function setSel(i) { S.sel = i; refreshStyles(); }

/* ---------- editing ---------- */
function pushHistory() { S.history.push(JSON.stringify([S.gcps, S.areas, S.autofill])); if (S.history.length > 100) S.history.shift(); }
function undo() {
  const h = S.history.pop(); if (!h) return;
  let af; [S.gcps, S.areas, af] = JSON.parse(h); S.autofill = !!af;
  S.sel = null; S.pending = null; clearPending();
  rebuildMarkers(); renderAreas(); afterEdit();
}
function edited() { renderAreas(); refreshStyles(); afterEdit(); }
function afterEdit() { updateDirty(); schedulePreview(); updateAutofillBtn(); }
function updateAutofillBtn() {
  const b = $("tool-autofill"); if (!b) return;
  const n = S.gcps.filter((g) => g.enabled).length;
  b.disabled = n < 4 || !!S.autofilling;
  b.title = n < 4 ? "Needs at least 4 enabled points" : "Seed the automatic matcher from your points";
}
async function autofill() {
  if (S.autofilling || S.gcps.filter((g) => g.enabled).length < 4) return;
  S.autofilling = true; clearTimeout(S.previewTimer); S.previewSeq++;
  $("busy").textContent = "Auto-filling…"; updateAutofillBtn();
  pushHistory();
  try {
    const r = await post(`/api/sheet/${S.series}/${S.sheet}/autofill`, { gcps: cleanGcps(), change_areas: areasFC() });
    S.gcps = r.gcps.map((g) => ({ px: g.px, py: g.py, wx: g.wx, wy: g.wy, source: g.source, enabled: g.enabled !== false, _r: g.residual_ft }));
    S.autofill = true; S.score = r.score; S.sel = null;
    await mapReady;
    setOverlay(r); rebuildMarkers(); renderScore(); edited();
    toast(`Auto-fill: ${r.n_matches} matched corners. Review, then Save.`);
  } catch (e) {
    S.history.pop();
    toast("Auto-fill failed: " + e.message, true);
  } finally {
    S.autofilling = false; $("busy").textContent = ""; updateAutofillBtn();
  }
}
function renderAreas() {
  setSource("areas", { type: "FeatureCollection", features: S.areas.map((a) => ({ type: "Feature", properties: {},
    geometry: { type: "Polygon", coordinates: a.geometry.coordinates.map((r) => r.map(([x, y]) => toLL(x, y))) } })) });
  renderTable();
}
function deleteSel() {
  if (S.sel == null) return;
  pushHistory(); S.gcps.splice(S.sel, 1); S.sel = null; rebuildMarkers(); afterEdit();
}
function toggleSel() {
  if (S.sel == null) return;
  pushHistory(); const g = S.gcps[S.sel]; g.enabled = !g.enabled; g._r = null; refreshStyles(); afterEdit();
}

function setMode(m) {
  if (m === S.mode) m = null;
  S.mode = m; S.pending = null; clearPending(); cancelDraw();
  document.body.classList.toggle("adding-scan", m === "add");
  document.body.classList.toggle("adding-map", false);
  document.body.classList.toggle("drawing", m === "area");
  $("tool-add").classList.toggle("active", m === "add");
  $("tool-area").classList.toggle("active", m === "area");
  $("hint").textContent = m === "add" ? "Click the scan where the point is, then the same spot on the map. Esc to finish."
    : m === "area" ? "Click map vertices. Double-click or Enter to finish, Esc to cancel." : "";
}
function clearPending() { if (scanPending) { scanMap.removeLayer(scanPending); scanPending = null; } }
function onScanClick(e) {
  if (S.mode !== "add" || S.pending) return;
  const [px, py] = ll2px(e.latlng);
  S.pending = { px, py };
  scanPending = L.marker(e.latlng, { interactive: false, icon: L.divIcon({ className: "gcpicon", html: '<div class="gcp pending">+</div>', iconSize: [22, 22], iconAnchor: [11, 11] }) }).addTo(scanMap);
  document.body.classList.remove("adding-scan"); document.body.classList.add("adding-map");
  $("hint").textContent = "Now click the matching spot on the map.";
}
function onMapClick(e) {
  if (S.mode === "add" && S.pending) {
    const [wx, wy] = fromLL(e.lngLat.lng, e.lngLat.lat);
    pushHistory();
    S.gcps.push({ px: S.pending.px, py: S.pending.py, wx, wy, source: "manual", enabled: true, _r: null });
    S.pending = null; clearPending(); S.sel = S.gcps.length - 1;
    document.body.classList.remove("adding-map"); document.body.classList.add("adding-scan");
    $("hint").textContent = "Added. Click the scan for another point, Esc to finish.";
    rebuildMarkers(); afterEdit();
  } else if (S.mode === "area") {
    S.drawPts.push([e.lngLat.lng, e.lngLat.lat]); drawTemp();
  }
}
function drawTemp() {
  const f = S.drawPts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
  if (S.drawPts.length >= 3) f.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...S.drawPts, S.drawPts[0]]] } });
  else if (S.drawPts.length === 2) f.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: S.drawPts } });
  setSource("draw", { type: "FeatureCollection", features: f });
}
function cancelDraw() { S.drawPts = []; if (map && map.getSource("draw")) setSource("draw", { type: "FeatureCollection", features: [] }); }
function finishArea(fromDbl) {
  let pts = S.drawPts.slice();
  if (fromDbl && pts.length > 1) { const a = pts[pts.length - 1], b = pts[pts.length - 2]; if (a[0] === b[0] && a[1] === b[1]) pts.pop(); }
  if (pts.length < 3) { toast("A change area needs at least 3 points.", true); return; }
  const ring = pts.map(([lng, lat]) => fromLL(lng, lat)); ring.push(ring[0]);
  pushHistory();
  S.areas.push({ type: "Feature", properties: { kind: "real_change" }, geometry: { type: "Polygon", coordinates: [ring] } });
  cancelDraw(); setMode(null); edited();
}

/* ---------- preview + score ---------- */
function updateDirty() {
  const d = isDirty();
  $("status").textContent = (S.data ? S.data.status : "") + (d ? " · unsaved edits" : "");
  $("status").classList.toggle("dirty", !!d);
}
function schedulePreview(delay) {
  clearTimeout(S.previewTimer);
  if (!S.gcps.filter((g) => g.enabled).length) return;
  $("busy").textContent = "updating…";
  S.previewTimer = setTimeout(runPreview, delay == null ? DEBOUNCE_MS : delay);
}
async function runPreview(initial) {
  const seq = ++S.previewSeq, sheet = S.sheet, series = S.series;
  try {
    const r = await post(`/api/sheet/${series}/${sheet}/preview`, { gcps: cleanGcps(), change_areas: areasFC(), autofill: S.autofill });
    if (seq !== S.previewSeq || sheet !== S.sheet) return;               // a newer edit superseded this
    await mapReady;
    setOverlay(r);
    if (initial !== true) {
      S.gcps.forEach((g, i) => { g._r = r.residuals[i]; });
      S.score = r.score; renderScore(); refreshStyles();
    }
    $("busy").textContent = "";
  } catch (e) {
    if (seq !== S.previewSeq) return;
    $("busy").textContent = "";
    toast("Preview failed: " + e.message, true);
  }
}
function renderScore() {
  const sc = S.score, base = S.baseScore;
  const set = (id, v, lim, was) => {
    const el = $(id); el.querySelector(".v").textContent = fmt(v);
    el.classList.toggle("ok", v != null && v <= lim); el.classList.toggle("bad", v == null || v > lim);
    el.querySelector(".rule").textContent = `≤ ${fmt(lim)} ft` + (was != null && was !== v && isFinite(was) ? ` · was ${fmt(was)}` : "");
  };
  // the limits the pipeline actually applied (scale-aware: PASS x k on coarse scans)
  const lim = (k, dflt) => (sc && sc[k] != null ? +sc[k] : dflt);
  set("m-p50", sc && sc.holdout_p50_ft, lim("pass_median_limit_ft", PASS_P50), base && base.holdout_p50_ft);
  set("m-p95", sc && sc.holdout_p95_ft, lim("pass_p95_limit_ft", PASS_P95), base && base.holdout_p95_ft);
  const v = $("verdict"); v.textContent = !sc ? "–" : sc.passed ? "PASS" : "FAIL";
  v.className = "verdict" + (sc ? (sc.passed ? " ok" : " bad") : "");
  $("reason").textContent = sc && sc.reason ? sc.reason : "";
}

/* ---------- worst areas ---------- */
function computeWorst(matches) {
  const CELL = 600, cells = new Map();
  (matches || []).forEach((m) => {
    if ((m.label !== "inlier" && m.label !== "outlier") || m.dist_ft == null || m.px == null) return;
    const k = Math.floor(m.wx / CELL) + "," + Math.floor(m.wy / CELL);
    const c = cells.get(k) || { n: 0, score: 0, px: 0, py: 0, wx: 0, wy: 0 };
    c.n++; c.score += Math.max(0, m.dist_ft - TOL_GREEN); c.px += m.px; c.py += m.py; c.wx += m.wx; c.wy += m.wy;
    cells.set(k, c);
  });
  return [...cells.values()].filter((c) => c.score > 0).sort((a, b) => b.score - a.score).slice(0, 5)
    .map((c) => ({ px: c.px / c.n, py: c.py / c.n, wx: c.wx / c.n, wy: c.wy / c.n, n: c.n, score: c.score }));
}
function jumpWorst(dir) {
  if (!S.worst.length) { toast(`No residual clusters above ${PASS_P50} ft.`); return; }
  S.worstIdx = (S.worstIdx + dir + S.worst.length) % S.worst.length;
  const w = S.worst[S.worstIdx];
  scanMap.setView(px2ll(w.px, w.py), Math.max(scanMap.getZoom(), S.zmax - 2));
  map.flyTo({ center: toLL(w.wx, w.wy), zoom: Math.max(map.getZoom(), 16), duration: 300 });
  if (scanWorst) scanMap.removeLayer(scanWorst);
  scanWorst = L.circleMarker(px2ll(w.px, w.py), { radius: 34, color: COL.red, weight: 3, fill: false, interactive: false }).addTo(scanMap);
  setSource("worst", { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: toLL(w.wx, w.wy) } }] });
  $("worst-n").textContent = `${S.worstIdx + 1}/${S.worst.length}`;
}

/* ---------- sheets, queue, save ---------- */
async function loadQueue(keepSelection) {
  S.queue = await api("/api/queue");
  const sel = $("sheet-select"); sel.innerHTML = "";
  S.queue.forEach((r) => {
    const o = document.createElement("option"); o.value = r.series + "/" + r.sheet_id;
    o.textContent = `${r.series}/${r.sheet_id} · ${r.status}` + (r.holdout_p50_ft != null ? ` · ${r.holdout_p50_ft.toFixed(1)} ft` : "");
    sel.appendChild(o);
  });
  if (keepSelection && S.series) sel.value = S.series + "/" + S.sheet;
}
async function openSheet(series, sheet) {
  const d = await api(`/api/sheet/${series}/${sheet}`);
  clearTimeout(S.previewTimer); S.previewSeq++;
  S.series = series; S.sheet = sheet; S.data = d; S.history = []; S.autofill = false; S.sel = null; S.worstIdx = -1;
  setMode(null);
  S.gcps = d.gcps.map((g) => ({ px: g.px, py: g.py, wx: g.wx, wy: g.wy, source: g.source, enabled: g.enabled !== false, _r: g.residual_ft }));
  S.areas = ((d.change_areas && d.change_areas.features) || []).filter((f) => f.geometry && f.geometry.type === "Polygon");
  S.baseScore = d.score; S.score = d.score;
  S.baseSig = signature(); updateAutofillBtn();
  $("note").value = d.note || "";
  $("credit").textContent = d.credit || "";
  $("sheet-select").value = series + "/" + sheet;
  loadScan(d);
  await loadMapData(d);
  rebuildMarkers(); renderAreas(); renderScore(); updateDirty();
  S.worst = computeWorst(d.matches); $("worst-n").textContent = S.worst.length ? `0/${S.worst.length}` : "";
  if (scanWorst) { scanMap.removeLayer(scanWorst); scanWorst = null; } setSource("worst", { type: "FeatureCollection", features: [] });
  if (S.gcps.length) runPreview(true);
}
async function guardDirty() {
  if (!isDirty()) return true;
  const v = await modal("Unsaved edits", "This sheet has edits that are not saved. Discard them?", [
    { label: "Keep editing", value: "keep", primary: true }, { label: "Discard edits", value: "discard" }]);
  return v === "discard";
}
async function onPickSheet() {
  const v = $("sheet-select").value;
  if (!(await guardDirty())) { $("sheet-select").value = S.series + "/" + S.sheet; return; }
  const [series, sheet] = v.split("/");
  try { await openSheet(series, sheet); } catch (e) { toast(e.message, true); }
}
async function save(status) {
  if (S.saving || !S.data) return;
  const note = $("note").value.trim();
  if (status === "accepted" && !(S.score && S.score.passed) && !note) {
    toast("Accepting a sheet that does not pass needs a note.", true); $("note").focus(); return;
  }
  S.saving = true; clearTimeout(S.previewTimer);
  $("btn-save").disabled = $("btn-accept").disabled = true; $("busy").textContent = "saving (re-running sheet)…";
  try {
    const r = await post(`/api/sheet/${S.series}/${S.sheet}/save`, { gcps: cleanGcps(), change_areas: areasFC(), status, note, autofill: S.autofill });
    toast(`Saved as ${r.status}: ${r.score.passed ? "PASS" : "FAIL"}, held-out median ${fmt(r.score.holdout_p50_ft)} ft, p95 ${fmt(r.score.holdout_p95_ft)} ft`);
    await loadQueue(true);
    await openSheet(S.series, S.sheet);
  } catch (e) { toast("Save failed: " + e.message, true); }
  finally { S.saving = false; $("btn-save").disabled = $("btn-accept").disabled = false; $("busy").textContent = ""; }
}

/* ---------- wiring ---------- */
function typing(e) { return /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && e.target.type !== "range" && e.target.type !== "checkbox"; }
document.addEventListener("keydown", (e) => {
  if (!$("modal-back").hidden) return;
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save("edited"); return; }
  if (typing(e)) return;
  if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); deleteSel(); }
  else if (e.key === "d" || e.key === "D") toggleSel();
  else if (e.key === "Escape") { if (S.pending) { S.pending = null; clearPending(); } else setMode(null); }
  else if (e.key === "Enter" && S.mode === "area") finishArea(false);
});
window.addEventListener("beforeunload", (e) => { if (isDirty()) { e.preventDefault(); e.returnValue = ""; } });

async function main() {
  initScan(); initMap();
  $("sheet-select").onchange = onPickSheet;
  $("btn-save").onclick = () => save("edited");
  $("btn-accept").onclick = () => save("accepted");
  $("tool-add").onclick = () => setMode("add");
  $("tool-area").onclick = () => setMode("area");
  $("tool-autofill").onclick = autofill;
  $("worst-next").onclick = () => jumpWorst(1);
  $("worst-prev").onclick = () => jumpWorst(-1);
  $("btn-undo").onclick = undo; $("btn-del").onclick = deleteSel; $("btn-dis").onclick = toggleSel;
  $("ov-on").onchange = () => { if (map.getLayer("preview")) map.setLayoutProperty("preview", "visibility", $("ov-on").checked ? "visible" : "none"); };
  $("ov-op").oninput = () => { if (map.getLayer("preview")) map.setPaintProperty("preview", "raster-opacity", $("ov-op").value / 100); };
  $("note").addEventListener("input", updateDirty);
  try {
    await loadQueue(false);
    // ?series=&sheet= (the review page's "Open in editor" link) selects that sheet
    const qs = new URLSearchParams(window.location.search);
    const want = { series: qs.get("series"), sheet: qs.get("sheet") };
    if (want.series && want.sheet) {
      try { await openSheet(want.series, want.sheet); return; }
      catch (e) { toast(`Cannot open ${want.series}/${want.sheet}: ${e.message}`, true); }
    }
    if (S.queue.length) { const q = S.queue[0]; await openSheet(q.series, q.sheet_id); }
    else toast("The queue is empty.");
  } catch (e) { toast(e.message, true); }
}
window.__ed = { S, get scanMap() { return scanMap; }, get map() { return map; }, toLL, fromLL, ll2px, px2ll };   // for scripted checks
main();
