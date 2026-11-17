import { capturing, reduced, cursor, initReveals, magnetic, marquee, setRangeFill, splitWords, whenVisible, countTo, debounce, slidingPill } from "./fx.js";
import { render, CONFIG } from "./charts.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const CAP = capturing ? params.get("capture") : null;
if (capturing) document.documentElement.classList.add("capture");
if (CAP) document.getElementById(CAP)?.classList.add("cap-target");

const MC = { cart: "#3987e5", forest: "#199e70", gboost: "#e0621f", gam: "#9085e9", crowd: "#c98500", stack: "#d55181" };
const GREY = "#a7aeb6";
const INK = "#eef0f2", INK2 = "#c3c8ce", MUTED = "#878e97", GRID = "#1f2328", AXIS = "#363c44", SURF = "#121417", ACC = "#ffcf33";
const RISK = [[0, "#1c1f23"], [0.45, "#a84a17"], [1, "#ffc29a"]];
const DIV = [[0, "#3987e5"], [0.5, "#2a2e33"], [1, "#e0621f"]];
const EPI = { lon: 84.731, lat: 28.231 };          // USGS epicentre of the 25 April 2015 main shock
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const num = (v) => Math.round(v).toLocaleString("en-US");
const sgn = (v, d = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const col = (k) => MC[k] || GREY;
const ax = (title = "", extra = {}) => ({ title: { text: title, font: { color: MUTED, size: 11 } }, gridcolor: GRID, linecolor: AXIS, zerolinecolor: AXIS, tickcolor: AXIS, tickfont: { size: 10.5 }, automargin: true, ...extra });
const lay = (x, y, extra = {}) => ({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "Inter, system-ui, sans-serif", color: INK2, size: 11 }, margin: { l: 44, r: 10, t: 10, b: 36 },
  hoverlabel: { bgcolor: "#1d2126", bordercolor: "#454c55", font: { color: INK, size: 12 } }, showlegend: false, xaxis: ax(x), yaxis: ax(y), ...extra });
const graphCache = {};
const getG = (id) => (graphCache[id] ??= fetch(`/api/graphs/${id}`).then((r) => r.json()));
async function plotG(id, el, h, extra = {}) {
  const g = await getG(id);
  const L = g.figure.layout;
  await render(el, { data: g.figure.data, layout: { ...L, autosize: true, height: h, margin: { ...L.margin, t: L.showlegend === false ? 10 : 30 }, legend: { ...L.legend, font: { size: 10.5 } }, ...extra } });
}
const lazy = (el, fn) => whenVisible(el, fn, 0.1);

function riskColor(v) {
  v = Math.max(0, Math.min(1, v));
  const [a, b] = v <= RISK[1][0] ? [RISK[0], RISK[1]] : [RISK[1], RISK[2]];
  const t = (v - a[0]) / (b[0] - a[0]);
  const h = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const [x, y] = [h(a[1]), h(b[1])];
  return `rgb(${x.map((c, i) => Math.round(c + (y[i] - c) * t)).join(",")})`;
}

async function boot() {
  const [ov, W, opts] = await Promise.all(["/api/overview", "/api/wards", "/api/options"].map((u) => fetch(u).then((r) => r.json())));
  const ctx = { ov, W, opts, L: ov.labels, S: ov.scores, D: ov.data, HL: ov.labels[ov.headline] };
  buildHero(ctx);
  buildMap(ctx);
  buildChecklist(ctx);
  buildModels(ctx);
  buildCompare(ctx);
  buildFragility(ctx);
  buildHouse(ctx);
  buildGrades(ctx);
  buildLimits(ctx);
  buildInside(ctx);
  $("#nGraphs").textContent = ov.n_graphs; $("#nGraphs2").textContent = ov.n_graphs;
  splitWords($("#heroTitle"), capturing ? 0 : 1250);
  cursor();
  initReveals();
  $$("[data-magnetic]").forEach((b) => magnetic(b));
  initNav();
  finishLoader();
  $("#heroTitle").classList.add("in");
}

function finishLoader() {
  const l = $("#loader");
  if (capturing) { l.remove(); document.body.classList.remove("is-loading"); return; }
  setTimeout(() => { l.classList.add("done"); document.body.classList.remove("is-loading"); setTimeout(() => l.remove(), 1100); }, reduced ? 0 : 1350);
}

function seg(box, onPick) {
  const pill = $(".tabs__pill", box);
  const move = slidingPill(box, pill, "button.active");
  $$("button", box).forEach((b) => b.addEventListener("click", () => { $$("button", box).forEach((x) => x.classList.toggle("active", x === b)); move(); onPick(b.dataset.mode); }));
  requestAnimationFrame(move);
  return { set(mode) { $$("button", box).forEach((x) => x.classList.toggle("active", x.dataset.mode === mode)); move(); } };
}

// ====================================================================== HERO
function buildHero({ ov, W, D }) {
  const destroyed = D.grade_counts[4];
  $("#heroStats").innerHTML = `
    <div class="stat"><div class="stat__v">${(D.buildings / 1000).toFixed(0)}<small>k</small></div><div class="stat__l">houses surveyed after the quake</div></div>
    <div class="stat"><div class="stat__v">${(destroyed / 1000).toFixed(0)}<small>k</small></div><div class="stat__l">of them destroyed outright</div></div>
    <div class="stat"><div class="stat__v">10</div><div class="stat__l">models built from scratch and compared</div></div>
    <div class="stat"><div class="stat__v">${(D.n_test_unseen / 1000).toFixed(0)}<small>k</small></div><div class="stat__l">houses in 3 districts no model ever saw</div></div>`;
  const words = ["Stone in mud mortar", "CART", "Bagging", "Random forest", "AdaBoost", "Gradient boosting", "GAM", "Stacking", "Bayesian model averaging", "Fragility curves", "Gorkha 2015"];
  const track = $("#marquee");
  track.innerHTML = [...words, ...words].map((w) => `<span>${w}</span>`).join("");
  if (!reduced) marquee(track);
  startQuake(W.wards);
}

/** Every ward on a canvas. An S-wave front (3.5 km/s) spreads from the epicentre and each ward lights up with its real damage. */
function startQuake(wards) {
  const c = $("#quakeCanvas"), g = c.getContext("2d"), clock = $("#quakeClock");
  const lon0 = 84.25, lon1 = 86.85, lat0 = 26.95, lat1 = 28.75, kx = Math.cos((27.8 * Math.PI) / 180);
  const km = (a, b) => { const R = 6371, r = Math.PI / 180, dl = (b.lat - a.lat) * r, dn = (b.lon - a.lon) * r; const h = Math.sin(dl / 2) ** 2 + Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(dn / 2) ** 2; return 2 * R * Math.asin(Math.sqrt(h)); };
  const pts = wards.map((w) => ({ ...w, d: km(EPI, w), r: Math.max(1.6, Math.min(7, Math.sqrt(w.n) / 6)), col: riskColor(w.actual) }));
  const maxD = Math.max(...pts.map((p) => p.d)), SPEED = 3.5, RUN = 7000, HOLD = 3200;
  let W, H, s, ox, oy, visible = true;
  const proj = (lon, lat) => [ox + (lon - lon0) * kx * s, oy + (lat1 - lat) * s];
  const size = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = c.clientWidth; H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    s = Math.min((W - 24) / ((lon1 - lon0) * kx), (H - 50) / (lat1 - lat0)); ox = (W - (lon1 - lon0) * kx * s) / 2; oy = 10;
    draw(reduced ? RUN : performance.now() % (RUN + HOLD));
  };
  const kmPx = () => (s * 1) / 111;         // one degree of latitude is about 111 km
  function draw(t) {
    if (!W) return;
    g.clearRect(0, 0, W, H);
    const sec = Math.min(t, RUN) / RUN * (maxD / SPEED), front = sec * SPEED;
    const [ex, ey] = proj(EPI.lon, EPI.lat);
    for (const p of pts) {
      const [x, y] = proj(p.lon, p.lat);
      const since = front - p.d;
      if (since < 0) { g.globalAlpha = 0.18; g.fillStyle = "#3a4048"; g.beginPath(); g.arc(x, y, p.r * 0.7, 0, 7); g.fill(); continue; }
      const flash = Math.max(0, 1 - since / 18);
      g.globalAlpha = 0.95;
      g.fillStyle = p.col; g.beginPath(); g.arc(x, y, p.r * (1 + flash * 0.9), 0, 7); g.fill();
      if (flash > 0) { g.globalAlpha = flash * 0.7; g.strokeStyle = ACC; g.lineWidth = 1; g.beginPath(); g.arc(x, y, p.r * (1.4 + (1 - flash) * 2), 0, 7); g.stroke(); }
    }
    g.globalAlpha = 1;
    if (t < RUN) {
      for (let k = 0; k < 3; k++) {
        const rr = (front - k * 14) * kmPx();
        if (rr <= 0) continue;
        g.globalAlpha = 0.5 - k * 0.15; g.strokeStyle = ACC; g.lineWidth = 1.5 - k * 0.4;
        g.beginPath(); g.ellipse(ex, ey, rr * kx, rr, 0, 0, 7); g.stroke();
      }
      g.globalAlpha = 1;
    }
    g.fillStyle = ACC; g.beginPath();
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? 4 : 10; g.lineTo(ex + rr * Math.cos(a), ey + rr * Math.sin(a)); }
    g.closePath(); g.fill();
    g.font = "600 11px 'IBM Plex Mono', monospace"; g.fillStyle = INK; g.textAlign = "left"; g.textBaseline = "middle";
    g.fillText("Epicentre · Gorkha", ex + 14, ey - 2);
    clock.textContent = `S-wave · ${Math.round(sec)} s · ${Math.round(front)} km`;
  }
  new ResizeObserver(size).observe(c);
  if (reduced) return;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(c);
  const loop = (now) => { if (visible) draw(now % (RUN + HOLD)); requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}

// ====================================================================== MAP
function buildMap({ ov, W, D }) {
  const ws = W.wards, shifts = W.shifts, i0 = shifts.indexOf(0);
  const actualTotal = ov.districts.reduce((a, d) => a + d.collapsed, 0);
  const F = ov.ward_fit;
  $("#mapLead").textContent = `Each dot is one of ${ws.length} wards, sized by its houses. The forecast comes only from each house's construction and its ward's estimated shaking. In the three yellow-ringed districts it never saw, ward by ward it still matches reality with r = ${F.unseen.r.toFixed(2)}.`;
  let mode = "actual", si = i0;
  const lon = ws.map((w) => w.lon), lat = ws.map((w) => w.lat);
  const size = ws.map((w) => Math.max(4, Math.min(19, Math.sqrt(w.n) / 3.2)));
  const text = ws.map((w) => `${esc(w.place)} · ${esc(w.district)}${w.unseen ? " (held out)" : ""}`);
  const cd = ws.map((w) => [w.n, w.shaking, w.actual, w.model]);
  const lineCol = ws.map((w) => (w.unseen ? ACC : SURF)), lineW = ws.map((w) => (w.unseen ? 1.2 : 0.5));
  const districts = {};
  ws.forEach((w) => { const d = (districts[w.district] ??= { lon: 0, lat: 0, n: 0, unseen: w.unseen }); d.lon += w.lon; d.lat += w.lat; d.n += 1; });
  const labels = Object.entries(districts).map(([name, d]) => ({ x: d.lon / d.n, y: d.lat / d.n + 0.09, text: name, showarrow: false, font: { size: 11, color: d.unseen ? ACC : INK2, family: "IBM Plex Mono, monospace" } }));
  const el = $("#mapPlot");
  const values = () => (mode === "actual" ? ws.map((w) => w.actual) : mode === "model" ? ws.map((w) => w.shift[si]) : ws.map((w) => w.actual - w.model));
  const hover = "%{text}<br>%{customdata[0]:,} houses · shaking %{customdata[1]:.2f}<br>real %{customdata[2]:.0%} · forecast %{customdata[3]:.0%}" + "<extra></extra>";
  const traces = () => [
    { type: "scatter", mode: "markers", x: lon, y: lat, text, customdata: cd, hovertemplate: hover,
      marker: { color: values(), colorscale: mode === "diff" ? DIV : RISK, cmin: mode === "diff" ? -0.5 : 0, cmax: mode === "diff" ? 0.5 : 1, size, line: { color: lineCol, width: lineW } } },
    { type: "scatter", mode: "markers", x: [EPI.lon], y: [EPI.lat], hovertemplate: "Epicentre of the 25 April 2015 main shock<extra></extra>", marker: { symbol: "star", size: 18, color: ACC, line: { color: SURF, width: 1 } } },
  ];
  const layout = lay("", "", { height: el.clientHeight || 520, margin: { l: 34, r: 8, t: 8, b: 28 }, annotations: labels, dragmode: false,
    xaxis: ax("", { showgrid: false, ticksuffix: "°E", range: [84.3, 86.75] }), yaxis: ax("", { showgrid: false, ticksuffix: "°N", scaleanchor: "x", scaleratio: 1.13 }) });
  const draw = () => Plotly.react(el, traces(), layout, { ...CONFIG, scrollZoom: false });
  lazy(el, () => render(el, { data: traces(), layout }, { config: { ...CONFIG, scrollZoom: false } }));
  const legend = () => {
    $("#legendTag").textContent = mode === "diff" ? "Real minus forecast, share destroyed" : mode === "model" ? "Forecast share destroyed" : "Real share destroyed";
    $("#legendRamp").classList.toggle("ramp--div", mode === "diff");
    $("#legendL").innerHTML = mode === "diff" ? "<span>forecast too high</span><span>about right</span><span>too low</span>" : "<span>0%</span><span>50%</span><span>100%</span>";
  };
  const segCtl = seg($("#mapMode"), (m) => { mode = m; legend(); draw(); });
  const sl = $("#shift");
  const upd = () => {
    si = +sl.value;
    setRangeFill(sl);
    const v = shifts[si];
    $("#shiftV").textContent = v === 0 ? "±0.00" : sgn(v, 2);
    countTo($("#shiftTotal"), W.totals[si], { duration: 700 });
    const d = W.totals[si] - W.totals[i0];
    $("#shiftDelta").className = "delta " + (d > 0 ? "up" : d < 0 ? "down" : "");
    $("#shiftDelta").textContent = v === 0 ? `Real count: ${num(actualTotal)} homes` : `${d > 0 ? "+" : "−"}${num(Math.abs(d))} homes against the real quake's forecast (${(W.totals[si] / W.totals[i0]).toFixed(2)}×)`;
  };
  sl.addEventListener("input", () => { upd(); if (mode !== "model") { mode = "model"; segCtl.set("model"); legend(); } draw(); });
  if (params.has("shift")) sl.value = params.get("shift");
  if (params.has("mapmode")) { mode = params.get("mapmode"); segCtl.set(mode); legend(); }
  si = +sl.value;
  upd();
  $("#shiftNote").textContent = "Houses stay exactly as surveyed; only the shaking changes. Half a point on the intensity scale is roughly the difference between a ward near the epicentre and one 30–40 km further away.";
  $("#mapKv").innerHTML = `
    <span>Real homes destroyed</span><b>${num(actualTotal)}</b>
    <span>${esc(ov.labels[ov.headline])} forecast</span><b>${num(W.totals[i0])}</b>
    <span>Ward match, held-out districts</span><b>r = ${F.unseen.r.toFixed(2)}</b>
    <span>Ward match, training districts</span><b>r = ${F.seen.r.toFixed(2)}</b>
    <span>Yellow ring</span><b>held-out ward</b>`;
}

// ====================================================================== CHECKLIST
function buildChecklist({ ov, S }) {
  const C = ov.cart.checklist, nodes = Object.fromEntries(C.nodes.map((n) => [n.id, n]));
  const nLeaves = C.leaves;
  $("#ckTitle").innerHTML = `A few questions, <em>asked at the door.</em>`;
  $("#ckLead").textContent = `CART grew a ${ov.cart.full_leaves}-leaf tree, then cost-complexity pruning cut it back. This ${nLeaves}-leaf version fits on a card: on the held-out districts it ranks houses with AUC ${S.checklist.unseen.auc.toFixed(3)}, against ${S.cart.unseen.auc.toFixed(3)} for the ${ov.cart.leaves}-leaf CART the comparison uses. The shaking comes from the ward's shaking map; everything else an inspector can see from the street.`;
  // layout: leaves left to right, parents centred over their children
  const parent = {}, depth = {}, x = {};
  let leafI = 0, maxDepth = 0;
  const walk = (i, d) => {
    depth[i] = d; maxDepth = Math.max(maxDepth, d);
    const n = nodes[i];
    if (n.leaf) { x[i] = leafI++; return; }
    parent[n.left] = i; parent[n.right] = i;
    walk(n.left, d + 1); walk(n.right, d + 1);
    x[i] = (x[n.left] + x[n.right]) / 2;
  };
  walk(0, 0);
  const BW = 150, BH = 70, GX = 160, GY = 110, PAD = 8;
  const Wd = leafI * GX + PAD * 2, Hd = (maxDepth + 1) * GY + PAD;
  const svg = $("#treeSvg");
  svg.setAttribute("viewBox", `0 0 ${Wd} ${Hd}`);
  svg.style.minWidth = Math.max(640, leafI * 110) + "px";
  const cx = (i) => PAD + x[i] * GX + GX / 2, cy = (i) => PAD + depth[i] * GY;
  const q = (n) => (n.no.startsWith("not ") ? n.no.slice(4) : n.no) + "?";
  let edges = "", boxes = "";
  for (const n of C.nodes) {
    if (n.leaf) continue;
    for (const [child, lab] of [[n.left, "no"], [n.right, "yes"]]) {
      const x1 = cx(n.id), y1 = cy(n.id) + BH, x2 = cx(child), y2 = cy(child), my = (y1 + y2) / 2;
      edges += `<path class="edge" data-to="${child}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"/>`;
      edges += `<text class="elabel" x="${(x1 + x2) / 2 + (x2 > x1 ? 6 : -6)}" y="${my - 4}" text-anchor="${x2 > x1 ? "start" : "end"}">${lab.toUpperCase()}</text>`;
    }
  }
  const lines = (s) => {            // long questions wrap onto two lines, at ": " when there is one
    if (s.length <= 17) return [s];
    const k = s.indexOf(": ");
    if (k > 0) return [s.slice(0, k + 1), s.slice(k + 2)];
    const mid = s.lastIndexOf(" ", Math.ceil(s.length / 2));
    return mid > 0 ? [s.slice(0, mid), s.slice(mid + 1)] : [s];
  };
  for (const n of C.nodes) {
    const X = cx(n.id) - BW / 2, Y = cy(n.id);
    const fill = n.leaf ? riskColor(n.p) : "#181b1f";
    const ink = n.leaf && n.p > 0.55 ? "#1c1500" : INK;
    const head = n.leaf ? [`${pct(n.p)} destroyed`] : lines(q(n));
    const sub = n.leaf ? `${pct(n.share, 1)} of houses` : `${pct(n.p)} destroyed · ${pct(n.share)}`;
    const y0 = head.length === 1 ? Y + 26 : Y + 19;
    boxes += `<g class="node" data-id="${n.id}"><title>${esc(n.leaf ? head[0] + " · " + sub : q(n) + " " + sub + " of houses")}</title><rect class="nbox" x="${X}" y="${Y}" width="${BW}" height="${BH}" rx="9" fill="${fill}"/>` +
      head.map((t, k) => `<text class="nq" x="${X + BW / 2}" y="${y0 + k * 17}" text-anchor="middle" style="fill:${ink}">${esc(t)}</text>`).join("") +
      `<text class="np" x="${X + BW / 2}" y="${Y + BH - 11}" text-anchor="middle" style="fill:${n.leaf && n.p > 0.55 ? "#3a2a00" : INK2}">${esc(sub)}</text></g>`;
  }
  svg.innerHTML = edges + boxes;
  const pathTo = (leaf) => { const ids = [leaf]; while (parent[ids[0]] !== undefined) ids.unshift(parent[ids[0]]); return ids; };
  const hl = (leaf) => {
    if (leaf === null) { svg.classList.remove("hl"); $$(".on", svg).forEach((e) => e.classList.remove("on")); return; }
    const ids = new Set(pathTo(leaf));
    svg.classList.add("hl");
    $$("g.node", svg).forEach((g) => g.classList.toggle("on", ids.has(+g.dataset.id)));
    $$(".edge", svg).forEach((e) => e.classList.toggle("on", ids.has(+e.dataset.to)));
  };
  $("#rules").innerHTML = C.rules.map((r, i) => `<div class="rule" data-leaf="${r.leaf}"><i>${String.fromCharCode(65 + i)}</i><span>${r.if.map(esc).join(" · ")}</span><span class="rule__bar"><i style="--w:${(r.p * 100).toFixed(1)}%"></i></span><b>${pct(r.p)}</b></div>`).join("");
  $$(".rule").forEach((r) => { r.addEventListener("pointerenter", () => hl(+r.dataset.leaf)); r.addEventListener("pointerleave", () => hl(null)); });
}

// ====================================================================== MODELS
const BLURB = {
  cart: "One classification tree, grown on Gini impurity and pruned by cost-complexity.",
  logit: "Logistic regression: the straight-line baseline every ensemble must beat.",
  gam: "Generalized additive model: one smooth learned curve per fact, added up.",
  bagging: "40 deep trees, each on a bootstrap resample of houses, averaged.",
  forest: "120 trees; every split only sees a random third of the facts.",
  adaboost: "400 one-question stumps, each re-weighted toward the last ones' mistakes.",
  gboost: "Small trees fitted one after another to what is still unexplained.",
  crowd: "A plain average of five different kinds of model. Nothing is learned.",
  stack: "A logistic regression that learns how much to trust each of seven models.",
  bma: "4,096 logistic models averaged by their Bayesian evidence (BIC).",
};
function buildModels({ ov, S, L }) {
  const G = ov.growth;
  $("#modelsLead").textContent = `A single tree changes a lot when the training houses change. Average many of them (bagging, random forests) and the noise cancels; build them one after another on the remaining errors (boosting) and the bias shrinks. Single forest trees score ${G.single_tree[0].toFixed(2)}–${G.single_tree[1].toFixed(2)} AUC on the held-out districts; the forest scores ${G.forest_auc.toFixed(3)}.`;
  const order = ["cart", "logit", "gam", "bagging", "forest", "adaboost", "gboost", "crowd", "stack", "bma"];
  $("#lineup").innerHTML = order.map((k, i) => `<article class="mcard" data-reveal style="--d:${(i % 5) * 70};--col:${col(k)}"><span class="kind kind--${S[k].kind}">${S[k].kind === "single" ? "single model" : S[k].kind}</span><h3>${esc(L[k])}</h3><p>${BLURB[k]}</p>
    <div class="auc">${S[k].unseen.auc.toFixed(3)}<small>AUC held-out</small></div><div class="fine" style="margin-top:2px">${S[k].seen.auc.toFixed(3)} on seen districts</div></article>`).join("");
  const plots = [["trees_alone", "#pAlone"], ["growth", "#pGrowth"], ["basis_cumulative", "#pBasis"], ["stack_weights", "#pStack"], ["family_bma", "#pFam"], ["bma_inclusion", "#pIncl"]];
  plots.forEach(([id, sel]) => lazy($(sel), () => plotG(id, $(sel), 340)));
}

// ====================================================================== COMPARE
function buildCompare({ ov, S, L, D }) {
  const B = ov.baselines, A20 = ov.capacity.at20, WR = ov.ward_rank;
  const learned = ["checklist", "cart", "logit", "gam", "bagging", "forest", "adaboost", "gboost", "crowd", "stack", "bma"];
  const cols = [
    { k: "seen", h: "AUC seen", get: (m) => S[m]?.seen.auc ?? B[m]?.seen, best: "max", f: (v) => v.toFixed(3) },
    { k: "unseen", h: "AUC held-out", get: (m) => S[m]?.unseen.auc ?? B[m]?.unseen, best: "max", f: (v) => v.toFixed(3) },
    { k: "ll", h: "Log loss", get: (m) => S[m]?.unseen.log_loss, best: "min", f: (v) => v.toFixed(3) },
    { k: "brier", h: "Brier", get: (m) => S[m]?.unseen.brier, best: "min", f: (v) => v.toFixed(3) },
    { k: "cap", h: "Top 20% catches", get: (m) => A20[m]?.recall ?? S[m]?.unseen.recall_at_capacity, best: "max", f: (v) => pct(v, 1) },
    { k: "ward", h: "Within-ward AUC", get: (m) => WR[m]?.auc, best: "max", f: (v) => v.toFixed(3) },
  ];
  const bestOf = {};
  cols.forEach((c) => { const vals = learned.map((m) => c.get(m)).filter((v) => v != null); bestOf[c.k] = c.best === "max" ? Math.max(...vals) : Math.min(...vals); });
  const row = (m, isBase) => `<tr><td><span class="sw" style="--col:${col(m)}"></span>${esc(L[m])}</td><td>${isBase ? "no learning" : S[m].kind}</td>${cols.map((c) => {
    const v = c.get(m);
    return `<td class="num ${!isBase && v != null && Math.abs(v - bestOf[c.k]) < 1e-9 ? "best" : ""}">${v == null ? "–" : c.f(v)}</td>`;
  }).join("")}</tr>`;
  const groups = [["Single models", ["checklist", "cart", "logit", "gam"]], ["Ensembles", ["bagging", "forest", "adaboost", "gboost"]], ["Combined models", ["crowd", "stack", "bma"]]];
  $("#cmpTable").innerHTML = `<thead><tr><th>Model</th><th>Type</th>${cols.map((c) => `<th class="num">${c.h}</th>`).join("")}</tr></thead><tbody>` +
    groups.map(([g, ms]) => `<tr class="grp"><td colspan="8">${g}</td></tr>` + ms.map((m) => row(m, false)).join("")).join("") +
    `<tr class="grp"><td colspan="8">Rules of thumb, no learning</td></tr>` + ["rule", "shaking", "random"].map((m) => row(m, true)).join("") + "</tbody>";
  $("#cmpNote").textContent = `Seen = ${D.n_test_seen.toLocaleString()} random houses from the 8 training districts that no model trained on. Held-out = all ${D.n_test_unseen.toLocaleString()} houses of ${D.unseen_districts.join(", ")}, never seen in training; every column after the first is on these. Top 20% catches = share of all destroyed houses found among the 20% ranked riskiest. Within-ward AUC = ranking inside each ward, where every house had the same estimated shaking. Bold = best learned model.`;
  const bestSeen = learned.reduce((a, b) => (S[b].seen.auc > S[a].seen.auc ? b : a));
  const bestUn = learned.reduce((a, b) => (S[b].unseen.auc > S[a].unseen.auc ? b : a));
  $("#cmpTitle").innerHTML = `${esc(L[bestSeen])} wins at home. <em>${esc(L[bestUn])} travels best.</em>`;
  $("#cmpLead").textContent = `All eleven models were scored on the same houses twice: new houses from districts they trained on, and three districts they never saw. Moving to new districts costs every model about 0.1 AUC, and it reshuffles the ranking.`;
  lazy($("#pAuc"), () => plotG("auc_compare", $("#pAuc"), 440, { margin: { l: 10, r: 10, t: 30, b: 40 } }));
  lazy($("#pBoot"), () => plotG("bootstrap", $("#pBoot"), 440, { margin: { l: 10, r: 10, t: 30, b: 40 } }));
  const bt = (set, a, b) => ov.bootstrap.find((r) => r.set === set && r.a === a && r.b === b);
  const ci = (r) => `${sgn(r.mean)} AUC (95% interval ${sgn(r.low)} to ${sgn(r.high)})`;
  const v = [];
  const gS = bt("seen", "gboost", "cart"), sU = bt("unseen", "stack", "gboost"), cU = bt("unseen", "crowd", "cart"), cG = bt("unseen", "crowd", "gboost"), bL = bt("unseen", "bma", "logit");
  if (gS) v.push(["Ensembles beat a single tree, at home and away", `On seen districts gradient boosting beats CART by ${ci(gS)}. On the held-out districts the crowd average beats CART by ${ci(cU)}. Every interval sits clear of zero, so with this many houses the gains are real, even if they look small.`]);
  if (sU) v.push(["Learned combination weights do not travel", `Stacking learns its weights on validation houses from the training districts. It is ${S.stack.seen.auc >= S.gboost.seen.auc ? "the best model at home" : "strong at home"} (${S.stack.seen.auc.toFixed(3)}), yet on the held-out districts it falls behind plain boosting by ${ci(sU)}. The crowd average learns nothing and comes out ahead of boosting by ${ci(cG)}.`]);
  v.push(["Bayesian model averaging chooses instead of averaging", `Across model families, the BMA weights put ${pct(Math.max(...ov.family_bma.weights), 1)} on a single model once all 20,000 validation houses count. Over 4,096 logistic explanations, ${ov.bma.models_for_99} models hold 99% of the weight, and the averaged prediction scores ${sgn(bL.mean)} AUC against one full logistic model. BMA's real value here is the inclusion probabilities: it shows which facts a survey must collect.`]);
  v.push(["Where the ground shook explains most of it", `Without the ward shaking estimate, gradient boosting drops from ${S.gboost.unseen.auc.toFixed(3)} to ${ov.no_shaking.gboost.unseen.toFixed(3)} AUC on the held-out districts, and ranking by shaking alone already catches ${pct(A20.shaking.recall, 1)} of collapses in the top 20%. The models' extra value is inside a ward, where shaking is equal: there the ${esc(L[ov.headline]).toLowerCase()} ranks houses at ${WR[ov.headline].auc.toFixed(3)} AUC against ${WR.rule.auc.toFixed(3)} for 'mud walls first'. The margin is modest because more than nine in ten held-out houses have stone or mud walls.`]);
  $("#verdicts").innerHTML = v.map(([h, p], i) => `<div class="verdict" data-reveal style="--d:${i * 80}"><h3>${h}</h3><p>${p}</p></div>`).join("");
}

// ====================================================================== FRAGILITY
function buildFragility({ ov, HL }) {
  const fr = ov.fragility, U = ov.upgrades;
  const stone = fr.types["Stone in mud mortar"], rc = fr.types["RC frame, engineered"];
  const top = (o) => Math.max(...o.filter((v) => v != null));
  $("#fragLead").textContent = `Same ward, same shaking, very different outcomes. At the strongest shaking, stone houses laid in mud mortar were destroyed ${pct(top(stone.observed))} of the time; engineered concrete frames ${pct(top(rc.observed.length ? rc.observed : [0]))}. Earthquake engineers call these fragility curves. Here they come out of ${ov.data.buildings.toLocaleString()} real outcomes, and the ${HL.toLowerCase()} learns them without being told what a wall is.`;
  let mode = "observed";
  const el = $("#pFrag");
  const draw = () => plotG(mode === "observed" ? "fragility_observed" : "fragility_model", el, 420);
  const tag = () => { $("#fragTag").textContent = mode === "observed" ? "Fragility curves, observed" : `Fragility curves, learned by the ${HL.toLowerCase()}`; };
  const fs = seg($("#fragMode"), (m) => { mode = m; tag(); draw(); });
  if (params.get("frag") === "model") { mode = "model"; fs.set("model"); tag(); }
  lazy(el, draw);
  lazy($("#pUp"), () => plotG("upgrades", $("#pUp"), 300, { margin: { l: 10, r: 10, t: 30, b: 40 } }));
  const best = U.reduce((a, b) => (b.drop > a.drop ? b : a), U[0]);
  const cem = U.find((u) => u.key === "cement");
  $("#upNote").textContent = `For ${best.n.toLocaleString()} real held-out houses, '${best.label.toLowerCase()}' takes the predicted risk from ${pct(best.before)} to ${pct(best.after)}.` + (cem ? ` The cheaper fix, cement instead of mud mortar, takes it from ${pct(cem.before)} to ${pct(cem.after)}.` : "") + " These are patterns learned from the survey, not engineering tests.";
  lazy($("#pDist"), () => plotG("district_totals", $("#pDist"), 380));
  lazy($("#pCap"), () => plotG("capacity", $("#pCap"), 380));
}

// ====================================================================== HOUSE BUILDER
const NUMS = [
  ["floors", "Storeys", 1, 6, 1, (v) => v],
  ["age", "Age (years)", 0, 100, 1, (v) => v],
  ["area", "Floor area (sq ft)", 100, 1500, 10, (v) => v],
  ["height", "Height (ft)", 6, 45, 1, (v) => v],
];
const SELECTS = [["ground", "land_surface_condition", "Ground"], ["foundation", "foundation_type", "Foundation"], ["roof", "roof_type", "Roof"],
  ["ground_floor", "ground_floor_type", "Ground floor"], ["upper_floors", "other_floor_type", "Upper floors"], ["position", "position", "Position"], ["plan", "plan_configuration", "Plan shape"]];
const RAW = { floors: "count_floors_pre_eq", age: "age_building", area: "plinth_area_sq_ft", height: "height_ft_pre_eq", shaking: "pred_intensity" };

function buildHouse({ ov, opts, L }) {
  const nice = (v) => opts.level_nice[v] || v;
  const house = { floors: 2, age: 20, area: 350, height: 16, shaking: 7.6, ground: "Flat", foundation: "Mud mortar-Stone/Brick", roof: "Bamboo/Timber-Light roof",
    ground_floor: "Mud", upper_floors: "TImber/Bamboo-Mud", position: "Not attached", plan: "Rectangular", walls: ["mud_mortar_stone"] };
  let heightTouched = false;
  const form = $("#form");
  form.innerHTML =
    NUMS.map(([k, lab, lo, hi, st]) => `<label class="ctl field"><span>${lab} <b id="v_${k}"></b></span><input type="range" id="i_${k}" min="${lo}" max="${hi}" step="${st}"></label>`).join("") +
    `<div class="ctl field full"><label class="ctl" style="display:block"><span>Ground shaking at the house <b id="v_shaking"></b></span><input type="range" id="i_shaking" min="6" max="8.5" step="0.05"></label><div class="chips" id="distChips"></div></div>` +
    `<div class="field full"><span>Walls and frame (pick all that apply)</span><div class="chips" id="wallChips"></div></div>` +
    SELECTS.map(([k, raw, lab]) => `<div class="field"><label for="s_${k}">${lab}</label><select class="pick" id="s_${k}">${opts.choices[raw].map((v) => `<option value="${esc(v)}">${esc(nice(v))}</option>`).join("")}</select></div>`).join("");
  const sync = () => {
    NUMS.forEach(([k]) => { const i = $(`#i_${k}`); i.value = house[k]; setRangeFill(i); $(`#v_${k}`).textContent = house[k]; });
    const s = $("#i_shaking"); s.value = house.shaking; setRangeFill(s); $("#v_shaking").textContent = (+house.shaking).toFixed(2);
    SELECTS.forEach(([k]) => { $(`#s_${k}`).value = house[k]; });
    $$("#wallChips .chip-btn").forEach((b) => b.classList.toggle("on", house.walls.includes(b.dataset.w)));
  };
  $("#wallChips").innerHTML = Object.entries(opts.walls).map(([k, lab]) => `<button type="button" class="chip-btn" data-w="${k}">${esc(lab.replace("Walls: ", "").replace("Frame: ", "frame: "))}</button>`).join("");
  $("#distChips").innerHTML = Object.entries(opts.district_shaking).sort((a, b) => b[1] - a[1]).map(([d, v]) => `<button type="button" class="chip-btn" data-s="${v}">${esc(d)} ${v.toFixed(2)}</button>`).join("");
  const go = debounce(predict, 220);
  NUMS.forEach(([k]) => $(`#i_${k}`).addEventListener("input", (e) => {
    house[k] = +e.target.value;
    if (k === "height") heightTouched = true;
    if (k === "floors" && !heightTouched) house.height = Math.min(45, Math.max(6, house.floors * 8));
    sync(); go();
  }));
  $("#i_shaking").addEventListener("input", (e) => { house.shaking = +e.target.value; sync(); go(); });
  $$("#distChips .chip-btn").forEach((b) => b.addEventListener("click", () => { house.shaking = +b.dataset.s; sync(); go(); }));
  $$("#wallChips .chip-btn").forEach((b) => b.addEventListener("click", () => {
    const w = b.dataset.w, on = house.walls.includes(w);
    if (on && house.walls.length === 1) return;
    house.walls = on ? house.walls.filter((x) => x !== w) : [...house.walls, w];
    sync(); go();
  }));
  SELECTS.forEach(([k]) => $(`#s_${k}`).addEventListener("change", (e) => { house[k] = e.target.value; go(); }));
  const ex = opts.examples;
  $("#examples").innerHTML = `<span class="tag" style="align-self:center;margin-right:6px">Real held-out houses:</span>` + ex.map((e, i) => `<button type="button" class="chip-btn" data-i="${i}">${esc(e.title)} · grade ${e.grade}</button>`).join("");
  const load = (e) => {
    const h = e.house;
    Object.entries(RAW).forEach(([k, raw]) => { house[k] = h[raw]; });
    house.shaking = Math.round(h.pred_intensity * 20) / 20;
    SELECTS.forEach(([k, raw]) => { house[k] = h[raw]; });
    house.walls = h.walls.map((w) => w.replace("has_superstructure_", ""));
    heightTouched = true;
    sync(); predict();
  };
  $$("#examples .chip-btn").forEach((b) => b.addEventListener("click", () => { $$("#examples .chip-btn").forEach((x) => x.classList.toggle("on", x === b)); load(ex[+b.dataset.i]); }));
  sync();
  const curveEl = $("#pCurve");
  let first = true, seq = 0;
  async function predict() {
    const my = ++seq;
    $("#result").classList.add("busy");
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(house) }).then((x) => x.json());
    if (my !== seq) return;
    $("#result").classList.remove("busy");
    if (!r.p) { $("#verbal").textContent = r.detail?.[0]?.msg || r.detail || "Could not score that house."; return; }
    const p = r.p[r.headline];
    const arc = $("#gArc");
    arc.style.strokeDashoffset = (314.16 * (1 - p)).toFixed(1);
    arc.style.setProperty("--col", riskColor(0.25 + p * 0.75));
    countTo($("#gVal"), p, { pct: true, duration: 700 });
    $("#gLab").textContent = `chance destroyed`;
    const verbal = p < 0.1 ? "Very likely to stand." : p < 0.25 ? "Probably stands." : p < 0.5 ? "A serious risk of collapse." : p < 0.75 ? "More likely destroyed than not." : "Very likely destroyed.";
    $("#verbal").textContent = verbal;
    $("#verbalNote").textContent = `${r.label}, at shaking ${house.shaking.toFixed(2)}. The models range from ${pct(r.spread[0])} to ${pct(r.spread[1])}.`;
    const keys = Object.keys(r.p);
    $("#spread").innerHTML = keys.map((k) => `<i class="${k === r.headline ? "big" : ""}" style="left:${(r.p[k] * 100).toFixed(1)}%;--col:${col(k)}" title="${esc(L[k] || k)}: ${pct(r.p[k], 1)}"></i>`).join("");
    const terms = r.why.terms.slice(0, 8), mx = Math.max(0.5, ...terms.map((t) => Math.abs(t.log_odds)));
    $("#why").innerHTML = terms.map((t) => { const w = (Math.abs(t.log_odds) / mx) * 50, left = t.log_odds >= 0 ? 50 : 50 - w;
      return `<div class="why__row"><span>${esc(t.fact)}</span><span class="why__t"><i style="left:${left}%;width:${w}%;background:${t.log_odds >= 0 ? "#e0621f" : "#3987e5"}"></i></span><b>${sgn(t.log_odds, 2)}</b></div>`; }).join("") +
      `<p class="fine" style="margin-top:6px">Log-odds pushed up (orange) or down (blue) by each fact. GAM alone says ${pct(r.why.p_gam)}.</p>`;
    $("#steps").innerHTML = r.checklist.steps.map((s) => `<li>${esc(s)}</li>`).join("");
    $("#stepsLeaf").textContent = `Ends in a group where ${pct(r.checklist.p)} of houses were destroyed (${pct(r.checklist.share, 1)} of all houses).`;
    $("#ups").innerHTML = r.upgrades.length ? r.upgrades.map((u) => `<div class="up"><span>${esc(u.label)}</span><b>${pct(p)} → ${pct(u.p)}</b></div>`).join("") : `<p class="fine">No upgrade applies: this is already an engineered concrete frame.</p>`;
    const data = [
      { type: "scatter", mode: "lines", x: r.curve.shaking, y: r.curve.p, line: { color: ACC, width: 2 }, hovertemplate: "shaking %{x:.2f}<br>%{y:.0%} chance destroyed<extra></extra>" },
      { type: "scatter", mode: "markers", x: [house.shaking], y: [p], marker: { color: ACC, size: 11, line: { color: SURF, width: 2 } }, hoverinfo: "skip" },
    ];
    const layout = lay("Shaking intensity", "", { height: 220, yaxis: ax("", { tickformat: ".0%", range: [0, 1] }), margin: { l: 40, r: 8, t: 6, b: 34 } });
    if (first) { first = false; await render(curveEl, { data, layout }, { animate: !capturing }); } else Plotly.react(curveEl, data, layout, CONFIG);
  }
  const start = Math.min(ex.length - 1, +(params.get("ex") || 0));
  lazy($("#house"), () => { $$("#examples .chip-btn")[start]?.classList.add("on"); load(ex[start]); });
}

// ====================================================================== GRADES
function buildGrades({ ov }) {
  const g = ov.grades.unseen;
  $("#gradesLead").textContent = `The survey has five grades, not two. A multi-class CART (pruned to ${ov.grades.cart_leaves} leaves) puts ${pct(g.cart.exact)} of held-out houses in exactly the right grade and ${pct(g.cart.within_one)} within one grade. Always guessing the most common grade gets ${pct(g.cart.majority_exact)} exact. The forest of ${ov.grades.forest_trees} five-class trees reaches ${pct(g.forest.exact)} and ${pct(g.forest.within_one)}.`;
  lazy($("#pGrades"), () => plotG("grade_scores", $("#pGrades"), 360));
  lazy($("#pConf"), () => plotG("grade_confusion_cart", $("#pConf"), 360));
}

// ====================================================================== LIMITS AND INSIDE
function buildLimits({ ov, S }) {
  const d = ov.districts.find((x) => x.name === "Gorkha");
  const items = [
    ["One earthquake", `Everything was learned from a single event. The biggest miss is Gorkha itself: the model expected ${num(d.predicted)} destroyed homes and ${num(d.collapsed)} were. The rupture ran east from the epicentre, and one shaking number per ward cannot capture that.`],
    ["Shaking is an estimate per ward", "Every house in a ward gets the same shaking value. Soil, slope failures, landslides and aftershocks are not in the data, which is why some wards on the difference map are far off."],
    ["Patterns, not engineering tests", "'Cement instead of mud' lowers predicted risk because houses built that way fell less often in this survey. Those houses may also differ in ways the survey did not record. Retrofits need engineers."],
    ["Mostly one kind of house", `Over nine in ten houses in the held-out districts have stone or mud walls, so inside one village the models separate houses only modestly (within-ward AUC ${ov.ward_rank[ov.headline].auc.toFixed(2)}).`],
    ["Rural Nepal only", "The survey covers 11 districts outside the Kathmandu valley. Cities, tall concrete buildings and other countries' building styles are barely represented."],
    ["Damage grades are human judgements", "Survey engineers graded each house by eye. Neighbouring grades blur, which caps how well any model can sort the five grades."],
  ];
  $("#limitsList").innerHTML = items.map(([h, p], i) => `<div class="limit" data-reveal style="--d:${(i % 2) * 90}"><h3>${h}</h3><p>${p}</p></div>`).join("");
}

function buildInside({ ov }) {
  const c = (t, p, eq) => `<article class="concept" data-reveal><h3>${t}</h3>${eq ? `<pre>${eq}</pre>` : ""}<p>${p}</p></article>`;
  $("#insideGrid").innerHTML = [
    c("CART", `Binary splits chosen to cut Gini impurity the most, searched over quantile bins so each split is a few counting passes. The full tree had ${ov.cart.full_leaves} leaves.`, "Gini(t) = 1 − Σ p<sub>k</sub>²      gain = n·G(t) − n<sub>L</sub>·G(L) − n<sub>R</sub>·G(R)"),
    c("Cost-complexity pruning", `Weakest-link pruning collapses the branch that buys the least error per leaf, giving a whole sequence of trees. Validation Brier score picks ${ov.cart.leaves} leaves.`, "g(t) = (R(t) − R(T<sub>t</sub>)) / (|leaves(T<sub>t</sub>)| − 1)"),
    c("Bagging and random forests", "Each tree learns from a bootstrap resample of houses; the forest also limits each split to a random third of the facts. Averaging cuts variance; out-of-bag houses give a free test.", "p̂(x) = (1/B) Σ<sub>b</sub> T<sub>b</sub>(x)"),
    c("AdaBoost", "One-question stumps fitted to re-weighted houses. Misclassified houses gain weight e<sup>α</sup>, and each stump votes with α.", "α<sub>m</sub> = ½ ln((1 − err<sub>m</sub>) / err<sub>m</sub>)     F(x) = Σ α<sub>m</sub> h<sub>m</sub>(x)"),
    c("Gradient boosting", `Small regression trees fitted to the residuals y − p, each leaf set by a Newton step, added with a learning rate and stopped on validation (${ov.growth.gboost_best} trees).`, "F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + ν · Σ<sub>leaf</sub> (Σ r / Σ p(1−p)) · 1[x ∈ leaf]"),
    c("Adaptive basis function models", "Every ensemble here is a weighted sum of basis functions whose shape is learned from the data. Each tree is one basis function.", "f(x) = w<sub>0</sub> + Σ<sub>m</sub> w<sub>m</sub> φ<sub>m</sub>(x)"),
    c("Generalized additive model", `Logistic GAM fitted by local scoring and backfitting. Each number gets a cubic P-spline with a wiggliness penalty, λ = ${ov.gam.lam} chosen on validation.`, "logit p = α + Σ<sub>j</sub> f<sub>j</sub>(x<sub>j</sub>)     penalty λ Σ (Δ²β)²"),
    c("Stacking and the crowd", "The crowd is a plain average of five model families. Stacking fits a logistic regression on the seven models' validation log-odds to learn the weights.", "p = σ(w<sub>0</sub> + Σ<sub>k</sub> w<sub>k</sub> logit p<sub>k</sub>(x))"),
    c("Bayesian model averaging", `Over all 4,096 subsets of 12 groups of facts, weights from BIC give each group an inclusion probability. Across model families, held-out likelihood gives the weights.`, "p(M | D) ∝ exp(−BIC<sub>M</sub> / 2)"),
    c("Testing where it counts", "Three whole districts are held out, including the epicentre district. Paired bootstrap intervals check every claimed gain. scikit-learn is used only in tests.", "ΔAUC* over 300 resamples of 50,000 houses"),
  ].join("");
}

function initNav() {
  const links = $$("#navLinks a"), pill = $(".nav__pill"), nav = $("#nav"), progress = $("#progress");
  const setActive = (a) => { links.forEach((l) => l.classList.toggle("active", l === a)); pill.style.opacity = a ? 1 : 0; if (a) { pill.style.width = a.offsetWidth + "px"; pill.style.transform = `translateX(${a.offsetLeft}px)`; } };
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) setActive(links.find((l) => l.getAttribute("href") === `#${e.target.id}`)); }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main > section[id]").forEach((s) => io.observe(s));
  let last = 0;
  addEventListener("scroll", () => {
    const y = scrollY, max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav.classList.toggle("hidden", y > last && y > 500);
    last = y;
  }, { passive: true });
}

boot();
