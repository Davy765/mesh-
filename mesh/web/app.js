"use strict";
// Mesh Playground. No libraries, no inline code (the page's security policy allows neither).
(function () {
  const $ = (id) => document.getElementById(id);
  const pad = $("pad"), ctx = pad.getContext("2d");
  const RING_COLORS = ["#2f5bea", "#e8833a", "#16a37f", "#c2417d", "#7a5af5", "#9a8700"];
  let info = null, drawing = false, dirty = false, last = null, timer = null, inflight = false, queued = false;
  let point = null, map = null;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function inkColor() { return getComputedStyle(document.documentElement).getPropertyValue("--pad-ink").trim() || "#000"; }
  async function api(path, body) {
    const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
    const r = await fetch(path, opt);
    let data = {};
    try { data = await r.json(); } catch (e) { /* not JSON */ }
    if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
    return data;
  }

  // ---------------------------------------------------------------- drawing pad (digits)
  function clearPad() {
    ctx.clearRect(0, 0, pad.width, pad.height);
    dirty = false; point = null;
    $("pad-hint").classList.remove("hidden");
    if (info && info.d_in === 2) drawMap();
    resetAnswer();
  }
  function pos(ev) {
    const r = pad.getBoundingClientRect();
    return { x: (ev.clientX - r.left) * pad.width / r.width, y: (ev.clientY - r.top) * pad.height / r.height };
  }
  function stroke(a, b) {
    ctx.strokeStyle = inkColor(); ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  function onDown(ev) {
    if (!info) return;
    ev.preventDefault();
    if (info.d_in === 2) return pickPoint(ev);
    pad.setPointerCapture(ev.pointerId);
    drawing = true; last = pos(ev); stroke(last, last); dirty = true;
    $("pad-hint").classList.add("hidden");
  }
  function onMove(ev) {
    if (!drawing) return;
    const p = pos(ev); stroke(last, p); last = p;
    clearTimeout(timer); timer = setTimeout(predict, 350);   // live updates while drawing
  }
  function onUp() {
    if (!drawing) return;
    drawing = false; clearTimeout(timer); predict();
  }
  // Ink goes on a transparent canvas, so how much ink a pixel has is its alpha, whatever the theme colours.
  function padImage(size) {
    const c = document.createElement("canvas"); c.width = c.height = size;
    const g = c.getContext("2d"); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.drawImage(pad, 0, 0, size, size);
    const d = g.getImageData(0, 0, size, size).data, out = new Array(size * size);
    for (let i = 0; i < out.length; i++) out[i] = Math.round(d[i * 4 + 3] / 255 * 1000) / 1000;
    return out;
  }
  function showSample(pixels) {
    ctx.clearRect(0, 0, pad.width, pad.height);
    const n = Math.sqrt(pixels.length), s = pad.width / n;
    ctx.fillStyle = inkColor();
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const v = pixels[y * n + x];
      if (v > 0.02) { ctx.globalAlpha = v; ctx.fillRect(x * s, y * s, s + 0.5, s + 0.5); }
    }
    ctx.globalAlpha = 1;
    dirty = true; $("pad-hint").classList.add("hidden");
  }

  // ---------------------------------------------------------------- point picker (ring task)
  function drawMap() {
    ctx.clearRect(0, 0, pad.width, pad.height);
    if (map) {      // paint the grid at its own size, then scale it up: no seams between cells
      const c = document.createElement("canvas"); c.width = c.height = map.size;
      const g = c.getContext("2d");
      map.classes.forEach((k, i) => {
        g.fillStyle = RING_COLORS[k % RING_COLORS.length];
        g.fillRect(i % map.size, Math.floor(i / map.size), 1, 1);
      });
      ctx.save(); ctx.globalAlpha = 0.28; ctx.imageSmoothingEnabled = false;
      ctx.drawImage(c, 0, 0, pad.width, pad.height); ctx.restore();
    }
    if (point) {
      const r = info.range || 5, px = (point[0] + r) / (2 * r) * pad.width, py = (r - point[1]) / (2 * r) * pad.height;
      ctx.fillStyle = "#000"; ctx.beginPath(); ctx.arc(px, py, 8, 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(px, py, 4, 0, 7); ctx.fill();
    }
  }
  function pickPoint(ev) {
    const p = pos(ev), r = info.range || 5;
    point = [p.x / pad.width * 2 * r - r, r - p.y / pad.height * 2 * r];
    $("pad-hint").classList.add("hidden"); drawMap(); predict();
  }
  async function loadMap() {
    try { map = await api("/api/map"); info.range = map.range; drawMap(); } catch (e) { /* try later */ }
  }

  // ---------------------------------------------------------------- asking the network
  async function predict(trueLabel) {
    if (!info) return;
    if (inflight) { queued = true; return; }
    let body;
    if (info.d_in === 2) { if (!point) return; body = { point: point }; }
    else { if (!dirty) return; body = { image: padImage(56) }; }
    inflight = true;
    $("answer-sub").textContent = "Asking the network…";
    try {
      showAnswer(await api("/api/predict", body), trueLabel);
    } catch (e) {
      $("answer-sub").textContent = e.message;
      $("answer-sub").className = "small err";
    } finally {
      inflight = false;
      if (queued) { queued = false; predict(); }
    }
  }
  function className(i) { return info.d_in === 784 ? String(i) : "class " + i; }
  function resetAnswer() {
    $("big").textContent = "?";
    $("answer-line").textContent = info && info.d_in === 2 ? "Click anywhere on the square" : "Waiting for your drawing";
    $("answer-sub").textContent = ""; $("answer-sub").className = "muted small";
    renderBars(null);
  }
  function renderBars(probs) {
    const bars = $("bars"); bars.textContent = "";
    const n = info ? info.classes : 10, top = probs ? probs.indexOf(Math.max(...probs)) : -1;
    for (let i = 0; i < n; i++) {
      const b = el("div", "bar" + (i === top ? " top" : "")), f = el("div", "fill");
      f.style.height = (probs ? Math.max(2, probs[i] * 100) : 2) + "%";
      if (info && info.d_in === 2) f.style.background = RING_COLORS[i % RING_COLORS.length];
      b.title = probs ? className(i) + ": " + Math.round(probs[i] * 100) + "%" : className(i);
      b.append(f, el("div", "lab", info && info.d_in === 2 ? "c" + i : String(i)));
      bars.append(b);
    }
  }
  function showAnswer(r, trueLabel) {
    const conf = r.probs[r.label];
    $("big").textContent = info.d_in === 784 ? String(r.label) : "c" + r.label;
    if (info.d_in === 2) $("big").style.color = RING_COLORS[r.label % RING_COLORS.length];
    const sure = conf > 0.9 ? "Pretty sure" : conf > 0.6 ? "Fairly sure" : "Not sure";
    $("answer-line").textContent = sure + " it's " + className(r.label) + " (" + Math.round(conf * 100) + "%)";
    let sub = "Model version " + r.model_version + ", from training round " + r.round;
    if (trueLabel !== undefined) sub = (trueLabel === r.label ? "Correct. " : "Wrong: it was " + trueLabel + ". ") + sub;
    $("answer-sub").textContent = sub; $("answer-sub").className = "muted small";
    renderBars(r.probs);
    const route = $("route"); route.textContent = "";
    const total = r.route.reduce((s, h) => s + h.ms, 0);
    const step = (n, who, ms, tags) => {
      const li = el("li"); li.append(el("span", "step", String(n)), el("span", "who", who));
      (tags || []).forEach((t) => li.append(el("span", "tag " + t[0], t[1])));
      if (ms !== null) li.append(el("span", "ms", ms.toFixed(0) + " ms"));
      route.append(li);
    };
    step(1, r.agent + " turns your input into features (input layer)", null);
    r.route.forEach((h, i) => {
      const tags = [];
      if (h.spot_checked) tags.push(["check", "double-checked"]);
      if (h.caught_liar) tags.push(["caught", "caught a liar"]);
      if (h.failovers) tags.push(["caught", h.failovers + " skipped"]);
      step(i + 2, h.name + " runs " + h.piece, h.ms, tags);
    });
    step(r.route.length + 2, r.agent + " reads the answer (output layer)", null);
    const li = el("li", "muted small"); li.textContent = "Network time: " + total.toFixed(0) + " ms across " + r.route.length + " servers";
    route.append(li);
  }
  async function sample() {
    try {
      const s = await api("/api/sample");
      if (info.d_in === 784) { showSample(s.pixels); predict(s.label); }
      else { point = s.point; $("pad-hint").classList.add("hidden"); drawMap(); predict(s.label); }
    } catch (e) { $("answer-sub").textContent = e.message; $("answer-sub").className = "small err"; }
  }

  // ---------------------------------------------------------------- live network panel
  function stat(label, value) {
    const d = el("div", "stat"); d.append(el("b", "", value), el("span", "", label)); return d;
  }
  function renderChart(acc) {
    const svg = $("chart"), NS = "http://www.w3.org/2000/svg", H = 160, L = 40, B = 18;
    const W = Math.max(240, Math.round(svg.getBoundingClientRect().width) || 600);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.textContent = "";
    const mk = (tag, attrs, text) => {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      svg.append(e); return e;
    };
    [0, 0.5, 1].forEach((v) => {
      const y = (H - B) - v * (H - B - 8);
      mk("line", { x1: L, x2: W, y1: y, y2: y, class: "grid-line" });
      mk("text", { x: 0, y: y + 4, class: "axis" }, Math.round(v * 100) + "%");
    });
    if (!acc.length) { mk("text", { x: L + 8, y: H / 2, class: "axis" }, "measuring…"); return; }
    const r0 = acc[0].round, r1 = Math.max(acc[acc.length - 1].round, r0 + 1);
    const pts = acc.map((a) => [L + (a.round - r0) / (r1 - r0) * (W - L - 4), (H - B) - a.acc * (H - B - 8)]);
    const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
    mk("path", { d: d + " L" + pts[pts.length - 1][0] + " " + (H - B) + " L" + pts[0][0] + " " + (H - B) + " Z", class: "area" });
    mk("path", { d: d, class: "line" });
    const end = pts[pts.length - 1];
    mk("circle", { cx: end[0], cy: end[1], r: 4, class: "dot-end" });
    mk("text", { x: L, y: H - 2, class: "axis" }, "round " + r0);
    if (acc.length > 1) mk("text", { x: W - 4, y: H - 2, class: "axis", "text-anchor": "end" }, "round " + acc[acc.length - 1].round);
  }
  async function refresh() {
    try {
      const s = await api("/api/status");
      $("net-pill").className = "pill live";
      const acc = s.accuracy.length ? s.accuracy[s.accuracy.length - 1].acc : null;
      $("net-text").textContent = (s.mode === "demo" ? "Demo network" : "Live network") + " · round " + s.round;
      const stats = $("stats"); stats.textContent = "";
      stats.append(
        stat("training round", s.target_rounds ? s.round + " / " + s.target_rounds : String(s.round)),
        stat(info && info.d_in === 2 ? "test points right" : "test digits right", acc === null ? "–" : Math.round(acc * 100) + "%"),
        stat("rounds accepted (recent)", s.accepted_recent + " / " + s.rounds_recent),
        stat("validators", String(s.validators)),
        stat("block height", String(s.height)),
        stat("answers served here", String(s.served)));
      renderChart(s.accuracy);
      const nodes = $("nodes"); nodes.textContent = "";
      s.nodes.forEach((n) => {
        const d = el("div", "node" + (n.active ? "" : " ejected"));
        d.append(el("b", "", n.name),
          el("div", "muted", (n.role === "client" ? "agent" : "server") + " · " + (n.piece === "ends" ? "input & output layers" : n.piece)),
          el("div", "st" + (n.active ? " ok" : ""), n.active ? "active · reputation " + n.rep : "ejected for cheating"));
        nodes.append(d);
      });
      const ej = s.nodes.filter((n) => !n.active).length;
      $("legend").textContent = ej ? ej + " node(s) were caught cheating and removed." : "Every node is in good standing.";
      if (info && info.d_in === 2 && (!map || map.round !== s.round)) loadMap();
    } catch (e) {
      $("net-pill").className = "pill down"; $("net-text").textContent = "can't reach the network";
    }
  }

  // ---------------------------------------------------------------- start
  async function start() {
    pad.addEventListener("pointerdown", onDown);
    pad.addEventListener("pointermove", onMove);
    pad.addEventListener("pointerup", onUp);
    pad.addEventListener("pointercancel", onUp);
    $("clear-btn").addEventListener("click", clearPad);
    $("sample-btn").addEventListener("click", sample);
    try { info = await api("/api/info"); } catch (e) { $("net-text").textContent = "can't reach the network"; return; }
    if (info.d_in === 2) {
      $("headline").textContent = "Click a point. A model split across many computers will say which class it is.";
      $("pad-hint").textContent = "Click anywhere on the square";
      $("pad-note").textContent = "This is the quick demo task: points on a ring, 4 classes. The colours show the model's current answer everywhere.";
      $("sample-btn").textContent = "Try a test point";
      $("subline").textContent = "No single machine holds this model. Your point passes through slices held by independent " +
        "servers, which check each other's work and are paid only for honest answers.";
      $("faq-what").textContent = "This network runs the quick demo task: it sorts points on a plane into 4 classes " +
        "arranged around a ring. A network started with the digit task recognises handwritten digits instead. " +
        "Either way it is not a chatbot, and it does not read or learn from websites.";
      $("pad").setAttribute("aria-label", "Point picker");
    }
    resetAnswer();
    refresh(); setInterval(refresh, 3000);
    if (info.d_in === 2) loadMap();
  }
  start();
})();
