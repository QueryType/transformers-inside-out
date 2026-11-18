// Small shared helpers: DOM/SVG builders, token chips, bars, tooltip, theme, scroll-spy.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  function h(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    kids.flat().forEach((c) => c != null && e.append(c.nodeType ? c : document.createTextNode(c)));
    return e;
  }

  function s(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "text") e.textContent = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  }

  function svgRoot(w, hgt, label) {
    return s("svg", { viewBox: `0 0 ${w} ${hgt}`, width: w, role: "img", "aria-label": label || "" });
  }

  // Render a token string visibly: leading/inner spaces as "·", newlines as "↵", tabs as "→".
  function tokNodes(t) {
    const out = [];
    let buf = "";
    const flush = () => { if (buf) { out.push(document.createTextNode(buf)); buf = ""; } };
    for (const ch of t) {
      if (ch === " ") { flush(); out.push(h("span", { class: "sp", text: "·" })); }
      else if (ch === "\n") { flush(); out.push(h("span", { class: "nl", text: "↵" })); }
      else if (ch === "\t") { flush(); out.push(h("span", { class: "nl", text: "→" })); }
      else buf += ch;
    }
    flush();
    if (!out.length) out.push(h("span", { class: "sp", text: "∅" }));
    return out;
  }
  function chip(t, opts = {}) {
    const c = h(opts.tag || "span", { class: "tok" + (opts.cls ? " " + opts.cls : ""), title: opts.title, type: opts.tag === "button" ? "button" : null }, tokNodes(t));
    if (opts.sub != null) c.append(h("sub", { text: opts.sub }));
    return c;
  }
  function showTok(t) { return t.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→"); }

  const pct = (p, d = 1) => (p >= 0.9995 && d < 2 ? "100%" : (p * 100).toFixed(p < 0.001 && p > 0 ? 3 : p < 0.01 ? 2 : d) + "%");

  // Horizontal bar list. items: [{label, value, max?, text?, cls?}]
  function bars(container, items, max) {
    container.replaceChildren();
    const m = max || Math.max(...items.map((i) => i.value), 1e-9);
    for (const it of items) {
      const lab = h("span", { class: "lab" }, it.labelNode || tokNodes(it.label));
      const fill = h("div", { class: "fill" });
      fill.style.width = Math.max(0, Math.min(100, (it.value / m) * 100)) + "%";
      container.append(h("div", { class: "bar" + (it.cls ? " " + it.cls : "") }, lab, h("div", { class: "track" }, fill), h("span", { class: "val", text: it.text ?? pct(it.value) })));
    }
  }

  // Segmented buttons. options: [{label, value}] ; returns setter
  function seg(container, options, initial, onChange) {
    container.replaceChildren();
    const btns = options.map((o) => {
      const b = h("button", { class: "btn", type: "button", text: o.label, onclick: () => set(o.value, true) });
      b.dataset.v = String(o.value);
      container.append(b);
      return b;
    });
    function set(v, fire) {
      btns.forEach((b) => b.classList.toggle("on", b.dataset.v === String(v)));
      btns.forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === String(v)));
      if (fire) onChange(v);
    }
    set(initial, false);
    return set;
  }

  // Tooltip
  const tipEl = () => $("tip");
  function tip(e, text) {
    const t = tipEl();
    if (!text) { t.hidden = true; return; }
    t.textContent = text;
    t.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - t.offsetWidth - 8);
    const y = Math.min(e.clientY + 14, window.innerHeight - t.offsetHeight - 8);
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  function bindTip(node, fn) {
    node.addEventListener("pointermove", (e) => tip(e, fn()));
    node.addEventListener("pointerleave", (e) => tip(e, null));
  }

  // Heatmap grid in SVG. data[r][c] in 0..1 (opacity of accent). Returns svg.
  function heat(data, o = {}) {
    const rows = data.length, cols = data[0].length;
    const cw = o.cell || 14, chh = o.cellH || cw, L = o.left ?? 34, T = o.top ?? 18, B = o.bottom ?? 4;
    const W = L + cols * cw + 4, H = T + rows * chh + B;
    const svg = svgRoot(W, H, o.label);
    if (o.width) svg.setAttribute("width", o.width);
    const color = o.color || "var(--accent)";
    for (let r = 0; r < rows; r++) {
      const rr = o.flip ? rows - 1 - r : r;
      for (let c = 0; c < cols; c++) {
        const v = data[rr][c];
        const rect = s("rect", { x: L + c * cw, y: T + r * chh, width: cw - 1, height: chh - 1, rx: 1.5,
          fill: v == null ? "transparent" : color, "fill-opacity": v == null ? 0 : Math.max(0.04, Math.min(1, v)) }, svg);
        if (o.tip) bindTip(rect, () => o.tip(rr, c, v));
        if (o.click) { rect.style.cursor = "pointer"; rect.addEventListener("click", () => o.click(rr, c, v)); }
        if (o.mark && o.mark(rr, c)) s("rect", { x: L + c * cw - 0.5, y: T + r * chh - 0.5, width: cw, height: chh, fill: "none", stroke: "var(--hot)", "stroke-width": 2, rx: 2, "pointer-events": "none" }, svg);
      }
    }
    if (o.rowLabel) for (let r = 0; r < rows; r++) {
      const rr = o.flip ? rows - 1 - r : r;
      const lab = o.rowLabel(rr);
      if (lab != null) s("text", { x: L - 6, y: T + r * chh + chh / 2 + 3.5, "text-anchor": "end", text: lab }, svg);
    }
    if (o.colLabel) for (let c = 0; c < cols; c++) {
      const lab = o.colLabel(c);
      if (lab != null) s("text", { x: L + c * cw + cw / 2, y: T - 6, "text-anchor": "middle", text: lab }, svg);
    }
    return svg;
  }

  // Simple bar/line chart over an index axis.
  function chart(container, o) {
    const W = o.width || 760, H = o.height || 220, L = o.left ?? 46, R = 12, T = 14, B = o.bottom ?? 30;
    const svg = svgRoot(W, H, o.label);
    const n = o.n;
    const ymax = o.ymax ?? Math.max(...o.series.flatMap((se) => se.values.filter((v) => v != null)));
    const ymin = o.ymin ?? 0;
    const log = o.log;
    const yv = (v) => log ? Math.log10(v) : v;
    const y0 = yv(log ? (o.ymin || 1) : ymin), y1 = yv(ymax);
    const Y = (v) => T + (H - T - B) * (o.invert ? (yv(v) - y0) / (y1 - y0 || 1) : 1 - (yv(v) - y0) / (y1 - y0 || 1));
    const bw = (W - L - R) / n;
    const X = (i) => L + i * bw + bw / 2;
    // grid
    for (const t of o.yticks || []) {
      s("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "axis", "stroke-dasharray": "2 3" }, svg);
      s("text", { x: L - 6, y: Y(t) + 3.5, "text-anchor": "end", text: o.yfmt ? o.yfmt(t) : t }, svg);
    }
    s("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axis" }, svg);
    for (let i = 0; i < n; i++) if (!o.xevery || i % o.xevery === 0 || i === n - 1)
      s("text", { x: X(i), y: H - B + 14, "text-anchor": "middle", text: o.xlabel ? o.xlabel(i) : i }, svg);
    if (o.xtitle) s("text", { x: L + (W - L - R) / 2, y: H - 2, "text-anchor": "middle", text: o.xtitle }, svg);
    if (o.bands) for (const [a, b, col] of o.bands)
      s("rect", { x: L + a * bw, y: T, width: (b - a) * bw, height: H - T - B, fill: col || "var(--hot)", "fill-opacity": 0.08 }, svg);
    const nb = o.series.filter((se) => se.type !== "line").length;
    let bi = 0;
    for (const se of o.series) {
      if (se.type === "line") {
        const pts = se.values.map((v, i) => (v == null ? null : `${X(i)},${Y(Math.max(v, log ? (o.ymin || 1) : -Infinity))}`)).filter(Boolean);
        s("polyline", { points: pts.join(" "), fill: "none", stroke: se.color, "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
        se.values.forEach((v, i) => {
          if (v == null) return;
          const c = s("circle", { cx: X(i), cy: Y(Math.max(v, log ? (o.ymin || 1) : -Infinity)), r: se.dot ?? 3, fill: se.dotColor ? se.dotColor(i) : se.color }, svg);
          if (o.tip) bindTip(c, () => o.tip(i, se));
        });
      } else {
        const w = (bw * 0.78) / nb;
        se.values.forEach((v, i) => {
          if (v == null) return;
          const base = Y(Math.max(ymin, 0)), top = Y(v);
          const r = s("rect", { x: L + i * bw + bw * 0.11 + bi * w, y: Math.min(base, top), width: Math.max(1, w - 1), height: Math.max(0.5, Math.abs(base - top)), rx: 1.5,
            fill: se.colorAt ? se.colorAt(i, v) : se.color }, svg);
          if (o.tip) bindTip(r, () => o.tip(i, se));
        });
        bi++;
      }
    }
    if (o.legend) {
      const lg = h("div", { class: "seg", style: "font:500 .78rem var(--font-mono);gap:1rem;color:var(--muted)" });
      o.series.forEach((se) => lg.append(h("span", {}, h("span", { style: `display:inline-block;width:.75rem;height:.75rem;border-radius:2px;background:${se.color};margin-right:.35rem;vertical-align:-1px` }), se.name)));
      container.replaceChildren(svg, lg);
    } else container.replaceChildren(svg);
    return svg;
  }

  // Theme toggle (system -> light -> dark), remembered per viewer.
  function initTheme() {
    const btn = $("theme-btn");
    const order = ["system", "light", "dark"];
    let cur = "system";
    try { cur = localStorage.getItem("tio-theme") || "system"; } catch (e) {}
    const apply = () => {
      if (cur === "system") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", cur);
      btn.textContent = "Theme: " + cur;
    };
    btn.addEventListener("click", () => {
      cur = order[(order.indexOf(cur) + 1) % 3];
      try { localStorage.setItem("tio-theme", cur); } catch (e) {}
      apply();
    });
    apply();
  }

  function initSpy() {
    const links = [...document.querySelectorAll("#toc-list a")];
    const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
    const io = new IntersectionObserver((ents) => {
      ents.forEach((en) => {
        if (en.isIntersecting) {
          const i = secs.indexOf(en.target);
          links.forEach((l, j) => l.classList.toggle("on", j === i));
          const on = links[i];
          if (on && on.scrollIntoView && window.innerWidth <= 1080) on.parentElement.parentElement.scrollTo({ left: on.offsetLeft - 80, behavior: "smooth" });
        }
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    secs.forEach((s_) => s_ && io.observe(s_));
  }

  function range(id, fmt, onInput) {
    const inp = $(id), out = $(id + "-out");
    const upd = () => { if (out) out.textContent = fmt(+inp.value); onInput(+inp.value); };
    inp.addEventListener("input", upd);
    return upd;
  }

  function select(id, options, onChange) {
    const sel = $(id);
    sel.replaceChildren(...options.map((o, i) => h("option", { value: i, text: o })));
    sel.addEventListener("change", () => onChange(+sel.value));
    return sel;
  }

  window.UI = { $, h, s, svgRoot, chip, tokNodes, showTok, pct, bars, seg, tip, bindTip, heat, chart, initTheme, initSpy, range, select };
})();
