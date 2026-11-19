// Widgets for the hero and lessons 01-05. All numbers come from window.MD (data/model_data.js).
(function () {
  const { $, h, s, chip, bars, seg, bindTip, chart, pct, showTok, svgRoot, range, select } = UI;
  const M = window.MD;
  const RUN_TOKENS = M.l07.sentences[0].tokens;
  const isParis = (t) => t === " Paris";

  // ------------------------------------------------------------ hero
  function hero() {
    const row = $("hero-tokens");
    RUN_TOKENS.forEach((t) => row.append(chip(t)));
    row.append(h("span", { class: "arrow", text: "→" }));
    row.append(h("span", { class: "answer-chip" }, "·Paris", h("small", { text: "88%" })));
    $("foot-model").textContent = M.meta.model + ` (${(M.meta.params / 1e6).toFixed(1)}M parameters)`;
    $("foot-date").textContent = M.meta.generated;
  }

  // ------------------------------------------------------------ 01
  function l01() {
    const g = $("w01-greedy");
    M.l01.greedy.forEach((x, i) => {
      const box = h("div", { class: "out-box" }, x.prompt, h("span", { class: "gen" + (i === 0 ? " hot" : ""), text: x.completion }));
      if (i === 0) box.style.gridColumn = "1 / -1";
      g.append(box);
    });
    const sm = $("w01-sampled");
    M.l01.sampled.forEach((t, i) => sm.append(h("div", { class: "out-box" }, h("b", { class: "muted", text: `run ${i + 1}\n` }), "Once upon a time, a little robot", h("span", { class: "gen", text: t }))));
  }

  // ------------------------------------------------------------ 02
  const TS = M.temps;
  function probsAt(d, ti) {
    const T = TS[ti];
    return d.logits.map((l) => Math.exp(l / T - d.lse[ti]));
  }
  function l02() {
    const dists = M.l02.dists;
    let di = 0, ti = 19;
    const draw = () => {
      const d = dists[di], p = probsAt(d, ti);
      bars($("w02-bars"), d.tokens.slice(0, 15).map((t, i) => ({ label: t, value: p[i], cls: isParis(t) ? "hot" : "" })), 1);
      const rest = 1 - p.slice(0, 15).reduce((a, b) => a + b, 0);
      bars($("w02-rest"), [{ labelNode: [h("span", { class: "muted", text: "49,137 others" })], value: Math.max(rest, 0), cls: "cut" }], 1);
    };
    select("w02-prompt", dists.map((d) => JSON.stringify(d.prompt).slice(1, -1)), (v) => { di = v; draw(); });
    range("w02-temp", (v) => TS[v].toFixed(2), (v) => { ti = v; draw(); })();

    // per-position guesses
    const row = $("w02-positions"), det = $("w02-pos-detail");
    const pos = M.l02.positions;
    const show = (i) => {
      [...row.children].forEach((c, j) => c.classList.toggle("sel", j === i));
      const p = pos[i];
      det.replaceChildren(h("span", {}, "After ", h("b", { text: `"${p.seen}"` }), " the model's top guess is ", chip(p.guess, { cls: isParis(p.guess) ? "hot" : "" }), ` with ${pct(p.p)}.`));
    };
    pos.forEach((p, i) => {
      const c = chip(p.token, { tag: "button" });
      c.style.cursor = "pointer";
      c.addEventListener("click", () => show(i));
      c.addEventListener("pointerenter", () => show(i));
      row.append(c);
    });
    show(pos.length - 1);

    // surprise
    const sp = M.l02.surprise;
    const names = ["…Paris, France.", "…Berlin, Germany.", "quick brown fox"];
    const order = [1, 2, 0];
    const drawS = (k) => {
      const x = sp[order[k]];
      chart($("w02-surprise"), {
        n: x.tokens.length, width: 760, height: 230, bottom: 44,
        series: [{ values: x.loss, colorAt: (i) => (/Paris|Berlin/.test(x.tokens[i]) ? "var(--hot)" : "var(--accent)") }],
        ymax: 10, yticks: [0, 2, 4, 6, 8, 10], xlabel: (i) => showTok(x.tokens[i]).slice(0, 8),
        tip: (i) => `${showTok(x.tokens[i])}\nloss ${x.loss[i]}  →  model gave it ${pct(Math.exp(-x.loss[i]))}`,
        label: "loss per token",
      });
    };
    seg($("w02-surprise-pick"), names.map((n, i) => ({ label: n, value: i })), 0, drawS);
    drawS(0);
  }

  // ------------------------------------------------------------ 03
  function l03() {
    const st = M.l03.steps;
    let i = 0, timer = null;
    const draw = () => {
      const gen = st.slice(0, i).map((x) => x.tokens[0]).join("");
      $("w03-text").replaceChildren(M.l03.prompt, h("span", { class: "gen", text: gen }), h("span", { class: "muted", text: i < st.length ? " ▍" : "" }));
      $("w03-count").textContent = `step ${i} of ${st.length}`;
      const cur = st[Math.min(i, st.length - 1)];
      bars($("w03-bars"), cur.tokens.map((t, j) => ({ label: t, value: cur.probs[j], cls: j === 0 ? "hot" : "" })), 1);
      const n90 = cur.n90;
      const box = $("w03-n90");
      box.replaceChildren(
        h("b", { text: n90.toLocaleString() + (n90 === 1 ? " token" : " tokens") }),
        h("span", { text: "needed to cover 90% of the probability at this step" }),
      );
      const holder = h("div", { style: "margin-top:.75rem" });
      box.append(holder);
      chart(holder, {
        n: st.length, width: 420, height: 170, left: 40, log: true, ymin: 1, ymax: 1000, yticks: [1, 10, 100, 1000],
        series: [{ type: "line", values: st.map((x) => x.n90), color: "var(--accent)", dotColor: (j) => (j === Math.min(i, st.length - 1) ? "var(--hot)" : "var(--accent)") }],
        xlabel: (j) => j + 1, xevery: 3, xtitle: "step",
        tip: (j) => `step ${j + 1}: next = ${showTok(st[j].tokens[0])}\n${st[j].n90} tokens for 90%`, label: "uncertainty per step",
      });
      $("w03-back").disabled = i === 0;
      $("w03-step").disabled = i >= st.length;
    };
    const stop = () => { clearInterval(timer); timer = null; $("w03-play").textContent = "play"; };
    $("w03-step").addEventListener("click", () => { if (i < st.length) i++; draw(); });
    $("w03-back").addEventListener("click", () => { if (i > 0) i--; draw(); });
    $("w03-reset").addEventListener("click", () => { stop(); i = 0; draw(); });
    $("w03-play").addEventListener("click", () => {
      if (timer) return stop();
      if (i >= st.length) i = 0;
      $("w03-play").textContent = "pause";
      timer = setInterval(() => { i++; draw(); if (i >= st.length) stop(); }, 650);
    });
    draw();

    // sampling lab
    const dists = M.l02.dists;
    let di = 0, ti = 19, k = 50, tp = 1;
    let tally = new Map();
    let kept = [];
    const drawLab = () => {
      const d = dists[di];
      const p = probsAt(d, ti);
      let cum = 0;
      kept = p.map((v, j) => { const keep = j < k && cum < tp; cum += v; return keep; });
      const mass = p.reduce((a, v, j) => a + (kept[j] ? v : 0), 0);
      bars($("w03s-bars"), d.tokens.slice(0, 15).map((t, j) => ({
        label: t, value: kept[j] ? p[j] / mass : p[j], cls: kept[j] ? (isParis(t) ? "hot" : "") : "cut",
        text: kept[j] ? pct(p[j] / mass) : "cut",
      })), 1);
      const nk = kept.filter(Boolean).length;
      $("w03s-bars").append(h("p", { class: "caption", text: `${nk} candidate${nk === 1 ? "" : "s"} left after top-k and top-p (from the top 50).` }));
    };
    const roll = (n) => {
      const d = dists[di], p = probsAt(d, ti);
      const w = p.map((v, j) => (kept[j] ? v : 0));
      const tot = w.reduce((a, b) => a + b, 0);
      for (let r = 0; r < n; r++) {
        let x = Math.random() * tot, j = 0;
        while (j < w.length - 1 && x > w[j]) { x -= w[j]; j++; }
        tally.set(d.tokens[j], (tally.get(d.tokens[j]) || 0) + 1);
      }
      drawRolls();
    };
    const drawRolls = () => {
      const box = $("w03s-rolls");
      box.replaceChildren();
      if (!tally.size) { box.append(h("span", { class: "caption", text: "No rolls yet. Each roll picks one token from the surviving candidates." })); return; }
      [...tally.entries()].sort((a, b) => b[1] - a[1]).forEach(([t, c]) => box.append(chip(t, { sub: "×" + c, cls: isParis(t) ? "hot" : "" })));
    };
    select("w03s-prompt", dists.map((d) => JSON.stringify(d.prompt).slice(1, -1)), (v) => { di = v; tally = new Map(); drawLab(); drawRolls(); });
    range("w03s-t", (v) => TS[v].toFixed(2), (v) => { ti = v; drawLab(); })();
    range("w03s-k", (v) => String(v), (v) => { k = v; drawLab(); })();
    range("w03s-p", (v) => v + "%", (v) => { tp = v / 100; drawLab(); })();
    $("w03s-roll").addEventListener("click", () => roll(1));
    $("w03s-roll100").addEventListener("click", () => roll(100));
    $("w03s-clear").addEventListener("click", () => { tally = new Map(); drawRolls(); });
    drawRolls();
  }

  // ------------------------------------------------------------ 04
  function l04() {
    let tk = null;
    const getTk = () => (tk = tk || makeTokenizer(window.TOK));
    const inp = $("w04-input");
    const presets = [
      ["Our sentence", M.meta.run],
      ["English", "Knowledge is the greatest wealth."],
      ["Hindi", "ज्ञान सबसे बड़ा धन है।"],
      ["French", "La connaissance est la plus grande richesse."],
      ["Numbers", "In 2025 we sold 1,234,567 units for $89.99 each."],
      ["Code", "def add(a, b):\n    return a + b"],
      ["Rare words", "Antidisestablishmentarianism is a sesquipedalian word."],
      ["Emoji", "I love pizza 🍕🍕 and coffee ☕"],
    ];
    const pr = $("w04-presets");
    presets.forEach(([n, t]) => pr.append(h("button", { class: "btn", type: "button", text: n, onclick: () => { inp.value = t; draw(); } })));
    const draw = () => {
      const t = getTk();
      const { ids } = t.encode(inp.value);
      const showIds = $("w04-ids").checked, raw = $("w04-raw").checked;
      const box = $("w04-tokens");
      box.replaceChildren();
      ids.slice(0, 400).forEach((id) => {
        const text = raw ? t.rawToken(id) : t.decodeOne(id);
        const c = chip(text, { sub: showIds ? id : null, cls: !raw && text === " Paris" ? "hot" : "" });
        c.title = `id ${id} · raw "${t.rawToken(id)}"`;
        box.append(c);
      });
      if (ids.length > 400) box.append(h("span", { class: "caption", text: `…and ${ids.length - 400} more` }));
      const chars = [...inp.value].length, bytes = new TextEncoder().encode(inp.value).length;
      $("w04-stats").replaceChildren(
        h("div", { class: "stat" }, h("b", { text: ids.length }), h("span", { text: "tokens" })),
        h("div", { class: "stat" }, h("b", { text: chars }), h("span", { text: "characters" })),
        h("div", { class: "stat" }, h("b", { text: bytes }), h("span", { text: "bytes (UTF-8)" })),
        h("div", { class: "stat" }, h("b", { text: ids.length ? (chars / ids.length).toFixed(2) : "–" }), h("span", { text: "characters per token" })),
      );
    };
    inp.value = M.meta.run;
    inp.addEventListener("input", draw);
    $("w04-ids").addEventListener("change", draw);
    $("w04-raw").addEventListener("change", draw);
    draw();

    const run = $("w04-run");
    RUN_TOKENS.forEach((t) => run.append(chip(t, { cls: t === " E" || t === "iffel" ? "hot" : "" })));

    // first 24 numbers of " Paris"
    const v = M.l04.first_numbers, mx = Math.max(...v.map(Math.abs));
    const W = 24 * 31 + 8, svg = svgRoot(W, 62, "first 24 embedding numbers for Paris");
    v.forEach((x, i) => {
      s("rect", { x: 4 + i * 31, y: 4, width: 29, height: 32, rx: 3, fill: x >= 0 ? "var(--accent)" : "var(--hot)", "fill-opacity": 0.12 + 0.88 * Math.abs(x) / mx }, svg);
      s("text", { x: 4 + i * 31 + 14.5, y: 52, "text-anchor": "middle", text: x.toFixed(2), style: "font-size:8.5px" }, svg);
    });
    $("w04-vec").replaceChildren(h("div", { class: "scroll-x" }, svg));

    // neighbours
    const words = ["Paris", "king", "Monday", "Tower", "located", "city", "happy", "dog", "seven", "function", "pizza", "teacher", "red"].filter((w) => M.l04.neighbours[w]);
    const showN = (w) => {
      bars($("w04-neigh"), M.l04.neighbours[w].map(([t, sim]) => ({ label: t, value: sim, text: sim.toFixed(2) })), 1);
      highlight(w);
    };
    seg($("w04-words"), words.map((w) => ({ label: w, value: w })), "Paris", showN);

    // scatter
    const pts = M.l04.scatter;
    const groups = [...new Set(pts.map((p) => p.g))];
    const palette = ["#2a9d8f", "#e76f51", "#7b61ff", "#d4a106", "#3a86ff", "#d1495b", "#5fa52c", "#b25ac2", "#8a8f98"];
    const col = (g) => palette[groups.indexOf(g) % palette.length];
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const SW = 460, SH = 380, P = 30;
    const svg2 = svgRoot(SW, SH, "embedding scatter");
    const X = (x) => P + (SW - 2 * P) * (x - x0) / (x1 - x0), Y = (y) => SH - P - (SH - 2 * P) * (y - y0) / (y1 - y0);
    const nodes = {};
    pts.forEach((p) => {
      const g = s("g", {}, svg2);
      s("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: col(p.g), "fill-opacity": 0.9 }, g);
      const t = s("text", { x: X(p.x) + 6, y: Y(p.y) + 3.5, text: p.w, style: "font-size:9.5px" }, g);
      bindTip(g, () => `${p.w}  (${p.g})`);
      nodes[p.w] = { g, t };
    });
    const highlight = (w) => {
      const near = new Set((M.l04.neighbours[w] || []).map(([t]) => t.trim()));
      Object.entries(nodes).forEach(([k, n]) => {
        const on = k === w || near.has(k);
        n.g.style.opacity = on ? 1 : 0.35;
        n.t.setAttribute("class", on ? "ink" : "");
        n.t.style.fontWeight = k === w ? "700" : "400";
      });
    };
    $("w04-scatter").replaceChildren(svg2);
    const lg = $("w04-legend");
    groups.forEach((g) => lg.append(h("span", {}, h("span", { style: `display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:${col(g)};margin-right:.3rem` }), g)));
    showN("Paris");
  }

  // ------------------------------------------------------------ 05
  function l05() {
    const m = M.meta;
    const attnPer = m.hidden * m.hidden * 2 + m.hidden * m.kv_heads * m.head_dim * 2;
    const mlpPer = 3 * m.hidden * m.mlp;
    const fmt = (n) => n.toLocaleString();
    const info = {
      embed: ["Embedding table", `${fmt(m.vocab)} × ${m.hidden}`, m.vocab * m.hidden, "Turns each token ID into a vector by looking up one row. Nothing is calculated. The same matrix is reused by the LM head at the top (tied embeddings).", "l04"],
      norm1: ["RMSNorm (before attention)", `${m.hidden}`, m.hidden, "Rescales the stream to a standard size so attention always sees numbers in a familiar range. The stream itself is not changed; only the copy that attention reads.", "l05-norm"],
      attn: ["Attention", `q ${m.hidden}→${m.heads * m.head_dim} · k,v ${m.hidden}→${m.kv_heads * m.head_dim} · o ${m.hidden}→${m.hidden}`, attnPer, `The only place where tokens exchange information. ${m.heads} heads each decide which earlier tokens to read from. Keys and values come in ${m.kv_heads} groups shared by ${m.heads / m.kv_heads} heads each (grouped-query attention).`, "l07"],
      add1: ["Residual add", "x = x + update", 0, "The update is added to the stream instead of replacing it. This is why layers can be read early (lesson 06) and skipped without total collapse (lesson 09). No parameters at all.", "l06"],
      norm2: ["RMSNorm (before MLP)", `${m.hidden}`, m.hidden, "Same job as the first norm, for the MLP's input.", "l05-norm"],
      mlp: ["MLP (SwiGLU)", `gate, up ${m.hidden}→${m.mlp} · down ${m.mlp}→${m.hidden}`, mlpPer, "Works on each token separately: expands the vector to 1,536 numbers, gates it, and compresses it back. The largest part of the model, and where much of its stored knowledge seems to live.", "l05-mlp"],
      add2: ["Residual add", "x = x + update", 0, "The MLP's result is added to the stream, and the stream moves on to the next layer.", "l06"],
      fnorm: ["Final RMSNorm", `${m.hidden}`, m.hidden, "One last rescale of the stream after all 30 layers.", "l06"],
      head: ["LM head", `${m.hidden} → ${fmt(m.vocab)}`, 0, "Scores every vocabulary token by comparing the final vector with each token's embedding row (the same table as at the bottom). The output is 49,152 logits, one per token.", "l02"],
    };
    const W = 400, H = 660;
    const svg = svgRoot(W, H, "architecture diagram");
    s("defs", {}, svg).innerHTML = '<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--muted)"/></marker>';
    // block frame
    s("rect", { x: 14, y: 196, width: 372, height: 344, rx: 10, fill: "none", stroke: "var(--line)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, svg);
    s("text", { x: 372, y: 216, "text-anchor": "end", class: "ink", text: "× 30 layers", style: "font-size:12px;font-weight:600" }, svg);
    // stream
    s("line", { x1: 70, y1: 566, x2: 70, y2: 176, class: "stream" }, svg);
    s("text", { x: 34, y: 520, text: "residual stream", transform: "rotate(-90 34 520)", style: "font-size:10.5px;fill:var(--hot)" }, svg);
    // branches
    const flow = (d) => s("path", { d, class: "flow", "marker-end": "url(#ah)" }, svg);
    flow("M70 505 H160 V488");
    flow("M210 466 H232");
    flow("M292 446 V404 H84");
    flow("M70 372 H160 V355");
    flow("M210 333 H232");
    flow("M292 313 V268 H84");
    flow("M190 640 V606");
    const box = (key, x, y, w, hh, label, sub) => {
      const g = s("g", { class: "arch-box", tabindex: 0, role: "button", "aria-label": label }, svg);
      s("rect", { x, y, width: w, height: hh, rx: 7 }, g);
      s("text", { x: x + w / 2, y: y + (sub ? hh / 2 - 2 : hh / 2 + 4.5), "text-anchor": "middle", text: label }, g);
      if (sub) s("text", { x: x + w / 2, y: y + hh / 2 + 12, "text-anchor": "middle", class: "sub", text: sub }, g);
      g.addEventListener("click", () => pick(key));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(key); } });
      g.dataset.k = key;
      return g;
    };
    const plus = (key, cy) => {
      const g = s("g", { class: "arch-box", tabindex: 0, role: "button", "aria-label": "residual add" }, svg);
      s("circle", { cx: 70, cy, r: 13, fill: "var(--surface)", stroke: "var(--hot)", "stroke-width": 2 }, g);
      s("text", { x: 70, y: cy + 5, "text-anchor": "middle", text: "+", style: "font-size:17px;fill:var(--hot);font-weight:700" }, g);
      g.addEventListener("click", () => pick(key));
      g.dataset.k = key;
    };
    box("embed", 40, 566, 300, 40, "Embedding table", "49,152 × 576");
    s("text", { x: 190, y: 654, "text-anchor": "middle", class: "ink", text: `token IDs: [${M.l04.examples.find((e) => e.text === M.meta.run).ids.slice(0, 4).join(", ")}, …]` }, svg);
    box("norm1", 112, 446, 98, 40, "RMSNorm");
    box("attn", 232, 446, 134, 40, "Attention", "9 heads");
    plus("add1", 404);
    box("norm2", 112, 313, 98, 40, "RMSNorm");
    box("mlp", 232, 313, 134, 40, "MLP", "576→1536→576");
    plus("add2", 268);
    box("fnorm", 40, 136, 300, 36, "Final RMSNorm");
    box("head", 40, 80, 300, 40, "LM head", "576 → 49,152 scores");
    flow("M190 80 V48");
    s("text", { x: 190, y: 38, "text-anchor": "middle", class: "ink", text: "softmax → next-token probabilities", style: "font-size:12px" }, svg);
    s("line", { x1: 70, y1: 176, x2: 70, y2: 172, class: "stream" }, svg);
    $("w05-arch").replaceChildren(svg);

    const pick = (key) => {
      svg.querySelectorAll(".arch-box").forEach((g) => g.classList.toggle("on", g.dataset.k === key));
      const [title, shape, params, text, lesson] = info[key];
      $("w05-info").replaceChildren(
        h("p", { class: "eyebrow", text: "part" }),
        h("h3", { text: title, style: "font-size:1.35rem" }),
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("b", { text: params ? (params >= 1e6 ? (params / 1e6).toFixed(2) + "M" : fmt(params)) : "0", style: "font-size:1.3rem" }), h("span", { text: key === "head" ? "parameters (shared with embedding)" : key.startsWith("norm") || key.startsWith("add") || key === "attn" || key === "mlp" ? "parameters per layer" : "parameters" })),
          h("div", { class: "stat", style: "min-width:0" }, h("b", { text: shape, style: "font:600 .9rem var(--font-mono);overflow-wrap:anywhere" }), h("span", { text: "shape" }))),
        h("p", { text }),
        h("p", {}, h("a", { href: "#" + lesson, text: lesson.startsWith("l05-") ? "Explained below ↓" : `More in lesson ${lesson.slice(1)} →` })),
      );
    };
    pick("attn");

    // params
    const P = M.l05.params;
    const tot = Object.values(P).reduce((a, b) => a + b, 0);
    const names = { mlp: "MLP", embedding: "Embedding (+ LM head)", attention: "Attention", norms: "Norms" };
    bars($("w05-params"), ["mlp", "embedding", "attention", "norms"].map((k) => ({
      labelNode: [h("span", { text: names[k], style: "font-family:var(--font-body)" })], value: P[k] / tot,
      text: (P[k] / 1e6).toFixed(P[k] < 1e6 ? 3 : 1) + "M",
    })), 1);
    $("w05-params").querySelectorAll(".bar").forEach((b) => (b.style.gridTemplateColumns = "11rem minmax(0,1fr) 4.5rem"));

    // stream
    const L = M.l05;
    const caps = {
      w: "Length of the update each sublayer adds to the stream at the last token. Attention writes (teal) are small almost everywhere except layers 23 and 24. MLP writes (amber) grow steadily towards the top of the model, as it turns its context into a prediction.",
      n: "Size of the stream itself going into each layer. It grows as updates accumulate: about 3 after the embedding lookup, nearly 700 by layer 29. The final RMSNorm scales it back down before scoring.",
    };
    const drawStream = (mode) => {
      if (mode === "w") chart($("w05-stream"), {
        n: 30, height: 240, series: [{ name: "attention write", values: L.attn_write, color: "var(--accent)" }, { name: "MLP write", values: L.mlp_write, color: "var(--hot)" }],
        yticks: [0, 100, 200, 300, 400], xevery: 1, xtitle: "layer", legend: true, bands: [[23, 25, "var(--accent)"]],
        tip: (i) => `layer ${i}\nattention wrote ${L.attn_write[i]}\nMLP wrote ${L.mlp_write[i]}`, label: "writes per layer",
      });
      else chart($("w05-stream"), {
        n: 30, height: 240, series: [{ type: "line", values: L.norms, color: "var(--hot)" }],
        yticks: [0, 200, 400, 600], xevery: 1, xtitle: "layer (stream going in)", tip: (i) => `into layer ${i}: size ${L.norms[i]}`, label: "stream size",
      });
      $("w05-stream-cap").textContent = caps[mode];
    };
    seg($("w05-mode"), [{ label: "what each layer writes", value: "w" }, { label: "size of the stream", value: "n" }], "w", drawStream);
    drawStream("w");
  }

  hero(); l01(); l02(); l03(); l04(); l05();
})();
