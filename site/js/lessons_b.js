// Widgets for lessons 06-10. All numbers come from window.MD (data/model_data.js).
(function () {
  const { $, h, s, chip, bars, seg, bindTip, chart, heat, pct, showTok, svgRoot, range, select } = UI;
  const M = window.MD;
  const NL = M.meta.layers, NH = M.meta.heads;
  const isParis = (t) => t === " Paris";
  const promptLabel = (p) => JSON.stringify(p).slice(1, -1);

  // ------------------------------------------------------------ 06 logit lens
  function l06() {
    const lens = M.l06;
    let pi = 0, L = 29;
    const draw = () => {
      const e = lens[pi], row = e.last[L];
      bars($("w06-bars"), row.top.map((t, j) => ({ label: t, value: row.p[j], cls: t === e.final ? "hot" : "" })), 1);
      const fin = e.final;
      $("w06-rank-cap").replaceChildren(
        "Rank of the final answer ", chip(fin, { cls: "hot" }), " after each layer (log scale, 1 = top). Right now: ",
        h("b", { text: `rank ${row.rank.toLocaleString()}` }), `, ${pct(row.pf)}.`);
      chart($("w06-rank"), {
        n: NL, width: 440, height: 200, left: 48, log: true, invert: true, ymin: 1, ymax: 20000, yticks: [1, 10, 100, 1000, 10000],
        yfmt: (t) => t.toLocaleString(),
        series: [{ type: "line", values: e.last.map((x) => x.rank), color: "var(--hot)", dot: 3, dotColor: (j) => (j === L ? "var(--ink)" : "var(--hot)") }],
        xevery: 5, xtitle: "after layer", tip: (j) => `after layer ${j}: ${showTok(fin)} is rank ${e.last[j].rank.toLocaleString()} (${pct(e.last[j].pf)})\nbest guess: ${showTok(e.last[j].top[0])}`,
        label: "rank across layers",
      });
    };
    select("w06-prompt", lens.map((e) => promptLabel(e.prompt)), (v) => { pi = v; draw(); });
    range("w06-layer", (v) => String(v), (v) => { L = v; draw(); })();

    // grid
    const drawGrid = (gi) => {
      const e = lens[gi], T = e.tokens.length;
      const cw = 76, ch = 17, left = 40, top = 6, bottom = 30;
      const W = left + T * cw + 4, H = top + NL * ch + bottom;
      const svg = svgRoot(W, H, "logit lens grid");
      for (let r = 0; r < NL; r++) {
        const layer = NL - 1 - r;
        if (layer % 5 === 0 || layer === NL - 1) s("text", { x: left - 6, y: top + r * ch + ch / 2 + 3.5, "text-anchor": "end", text: layer }, svg);
        for (let c = 0; c < T; c++) {
          const [tok, p] = e.grid[layer][c];
          const g = s("g", {}, svg);
          s("rect", { x: left + c * cw, y: top + r * ch, width: cw - 2, height: ch - 2, rx: 2, fill: tok === e.final && c === T - 1 ? "var(--hot)" : "var(--accent)", "fill-opacity": 0.06 + 0.94 * p }, g);
          const label = showTok(tok).slice(0, 9);
          s("text", { x: left + c * cw + 4, y: top + r * ch + ch / 2 + 3, text: label, style: `font-size:9.5px;fill:${p > 0.5 ? "var(--surface)" : "var(--ink)"}` }, g);
          bindTip(g, () => `after layer ${layer}, position "${showTok(e.tokens[c])}"\ntop guess ${showTok(tok)}  ${pct(p)}`);
        }
      }
      for (let c = 0; c < T; c++) s("text", { x: left + c * cw + cw / 2 - 1, y: H - 10, "text-anchor": "middle", class: "ink", text: showTok(e.tokens[c]).slice(0, 10), style: "font-weight:600" }, svg);
      $("w06-grid").replaceChildren(svg);
    };
    select("w06g-prompt", lens.map((e) => promptLabel(e.prompt)), drawGrid);
    drawGrid(0);
  }

  // ------------------------------------------------------------ 07 attention
  function l07() {
    const S = M.l07.sentences;
    const rep = M.l07.rep;
    let si = 0, L = 23, H = 4, q = S[0].tokens.length - 1;
    const sinkHead = rep.small_heads[3];
    const prevHead = rep.small_heads[0], indHead = rep.small_heads[1];
    const presets = [
      { label: "Eiffel lookup: L23 H4", l: 23, h: 4, s: 0 },
      { label: `previous-token: L${prevHead.layer} H${prevHead.head}`, l: prevHead.layer, h: prevHead.head, s: 0 },
      { label: `induction: L${indHead.layer} H${indHead.head}`, l: indHead.layer, h: indHead.head, s: 1 },
      { label: `sink: L${sinkHead.layer} H${sinkHead.head}`, l: sinkHead.layer, h: sinkHead.head, s: 0 },
    ];
    const weights = () => {
      const A = S[si].att;
      if (H === "avg") {
        const T = S[si].tokens.length;
        return Array.from({ length: T }, (_, r) => Array.from({ length: T }, (_, c) => A[L].reduce((a, hd) => a + hd[r][c], 0) / NH));
      }
      return A[L][H];
    };
    const draw = () => {
      const toks = S[si].tokens, T = toks.length, W = weights();
      // arcs
      const rowH = 26, Wd = 360, Hh = T * rowH + 30, xl = 110, xr = 250;
      const svg = svgRoot(Wd, Hh, "attention lines");
      s("text", { x: xl, y: 14, "text-anchor": "end", text: "looked at (keys)" }, svg);
      s("text", { x: xr, y: 14, text: "looking (query)" }, svg);
      for (let k = 0; k <= q; k++) {
        const w = W[q][k];
        if (w < 0.01) continue;
        s("line", { x1: xr - 6, y1: 30 + q * rowH, x2: xl + 6, y2: 30 + k * rowH, stroke: "var(--hot)", "stroke-width": 1 + 9 * w, "stroke-opacity": 0.15 + 0.85 * w, "stroke-linecap": "round" }, svg);
      }
      toks.forEach((t, i) => {
        const y = 30 + i * rowH;
        const w = i <= q ? W[q][i] : null;
        s("text", { x: xl, y: y + 4, "text-anchor": "end", class: "ink", text: showTok(t), style: `font-weight:${w > 0.3 ? 700 : 400};opacity:${i <= q ? 1 : 0.35}` }, svg);
        if (w != null) s("text", { x: 8, y: y + 4, text: (w * 100).toFixed(0) + "%", style: "font-size:10px" }, svg);
        const g = s("g", { style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": "query " + t }, svg);
        s("rect", { x: xr - 4, y: y - 10, width: 104, height: 20, rx: 4, fill: i === q ? "var(--hot-soft)" : "transparent", stroke: i === q ? "var(--hot)" : "none" }, g);
        s("text", { x: xr + 2, y: y + 4, class: "ink", text: showTok(t), style: `font-weight:${i === q ? 700 : 400}` }, g);
        g.addEventListener("click", () => { q = i; draw(); });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter") { q = i; draw(); } });
      });
      $("w07-arcs").replaceChildren(svg);
      // matrix
      const m = heat(W.map((r, i) => r.map((v, j) => (j > i ? null : v))), {
        cell: Math.min(26, Math.floor(300 / T)), left: 70, top: 70, label: "attention matrix",
        rowLabel: (r) => showTok(toks[r]).slice(0, 9), colLabel: () => null,
        tip: (r, c, v) => (v == null ? "future token: masked" : `${showTok(toks[r])} → ${showTok(toks[c])}: ${pct(v)}`),
        click: (r) => { q = r; draw(); }, mark: (r, c) => r === q && c === W[q].indexOf(Math.max(...W[q])),
      });
      const cell = Math.min(26, Math.floor(300 / T));
      toks.forEach((t, c) => s("text", { x: 70 + c * cell + cell / 2, y: 64, transform: `rotate(-55 ${70 + c * cell + cell / 2} 64)`, text: showTok(t).slice(0, 9) }, m));
      $("w07-matrix").replaceChildren(m);
      // caption
      const best = W[q].indexOf(Math.max(...W[q]));
      const name = H === "avg" ? `layer ${L}, average of all heads` : `layer ${L}, head ${H}`;
      $("w07-cap").replaceChildren(`${name}: from `, chip(toks[q]), ` the biggest share (${pct(W[q][best])}) goes to `, chip(toks[best], { cls: "hot" }),
        best === 0 && q > 0 ? ". That's the first token: this head is idling on the attention sink." : ".");
    };
    const headSet = seg($("w07-heads"), [...Array(NH).keys()].map((i) => ({ label: "H" + i, value: i })).concat([{ label: "average", value: "avg" }]), H, (v) => { H = v === "avg" ? "avg" : +v; draw(); });
    const sentSel = select("w07-sent", S.map((x) => x.text), (v) => { si = v; q = S[si].tokens.length - 1; draw(); });
    const layerUpd = range("w07-layer", (v) => String(v), (v) => { L = v; draw(); });
    const go = (p) => {
      si = p.s; sentSel.value = si; L = p.l; H = p.h; $("w07-layer").value = L; $("w07-layer-out").textContent = L; headSet(H);
      q = S[si].tokens.length - 1;
      if (p.s === 1) q = S[1].tokens.lastIndexOf(" cat");
      draw();
    };
    const pr = $("w07-presets");
    presets.forEach((p) => pr.append(h("button", { class: "btn", type: "button", text: p.label, onclick: () => go(p) })));
    layerUpd();

    // loss on repeated
    chart($("w07-loss"), {
      n: rep.loss.length, height: 220, series: [{ values: rep.loss, colorAt: (i) => (i >= rep.L - 1 ? "var(--accent)" : "var(--line)") }],
      ymax: 20, yticks: [0, 5, 10, 15, 20], xevery: 10, xlabel: (i) => i + 1, xtitle: "token position (the second copy starts at 51)",
      bands: [[rep.L - 1, rep.loss.length, "var(--accent)"]], tip: (i) => `position ${i + 2}: loss ${rep.loss[i]}`, label: "loss on repeated tokens",
    });

    // head score grids
    const grid = (id, data, name) => {
      const svg = heat(data, {
        cell: 22, cellH: 13, left: 34, top: 18, label: name,
        rowLabel: (r) => (r % 5 === 0 || r === NL - 1 ? "L" + r : null), colLabel: (c) => "H" + c,
        tip: (r, c, v) => `layer ${r}, head ${c}\n${name}: ${v.toFixed(2)}\nclick to open in the explorer`,
        click: (r, c) => { go({ l: r, h: c, s: 1 }); document.getElementById("w07-arcs").scrollIntoView({ block: "center" }); },
        mark: (r, c) => r === 23 && c === 4,
      });
      svg.setAttribute("width", 240);
      $(id).replaceChildren(svg);
    };
    grid("w07-prev", rep.prev, "previous-token score");
    grid("w07-ind", rep.ind, "induction score");

    // repeated patterns
    const box = $("w07-patterns");
    const kinds = ["previous-token head", "induction head", "induction head", "sink head"];
    rep.small_heads.forEach((hd, i) => {
      const svg = heat(hd.att.map((r, a) => r.map((v, b) => (b > a ? null : Math.sqrt(v)))), {
        cell: 9, left: 6, top: 6, label: kinds[i],
        tip: (r, c, v) => (v == null ? "masked" : `query pos ${r} → key pos ${c}: ${pct(v * v)}`),
      });
      svg.setAttribute("width", 300);
      box.append(h("div", { style: "display:grid;gap:.4rem;min-width:0" }, h("p", { style: "font:600 .85rem var(--font-mono)", text: `L${hd.layer} H${hd.head} · ${kinds[i]}` }), svg));
    });
  }

  // ------------------------------------------------------------ 08 kv cache
  function l08() {
    const tk = makeTokenizer(window.TOK);
    const prompt = M.l07.sentences[0].tokens;
    const gen = tk.encode(M.l01.greedy[0].completion).ids.slice(0, 7).map((i) => tk.decodeOne(i));
    let step = 0;
    const draw = () => {
      const nTok = step === 0 ? 0 : prompt.length + step - 1;
      const all = prompt.concat(gen);
      const rows = prompt.length + gen.length, cw = 13, ch = 19, left = 92, top = 24;
      const W = left + NL * cw + 10, Hh = top + rows * ch + 8;
      const svg = svgRoot(W, Hh, "KV cache");
      s("text", { x: left, y: 14, text: "layer 0 → 29 (keys + values for each layer)" }, svg);
      for (let r = 0; r < rows; r++) {
        const filled = r < nTok;
        const fresh = step === 1 ? r < prompt.length : r === nTok - 1;
        s("text", { x: left - 8, y: top + r * ch + 13, "text-anchor": "end", class: filled ? "ink" : "", text: showTok(all[r]), style: `opacity:${filled ? 1 : 0.35};font-weight:${fresh && filled ? 700 : 400}` }, svg);
        for (let c = 0; c < NL; c++) s("rect", { x: left + c * cw, y: top + r * ch, width: cw - 2, height: ch - 3, rx: 2,
          fill: !filled ? "var(--surface-2)" : fresh ? "var(--hot)" : "var(--accent)", "fill-opacity": !filled ? 1 : fresh ? 0.9 : 0.45 }, svg);
      }
      $("w08-anim").replaceChildren(svg);
      let phase, cap;
      const workCache = step === 0 ? 0 : prompt.length + (step - 1);
      let workNo = 0;
      for (let k = 1; k <= step; k++) workNo += k === 1 ? prompt.length : prompt.length + k - 1;
      if (step === 0) { phase = "empty cache"; cap = "Press next step. The first step is prefill: the whole prompt goes through the model in one parallel pass."; }
      else if (step === 1) { phase = "PREFILL"; cap = `All ${prompt.length} prompt tokens were processed in one pass, and their keys and values were stored for all 30 layers (amber). The model's output for the last position gives the first new token: ${showTok(gen[0])}.`; }
      else { phase = "DECODE"; cap = `Only the new token ${showTok(all[nTok - 1])} went through the model (amber row). Its attention read the ${nTok - 1} cached rows (teal) instead of recomputing them.`; }
      $("w08-phase").textContent = phase;
      $("w08-phase").style.color = step === 1 ? "var(--hot)" : "var(--accent)";
      $("w08-cap").replaceChildren(cap, h("br"), h("span", { style: "font-family:var(--font-mono)" }, `Token positions processed so far: with cache ${workCache}, without cache ${workNo}.`));
      $("w08-step").disabled = step > gen.length;
    };
    $("w08-step").addEventListener("click", () => { step++; draw(); });
    $("w08-reset").addEventListener("click", () => { step = 0; draw(); });
    draw();

    const t = M.l08.timing;
    chart($("w08-speed"), {
      n: t.length, height: 230, series: [{ name: "without cache", values: t.map((x) => x.no_cache), color: "var(--line)" }, { name: "with KV cache", values: t.map((x) => x.cache), color: "var(--accent)" }],
      yticks: [0, 50, 100, 150], ymax: 160, xlabel: (i) => `${t[i].prompt_tokens}-token prompt`, legend: true,
      tip: (i, se) => `${t[i].prompt_tokens}-token prompt\nwithout cache ${t[i].no_cache} tok/s\nwith cache ${t[i].cache} tok/s`, label: "tokens per second",
    });

    // calculator
    const models = [
      ["SmolLM2-135M", 30, 9, 3, 64], ["Llama-3.2-1B", 16, 32, 8, 64], ["Llama-3.1-8B", 32, 32, 8, 128],
      ["Qwen2.5-7B", 28, 28, 4, 128], ["Qwen2.5-32B", 64, 40, 8, 128], ["Llama-3.1-70B", 80, 64, 8, 128],
    ];
    const ctxs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    let mi = 2, ci = 5, users = 1;
    const gb = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2)) + " GB";
    const calc = () => {
      const [name, layers, qh, kvh, hd] = models[mi];
      const bytes = +$("w08-bytes").value;
      const tokens = ctxs[ci] * users;
      const g = 2 * layers * kvh * hd * tokens * bytes / 1e9;
      const f = 2 * layers * qh * hd * tokens * bytes / 1e9;
      const per = 2 * layers * kvh * hd * bytes / 1024;
      $("w08-calc").replaceChildren(
        h("div", { class: "stats" },
          h("div", { class: "stat" }, h("b", { text: gb(g), style: "color:var(--accent)" }), h("span", { text: `KV cache for ${name}` })),
          h("div", { class: "stat" }, h("b", { text: gb(f) }), h("span", { text: `same model without grouped-query attention (${qh} KV heads instead of ${kvh})` })),
          h("div", { class: "stat" }, h("b", { text: per.toFixed(per < 10 ? 1 : 0) + " KB" }), h("span", { text: "per token of context" }))),
        h("p", { class: "caption", text: `${layers} layers × ${kvh} KV heads × ${hd} dims × 2 (K and V) × ${bytes} bytes × ${tokens.toLocaleString()} tokens.` }));
    };
    seg($("w08-presets"), models.map((m, i) => ({ label: m[0], value: i })), mi, (v) => { mi = +v; calc(); });
    range("w08-ctx", (v) => (ctxs[v] / 1024) + "k", (v) => { ci = v; calc(); });
    $("w08-ctx").value = ci; $("w08-ctx-out").textContent = ctxs[ci] / 1024 + "k";
    range("w08-users", (v) => String(v), (v) => { users = v; calc(); })();
    $("w08-bytes").addEventListener("change", calc);
    calc();
  }

  // ------------------------------------------------------------ 09 ablation
  function l09() {
    const A = M.l09;
    const caps = {
      text: "Increase in loss on an ordinary paragraph (about the water cycle) when that one layer is skipped. Layer 0 is catastrophic (+12). Layers 1 and 11 are severe. Most other layers cost less than 0.5.",
      copy: "Increase in loss on the second copy of the random sequence (lesson 07). The copying circuit needs layers 0–2, 8, 11, 16 (strongest previous-token head), 18 (strongest induction head) and 29.",
      paris: `Probability of ' Paris' for our sentence with that layer removed (normal: ${pct(A.base.paris)}). Removing layer 23 leaves 3%.`,
    };
    const drawL = (m) => {
      const vals = A.layers.map((x) => x[m]);
      chart($("w09-layers"), {
        n: NL, height: 230, series: [{ values: vals, colorAt: (i) => (m === "paris" ? (vals[i] < 0.05 ? "var(--bad)" : "var(--accent)") : i === 23 ? "var(--hot)" : "var(--accent)") }],
        ymax: m === "paris" ? 1 : 15, yticks: m === "paris" ? [0, 0.25, 0.5, 0.75, 1] : [0, 5, 10, 15], yfmt: m === "paris" ? (t) => t * 100 + "%" : null,
        xevery: 1, xtitle: "layer removed", tip: (i) => `without layer ${i}\nΔ text loss ${A.layers[i].text}\nΔ copy loss ${A.layers[i].copy}\np(' Paris') ${pct(A.layers[i].paris)}`, label: "layer ablation",
      });
      $("w09-layers-cap").textContent = caps[m];
    };
    seg($("w09-metric"), [{ label: "Δ loss on normal text", value: "text" }, { label: "Δ loss on copying", value: "copy" }, { label: "p(' Paris')", value: "paris" }], "text", drawL);
    drawL("text");

    const top = new Set(A.top_ind.slice(0, 5).map(([l, hh]) => l + "," + hh));
    const drawH = (m) => {
      const vals = A.single.map((r) => r.map(([c, p]) => (m === "copy" ? c : A.base.paris - p)));
      const mx = Math.max(...vals.flat());
      const svg = heat(vals.map((r) => r.map((v) => Math.max(0, v) / mx)), {
        cell: 30, cellH: 13, left: 36, top: 18, label: "single-head ablation",
        rowLabel: (r) => (r % 5 === 0 || r === NL - 1 || r === 23 ? "L" + r : null), colLabel: (c) => "H" + c,
        tip: (r, c) => { const [cl, p] = A.single[r][c]; return `silence L${r} H${c}\nΔ copy loss ${cl}\np(' Paris') ${pct(A.base.paris)} → ${pct(p)}`; },
        mark: (r, c) => (m === "copy" ? top.has(r + "," + c) : r === 23 && c === 4),
      });
      svg.setAttribute("width", 330);
      $("w09-heads").replaceChildren(svg);
      $("w09-heads-cap").textContent = m === "copy"
        ? "Brighter = more damage to copying. Outlined: the top 5 induction heads from lesson 07. Individually they barely matter; the biggest single-head effects come from L2 H0 and the previous-token head L16 H0, which every induction head depends on."
        : `Brighter = bigger drop in p(' Paris'). Outlined: L23 H4, the head that looks at "iffel". Alone it lowers ' Paris' from ${pct(A.base.paris)} to ${pct(A.single[23][4][1])}, the third-largest single-head effect. Removing all of layer 23 gives 3%, so several heads share the job.`;
    };
    seg($("w09-head-metric"), [{ label: "copying damage", value: "copy" }, { label: "p(' Paris') drop", value: "paris" }], "copy", drawH);
    drawH("copy");

    chart($("w09-table"), {
      n: A.table.length, height: 220, series: [{ name: "top induction heads silenced", values: A.table.map((x) => x.induction), color: "var(--hot)" }, { name: "random heads silenced (avg of 5)", values: A.table.map((x) => x.random), color: "var(--line)" }],
      ymax: 12, yticks: [0, 4, 8, 12], xlabel: (i) => `${A.table[i].k} head${A.table[i].k > 1 ? "s" : ""}`, legend: true,
      tip: (i) => `${A.table[i].k} heads off\ninduction: loss ${A.table[i].induction}\nrandom: loss ${A.table[i].random}`, label: "induction vs random",
    });

    const z = $("w09-zorb");
    const tbl = h("table", { class: "data" }, h("tr", {}, h("th", { text: "induction heads off" }), h("th", { text: "continuation after \"…Every morning, Mrs.\"" })));
    A.zorb.forEach((r) => tbl.append(h("tr", {}, h("td", { text: r.k }), h("td", {}, h("span", { class: "tok" + (r.k === 0 ? " hot" : "") }, UI.tokNodes(r.text))))));
    z.replaceChildren(h("p", { class: "out-box", text: A.zprompt }), h("div", { class: "scroll-x" }, tbl));
  }

  // ------------------------------------------------------------ 10 rebuild
  function l10() {
    const code = [
      ["def my_forward(ids):", null],
      ["    x = E[ids]                               # (10, 576)", 0],
      ["    for i in range(30):", null],
      ["        h = rms_norm(x, w_attn_norm[i])", 1],
      ["        q = h @ Wq[i].T   # (10, 576) = 9 heads × 64", 2],
      ["        k = h @ Wk[i].T   # (10, 192) = 3 heads × 64", 2],
      ["        v = h @ Wv[i].T", 2],
      ["        q, k = rope(q, pos), rope(k, pos)", 3],
      ["        k, v = repeat(k, 3), repeat(v, 3)", 2],
      ["        scores = q @ k.T / sqrt(64)       # (9, 10, 10)", 4],
      ["        scores = mask_future(scores)", 4],
      ["        a = softmax(scores) @ v", 4],
      ["        x = x + concat_heads(a) @ Wo[i].T", 5],
      ["        h = rms_norm(x, w_mlp_norm[i])", 6],
      ["        x = x + (silu(h @ Wg[i].T) * (h @ Wu[i].T)) @ Wd[i].T", 6],
      ["    x = rms_norm(x, w_final_norm)", 7],
      ["    return x @ E.T                           # (10, 49152)", 7],
    ];
    const steps = [
      ["Look up", "Each of our 10 token IDs picks one row of the embedding table E. The result is a 10 × 576 grid: one vector per token. This is the start of the residual stream.", "l04"],
      ["Normalise", "RMSNorm makes a rescaled copy of the stream for attention to read. The stream itself is untouched.", "l05"],
      ["Queries, keys, values", "Three matrix multiplications turn each token's vector into a query (9 heads × 64 numbers), a key and a value (3 heads × 64 each). Each key/value head is then shared by 3 query heads: grouped-query attention.", "l07"],
      ["Add positions (RoPE)", "Queries and keys are rotated by an angle that depends on their position, so that attention scores can depend on distance. Values are not rotated.", "l07-rope"],
      ["Attention", "Every query is compared with every key (dot product, divided by √64 to keep the numbers tame). Future positions are set to −∞ so softmax gives them zero weight. Each row of weights then averages the values. This is where head L23 H4 at \" of\" reads from \"iffel\".", "l07"],
      ["Write back", "The 9 heads' outputs are placed side by side (9 × 64 = 576) and mixed by Wo. The result is added to the stream, not substituted for it.", "l05"],
      ["MLP", "Another norm, then the SwiGLU MLP: expand to 1,536 twice, gate one copy with the other, shrink back to 576, and add to the stream. Then the loop goes round for the next layer, 30 times in all.", "l05"],
      ["Score", "A final RMSNorm, then the stream is compared with every row of the same embedding table E (tied weights). That gives 49,152 logits per position. Softmax of the last row says ' Paris' at 88%.", "l02"],
    ];
    let cur = 0;
    const draw = () => {
      const pre = $("w10-code");
      pre.replaceChildren(...code.map(([line, st]) => {
        const span = h("span", { class: st === cur ? "hl" : "" });
        const [c0, ...cm] = line.split("#");
        span.append(c0);
        if (cm.length) span.append(h("span", { class: "c", text: "#" + cm.join("#") }));
        return h("span", {}, span, "\n");
      }));
      const [title, text, lesson] = steps[cur];
      $("w10-explain").replaceChildren(
        h("p", { class: "eyebrow", text: `step ${cur + 1} of ${steps.length}` }),
        h("h3", { text: title, style: "font-size:1.35rem" }), h("p", { text }),
        h("p", {}, h("a", { href: "#" + lesson, text: `Back to lesson ${lesson.slice(1, 3)} →` })));
    };
    seg($("w10-steps"), steps.map((st, i) => ({ label: `${i + 1}. ${st[0]}`, value: i })), 0, (v) => { cur = +v; draw(); });
    draw();

    // RoPE
    const inv = M.l10.inv_freq;
    let qp = 9, kp = 2, f = 2;
    const q0 = 0.35, k0 = 1.25; // base angles (radians) of the unrotated query/key pair
    const drawR = () => {
      const W = 300, Hh = 300, cx = 150, cy = 150, R = 110;
      const svg = svgRoot(W, Hh, "rotary embedding");
      s("circle", { cx, cy, r: R, fill: "none", stroke: "var(--line)", "stroke-width": 1.5 }, svg);
      s("line", { x1: cx - R - 10, x2: cx + R + 10, y1: cy, y2: cy, class: "axis" }, svg);
      s("line", { y1: cy - R - 10, y2: cy + R + 10, x1: cx, x2: cx, class: "axis" }, svg);
      const aq = q0 + qp * inv[f], ak = k0 + kp * inv[f];
      const vec = (a, col, lab) => {
        const x = cx + R * Math.cos(a), y = cy - R * Math.sin(a);
        s("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: col, "stroke-width": 3.5, "stroke-linecap": "round" }, svg);
        s("circle", { cx: x, cy: y, r: 5, fill: col }, svg);
        s("text", { x: cx + (R + 18) * Math.cos(a), y: cy - (R + 18) * Math.sin(a) + 4, "text-anchor": "middle", text: lab, style: `fill:${col};font-weight:700;font-size:12px` }, svg);
      };
      vec(ak, "var(--accent)", "key");
      vec(aq, "var(--hot)", "query");
      $("w10-rope").replaceChildren(svg);
      const deg = (r) => ((r * 180 / Math.PI) % 360 + 360) % 360;
      const dot = Math.cos(aq - ak);
      const speed = inv[f] * 180 / Math.PI;
      $("w10-rope-text").replaceChildren(
        h("p", {}, `Pair ${f} turns `, h("b", { text: speed >= 0.1 ? speed.toFixed(1) + "° per position" : speed.toExponential(1) + "° per position" }),
          f < 4 ? " (a fast hand)." : f > 24 ? " (an extremely slow hand: it barely moves over a whole document)." : "."),
        h("p", {}, `The query at position ${qp} has turned ${deg(qp * inv[f]).toFixed(0)}°, the key at position ${kp} has turned ${deg(kp * inv[f]).toFixed(0)}°.`),
        h("p", {}, "Their match score (the dot product) is ", h("b", { text: dot.toFixed(3) }), ` and depends only on the distance ${qp - kp}. Drag both sliders by the same amount: the picture rotates, the score stays the same.`),
      );
    };
    range("w10-q", String, (v) => { qp = v; drawR(); })();
    range("w10-k", String, (v) => { kp = v; drawR(); })();
    range("w10-f", String, (v) => { f = v; drawR(); })();
  }

  l06(); l07(); l08(); l09(); l10();
  UI.initTheme();
  UI.initSpy();
})();
