// Widgets for lessons 11-13 (quantization, GGUF, LoRA).
// All measured numbers come from window.MD2 (data/more_data.js, written by site/build_data_more.py).
(function () {
  const { $, h, s, bars, seg, bindTip, chart, pct, svgRoot, range } = UI;
  const M = window.MD2;
  if (!M) return;
  const fmtInt = (n) => Math.round(n).toLocaleString();
  const gb = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toFixed(3)) + " GB";
  const bigNum = (x) => (x >= 1e6 ? (x / 1e6).toFixed(1) + " million" : x >= 1000 ? fmtInt(x) : x.toFixed(2));
  const ppl = (x) => (x >= 1000 ? bigNum(x) : x.toFixed(2));

  function table(el, head, rows, opts = {}) {
    const thead = h("tr", {}, head.map((c) => h("th", { text: c })));
    const body = rows.map((r, i) => {
      const tr = h("tr", {}, r.map((c) => h("td", {}, c)));
      if (opts.click) {
        tr.style.cursor = "pointer"; tr.tabIndex = 0;
        tr.addEventListener("click", () => opts.click(i));
        tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); opts.click(i); } });
      }
      if (opts.rowStyle) tr.setAttribute("style", opts.rowStyle(i) || "");
      return tr;
    });
    el.replaceChildren(h("thead", {}, thead), h("tbody", {}, body));
  }

  // fp16 (2 bytes, little-endian) -> number
  function fp16(lo, hi) {
    const v = lo | (hi << 8), sign = v >> 15 ? -1 : 1, e = (v >> 10) & 31, f = v & 1023;
    if (e === 0) return sign * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : sign * Infinity;
    return sign * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  // ------------------------------------------------------------ 11 quantization
  function l11() {
    const Q = M.l11;
    // memory bill
    const models = [["SmolLM2-135M", Q.params], ["Llama-3.1-8B", 8.03e9], ["Qwen2.5-32B", 32.8e9], ["Llama-3.1-70B", 70.6e9]];
    const precs = [["fp32", 32], ["bf16", 16], ["8-bit", 8.5], ["4-bit", 4.5]];
    let mi = 3;
    const mem = () => {
      const [name, n] = models[mi];
      const vals = precs.map(([, b]) => n * b / 8 / 1e9);
      bars($("w11-mem"), precs.map(([lab], i) => ({ labelNode: h("span", { text: lab }), value: vals[i], text: gb(vals[i]), cls: i === 3 ? "hot" : "" })), vals[0]);
      const fits = (g) => [16, 24, 32, 48, 64, 128, 192].find((r) => r >= g * 1.15);
      $("w11-mem-cap").textContent = `8-bit and 4-bit include the per-group scales (8.5 and 4.5 bits per weight). ${name}: ${gb(vals[1])} in bf16, ${gb(vals[3])} in 4-bit. ` +
        (fits(vals[3]) ? `The 4-bit version fits on a ${fits(vals[3])} GB machine (with room for the KV cache from lesson 08)` : "Still too big for most single machines") +
        (fits(vals[1]) ? `; bf16 needs ${fits(vals[1])} GB.` : "; bf16 needs more memory than any laptop has.");
    };
    seg($("w11-models"), models.map((m, i) => ({ label: m[0], value: i })), mi, (v) => { mi = +v; mem(); });
    mem();

    // rounding 32 weights
    const groups = { outlier: Q.outlier_group, calm: Q.calm_group };
    let gk = "outlier", bits = 4;
    $("w11-src").textContent = Q.tensor;
    const draw = () => {
      const w = groups[gk];
      const qmax = Math.pow(2, bits - 1) - 1;
      const amax = Math.max(...w.map(Math.abs));
      const scale = amax / qmax;
      const q = w.map((x) => Math.max(-qmax - 1, Math.min(qmax, Math.round(x / scale))));
      const wq = q.map((k) => k * scale);
      const W = 760, H = 150, L = 20, R = 20, yA = 42, yB = 112;
      const lim = amax * 1.06;
      const X = (v) => L + (W - L - R) * (v + lim) / (2 * lim);
      const svg = svgRoot(W, H, "weights before and after rounding");
      for (let k = -qmax - 1; k <= qmax; k++) {
        const x = X(k * scale);
        if (x < L - 1 || x > W - R + 1) continue;
        s("line", { x1: x, x2: x, y1: yA - 14, y2: yB + 10, stroke: k === 0 ? "var(--muted)" : "var(--line)", "stroke-width": bits >= 7 ? 0.5 : 1 }, svg);
      }
      s("text", { x: L, y: 14, text: "original weights" }, svg);
      s("text", { x: L, y: H - 6, text: `after rounding to ${bits} bits (${Math.pow(2, bits)} levels, step ${scale.toPrecision(3)})` }, svg);
      w.forEach((x, i) => {
        const hot = Math.abs(x) === amax;
        s("line", { x1: X(x), y1: yA, x2: X(wq[i]), y2: yB, stroke: hot ? "var(--hot)" : "var(--accent)", "stroke-opacity": 0.45 }, svg);
        const a = s("circle", { cx: X(x), cy: yA, r: 4.5, fill: hot ? "var(--hot)" : "var(--accent)" }, svg);
        const b = s("circle", { cx: X(wq[i]), cy: yB, r: 4.5, fill: hot ? "var(--hot)" : "var(--accent)" }, svg);
        const t = () => `weight ${i}: ${x.toFixed(5)}\nstored integer: ${q[i]}\nback: ${q[i]} × ${scale.toPrecision(4)} = ${wq[i].toFixed(5)}`;
        bindTip(a, t); bindTip(b, t);
      });
      $("w11-line").replaceChildren(svg);
      const err = Math.sqrt(w.reduce((a, x, i) => a + (x - wq[i]) ** 2, 0) / w.reduce((a, x) => a + x * x, 0));
      const zeros = q.filter((k) => k === 0).length;
      const used = new Set(q).size;
      $("w11-stats").replaceChildren(
        h("div", { class: "stat" }, h("b", { text: scale.toPrecision(3) }), h("span", { text: `scale = ${amax.toFixed(3)} / ${qmax}` })),
        h("div", { class: "stat" }, h("b", { text: `${used} of ${Math.pow(2, bits)}` }), h("span", { text: "integer levels actually used" })),
        h("div", { class: "stat" }, h("b", { text: `${zeros} / 32`, style: zeros > 8 ? "color:var(--bad)" : "" }), h("span", { text: "weights rounded to exactly 0" })),
        h("div", { class: "stat" }, h("b", { text: pct(err), style: err > 0.2 ? "color:var(--bad)" : "" }), h("span", { text: "relative error of this group" })));
      $("w11-cap").textContent = gk === "outlier"
        ? `This group holds the matrix's biggest weight (${Q.max_abs}; amber), about ${Math.round(Q.max_abs / Q.median_abs)}× a typical weight. It alone sets the scale, so the other 31 share just a few levels near zero.`
        : "A group without an outlier: the levels are packed closely around the weights that are actually there, so rounding barely moves them.";
    };
    seg($("w11-group"), [{ label: "group with an outlier", value: "outlier" }, { label: "ordinary group", value: "calm" }], gk, (v) => { gk = v; draw(); });
    range("w11-bits", (v) => `${v}`, (v) => { bits = v; draw(); })();
    const me = Q.matrix_err;
    $("w11-matrix-err").textContent = `a relative error of ${pct(me["whole matrix"])} with one scale for the whole matrix, ${pct(me["per row"])} with one per row, and ${pct(me["per 32"])} with one per 32 weights`;

    // whole-model table + chart
    const rows = [[32, Q.base[0], 1, Q.base[1]], ...Q.bits];
    let sel = 4;
    const prompt = "The Eiffel Tower is located in the city of";
    const pick = (i) => {
      sel = i;
      table($("w11-table"), ["bits", "perplexity", "agrees with original"],
        rows.map((r) => [r[0] === 32 ? "32 (original)" : String(r[0]), ppl(r[1]), pct(r[2], 0)]),
        { click: pick, rowStyle: (j) => (j === sel ? "background:var(--accent-soft)" : r_bad(j) ? "color:var(--bad)" : "") });
      $("w11-text").replaceChildren(h("span", { class: "muted", text: `${rows[i][0]} bits · ` }), prompt, h("span", { class: "gen" + (rows[i][3].startsWith(" Paris") ? " hot" : ""), text: rows[i][3] }));
      drawChart();
    };
    const r_bad = (j) => rows[j][1] > 100;
    const drawChart = () => chart($("w11-chart"), {
      n: rows.length, width: 420, height: 240, left: 64, log: true, ymin: 10, ymax: 1e7, yticks: [10, 100, 1e3, 1e4, 1e5, 1e6, 1e7],
      yfmt: (t) => (t >= 1e6 ? t / 1e6 + "M" : t >= 1e3 ? t / 1e3 + "k" : t),
      series: [{ type: "line", values: rows.map((r) => r[1]), color: "var(--accent)", dot: 4, dotColor: (i) => (i === sel ? "var(--hot)" : "var(--accent)") }],
      xlabel: (i) => String(rows[i][0]), xtitle: "bits per weight (group 32)",
      tip: (i) => `${rows[i][0]} bits: perplexity ${ppl(rows[i][1])}\nsame next token as original: ${pct(rows[i][2], 0)}`, label: "perplexity by bits",
    });
    pick(sel);

    table($("w11-groups"), ["one scale per", "perplexity", "agree"], Q.groups.map((g) => [g[0], ppl(g[1]), pct(g[2], 0)]),
      { rowStyle: (j) => (Q.groups[j][1] > 1000 ? "color:var(--bad)" : "") });
    const worst = Math.max(...Q.fragile.map((f) => f[2]));
    table($("w11-fragile"), ["at 3 bits", "share", "perplexity"], Q.fragile.map((f) => [f[0], pct(f[1], 0), ppl(f[2])]),
      { rowStyle: (j) => (Q.fragile[j][2] > worst * 0.9 ? "color:var(--hot);font-weight:600" : "") });
  }

  // ------------------------------------------------------------ 12 GGUF
  function l12() {
    const G = M.l12;
    $("w12-size").textContent = `${(G.size / 1e6).toFixed(1)} MB · ${G.n_kv} metadata entries · ${G.n_tensors} tensors`;
    const secs = [
      ["header", "header", "var(--hot)"], ["metadata", "metadata", "var(--accent)"], ["tensor_table", "tensor table", "var(--muted)"],
      ["padding", "padding", "var(--line)"], ["weights", "weights", "var(--accent)"],
    ];
    let cur = "metadata";
    const detail = {
      header: () => {
        const b = G.head24.split(" ");
        return [h("p", {}, "The first 24 bytes. All numbers are little-endian (lowest byte first)."),
          (() => { const t = h("table", { class: "data" }); table(t, ["bytes", "raw", "meaning"], [
            ["0–3", b.slice(0, 4).join(" "), '"GGUF" in ASCII: the magic word'],
            ["4–7", b.slice(4, 8).join(" "), "format version 3"],
            ["8–15", b.slice(8, 16).join(" "), `${G.n_tensors} tensors`],
            ["16–23", b.slice(16, 24).join(" "), `${G.n_kv} metadata entries`]]); return t; })()];
      },
      metadata: () => {
        const t = h("table", { class: "data" });
        table(t, ["key", "value"], G.meta.map(([k, v]) => [k, v]),
          { rowStyle: (i) => (G.meta[i][0].startsWith("tokenizer") ? "color:var(--accent)" : "") });
        return [h("p", {}, `${fmtInt(G.sections.metadata)} bytes. Almost all of it is the tokenizer (teal): 49,152 token strings and 48,900 merge rules. The rest is config.json in key/value form.`),
          h("div", { style: "max-height:22rem;overflow:auto" }, t)];
      },
      tensor_table: () => {
        const t = h("table", { class: "data" });
        table(t, ["name", "shape", "type", "offset"], G.layer0.map(([n, sh, ty, off]) => [n, `[${sh.join(", ")}]`, ty, "+" + fmtInt(off)]));
        return [h("p", {}, `${fmtInt(G.sections.tensor_table)} bytes of table of contents: name, shape, quantization type and where the bytes start. Layer 0 shown; layers 1–29 repeat it.`),
          h("div", { class: "scroll-x" }, t)];
      },
      padding: () => [h("p", {}, `${G.sections.padding} zero bytes, so the weights start at a multiple of 32 bytes. Aligned data can be memory-mapped and read by fast vector instructions directly.`)],
      weights: () => [h("p", {}, `${(G.sections.weights / 1e6).toFixed(1)} MB of raw blocks, one tensor after another: ` +
        Object.entries(G.types).map(([k, v]) => `${v} tensors in ${k}`).join(", ") +
        ". The 1-D norm weights stay F32: they're tiny and fragile. Nothing here is compressed; the runtime reads these bytes straight from the file.")],
    };
    const W = 760, Hh = 112, L = 4, R = 4;
    const draw = () => {
      const svg = svgRoot(W, Hh, "GGUF file layout");
      const rowBar = (y, items, total, label) => {
        s("text", { x: L, y: y - 6, text: label }, svg);
        let x = L;
        items.forEach(([key, name, col]) => {
          const bytes = G.sections[key];
          const w = Math.max(key === "header" || key === "padding" ? 5 : 2, (W - L - R) * bytes / total);
          const g = s("g", { style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": name }, svg);
          s("rect", { x, y, width: w - 1, height: 26, rx: 3, fill: col, "fill-opacity": cur === key ? 0.95 : 0.4, stroke: cur === key ? "var(--ink)" : "none" }, g);
          if (w > 70) s("text", { x: x + 6, y: y + 17, class: "ink", text: name }, g);
          g.addEventListener("click", () => { cur = key; draw(); });
          g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); cur = key; draw(); } });
          bindTip(g, () => `${name}: ${fmtInt(bytes)} bytes`);
          x += w;
        });
      };
      rowBar(18, secs, G.size, `whole file, ${(G.size / 1e6).toFixed(1)} MB: weights are ${pct(G.sections.weights / G.size)}`);
      const front = G.sections.header + G.sections.metadata + G.sections.tensor_table + G.sections.padding;
      rowBar(78, secs.slice(0, 4), front, `zoom: the first ${(front / 1e6).toFixed(2)} MB, everything before the weights`);
      $("w12-map").replaceChildren(h("div", { class: "seg", style: "margin-bottom:.5rem" },
        secs.map(([key, name]) => h("button", { class: "btn" + (cur === key ? " on" : ""), type: "button", text: name, onclick: () => { cur = key; draw(); } }))), svg);
      $("w12-detail").replaceChildren(h("div", { class: "prose", style: "gap:.6rem" }, detail[cur]()));
    };
    draw();

    // Q8_0 block
    const bytes = G.paris_block.match(/../g).map((x) => parseInt(x, 16));
    const scale = fp16(bytes[0], bytes[1]);
    const ints = bytes.slice(2).map((b) => (b > 127 ? b - 256 : b));
    let selB = 2;
    const drawBlock = () => {
      const cw = 44, ch = 40, gap = 3, L2 = 70;
      const svg = svgRoot(L2 + 16 * (cw + gap), 3 * (ch + 12) + 10, "Q8_0 block bytes");
      const cell = (i, x, y, sub) => {
        const isScale = i < 2;
        const g = s("g", { style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": `byte ${i}` }, svg);
        s("rect", { x, y, width: cw, height: ch, rx: 4, fill: isScale ? "var(--hot)" : "var(--accent)", "fill-opacity": selB === i || (isScale && selB < 2) ? 0.9 : 0.22 }, g);
        s("text", { x: x + cw / 2, y: y + 17, "text-anchor": "middle", class: "ink", text: bytes[i].toString(16).padStart(2, "0") }, g);
        s("text", { x: x + cw / 2, y: y + 32, "text-anchor": "middle", text: sub }, g);
        g.addEventListener("click", () => { selB = i; drawBlock(); });
        g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selB = i; drawBlock(); } });
      };
      s("text", { x: 0, y: 25, text: "scale" }, svg);
      cell(0, L2, 4, "lo"); cell(1, L2 + cw + gap, 4, "hi");
      s("text", { x: 0, y: 25 + ch + 12, text: "int8 0–15" }, svg);
      s("text", { x: 0, y: 25 + 2 * (ch + 12), text: "int8 16–31" }, svg);
      for (let j = 0; j < 32; j++) cell(j + 2, L2 + (j % 16) * (cw + gap), 4 + (1 + Math.floor(j / 16)) * (ch + 12), String(ints[j]));
      $("w12-block").replaceChildren(svg);
      let msg;
      if (selB < 2) {
        msg = [h("p", {}, "Bytes ", h("code", { text: "0x" + bytes[0].toString(16).padStart(2, "0") }), " and ", h("code", { text: "0x" + bytes[1].toString(16).padStart(2, "0") }),
          ` together are a 16-bit float (low byte first): `, h("b", { text: `scale = ${scale.toPrecision(5)}` }),
          `. It's this block's biggest |weight| divided by 127.`)];
      } else {
        const j = selB - 2, dec = ints[j] * scale, orig = G.paris_orig[j];
        msg = [h("p", {}, `Weight ${j}: byte `, h("code", { text: "0x" + bytes[selB].toString(16).padStart(2, "0") }), ` read as a signed 8-bit number is `, h("b", { text: String(ints[j]) }),
          `. Times the scale: ${ints[j]} × ${scale.toPrecision(5)} = `, h("b", { text: dec.toFixed(5) }),
          `. The original model has ${orig.toFixed(5)}, an error of ${Math.abs(dec - orig).toFixed(5)}.`)];
      }
      $("w12-byte").replaceChildren(h("div", { class: "prose" }, msg));
    };
    drawBlock();
    $("w12-stats").replaceChildren(
      h("div", { class: "stat" }, h("b", { text: pct(G.emb_rel_err, 2) }), h("span", { text: "relative error of the whole embedding table" })),
      h("div", { class: "stat" }, h("b", { text: pct(G.same_ints, 2) }), h("span", { text: "of the integers are identical to what lesson 11's absmax code makes" })));

    // RoPE pairing
    const rs = svgRoot(380, 170, "RoPE pairs");
    const rowP = (y, label, pairs) => {
      s("text", { x: 0, y: y - 14, text: label }, rs);
      for (let i = 0; i < 8; i++) {
        s("rect", { x: 10 + i * 44, y, width: 36, height: 26, rx: 4, fill: "var(--surface-2)", stroke: "var(--line)" }, rs);
        s("text", { x: 28 + i * 44, y: y + 17, "text-anchor": "middle", class: "ink", text: String(i) }, rs);
      }
      pairs.forEach(([a, b], k) => {
        const x1 = 28 + a * 44, x2 = 28 + b * 44, yy = y + 26;
        s("path", { d: `M${x1},${yy} Q${(x1 + x2) / 2},${yy + 14 + (x2 - x1) / 6} ${x2},${yy}`, fill: "none", stroke: k % 2 ? "var(--hot)" : "var(--accent)", "stroke-width": 2 }, rs);
      });
    };
    rowP(26, "Hugging Face: i with i + half", [[0, 4], [1, 5], [2, 6], [3, 7]]);
    rowP(106, "llama.cpp: neighbours", [[0, 1], [2, 3], [4, 5], [6, 7]]);
    $("w12-rope").replaceChildren(rs, h("p", { class: "caption", text: "Same rotation, different pairing. The converter reorders q_proj and k_proj rows once so the pairs sit next to each other." }));
    $("w12-q").replaceChildren(
      h("div", { class: "stat" }, h("b", { text: pct(G.q_err_raw, 0), style: "color:var(--bad)" }), h("span", { text: "attn_q vs q_proj, compared directly" })),
      h("div", { class: "stat" }, h("b", { text: pct(G.q_err_fixed, 2), style: "color:var(--accent)" }), h("span", { text: "after undoing the row shuffle" })));
    const t = $("w12-reports");
    table(t, ["weights", "perplexity", "continues “…city of”"], G.reports.map(([k, p, txt]) => [k, p.toFixed(2), h("span", { style: "color:var(--hot)", text: txt.slice(0, 22) + "…" })]));

    // Q4_K_M census
    const kinds = Object.entries(G.census);
    table($("w12-census"), ["tensor kind", "row length", "÷ 256?", "stored as"],
      kinds.map(([k, v]) => [k, String(v.row), v.row % 256 === 0 ? "yes" : "no", Object.entries(v.types).map(([ty, n]) => `${ty} × ${n}`).join(", ")]),
      { rowStyle: (i) => (Object.keys(kinds[i][1].types).includes("Q4_K") ? "color:var(--hot);font-weight:600" : "") });
    const bpw4 = G.size_q4 * 8 / G.params, bpw8 = G.size * 8 / G.params;
    $("w12-census-cap").textContent = `Only ffn_down (amber) actually got Q4_K. K-quants need rows whose length is a multiple of 256; ffn_down rows are 1536 = 6 × 256. Everything else here has rows of 576, so llama.cpp fell back to the block-32 formats Q5_0 and Q8_0. ` +
      `Result: this "4-bit" file averages ${bpw4.toFixed(2)} bits per weight (the Q8_0 file: ${bpw8.toFixed(2)}). On big models, with rows of 4096 or more, you'd see mostly Q4_K with some Q6_K.`;

    const formats = [["bf16", 16], ["Q8_0", 8.5], ["Q6_K", 6.5625], ["Q5_1", 6], ["Q5_0", 5.5], ["Q5_K", 5.5], ["Q4_1", 5], ["Q4_0", 4.5], ["Q4_K", 4.5], ["Q3_K", 3.4375], ["Q2_K", 2.625]];
    bars($("w12-formats"), formats.map(([n, b]) => ({ labelNode: h("span", { text: n }), value: b, text: String(+b.toFixed(2)), cls: n === "Q4_K" || n === "Q4_0" ? "hot" : n === "bf16" ? "cut" : "" })), 16);
  }

  // ------------------------------------------------------------ 13 LoRA
  function l13() {
    const Lr = M.l13;
    const ranks = [1, 2, 4, 8, 16, 32, 64];
    const perLayer = 576 * 2 + (576 + 192) * 2 + (576 + 576) + 3 * (576 + 1536);   // sum of (in + out) over the 7 matrices
    const drawShapes = (ri) => {
      const r = ranks[ri], sc = 190 / 576, side = 190;
      const W = 640, Hh = side + 40;
      const svg = svgRoot(W, Hh, "LoRA shapes");
      const box = (x, y, w, hh, fill, label, sub) => {
        s("rect", { x, y, width: Math.max(w, 1.5), height: Math.max(hh, 1.5), fill, "fill-opacity": 0.85, rx: 2 }, svg);
        s("text", { x: x + Math.max(w, 1.5) / 2, y: Hh - 18, "text-anchor": "middle", class: "ink", text: label }, svg);
        s("text", { x: x + Math.max(w, 1.5) / 2, y: Hh - 4, "text-anchor": "middle", text: sub }, svg);
      };
      const top = 6;
      box(10, top, side, side, "var(--line)", "W (frozen)", "576 × 576");
      s("text", { x: side + 30, y: top + side / 2 + 6, class: "ink", "font-size": 22, text: "+" }, svg);
      const bx = side + 60, bw = Math.max(r * sc, 1.5);
      box(bx, top, bw, side, "var(--hot)", "B", `576 × ${r}`);
      s("text", { x: bx + bw + 26, y: top + side / 2 + 6, class: "ink", "font-size": 20, text: "×" }, svg);
      const ax = bx + bw + 56;
      box(ax, top + side / 2 - (r * sc) / 2, side, r * sc, "var(--accent)", "A", `${r} × 576`);
      $("w13-shapes").replaceChildren(svg);
      const one = r * 1152, all = r * perLayer * 30;
      $("w13-stats").replaceChildren(
        h("div", { class: "stat" }, h("b", { text: fmtInt(one) }), h("span", { text: `numbers to train here, ${pct(one / 331776)} of W's 331,776` })),
        h("div", { class: "stat" }, h("b", { text: fmtInt(all) }), h("span", { text: `trainable in the whole model (all 7 matrices × 30 layers), ${pct(all / M.l11.params)}` })),
        h("div", { class: "stat" }, h("b", { text: (all * 2 / 1e6).toFixed(1) + " MB" }), h("span", { text: "adapter file in fp16" })));
    };
    range("w13-r", (v) => String(ranks[v]), drawShapes)();

    $("w13-data").replaceChildren(...Lr.train.map((t) => h("div", { class: "gen", text: t })), ...Lr.anchors.map((t) => h("div", { class: "muted", text: t })));
    chart($("w13-loss"), {
      n: Lr.losses_plain.length, width: 420, height: 200, ymax: 3, yticks: [0, 1, 2, 3],
      series: [{ type: "line", name: "4 edit sentences", values: Lr.losses_plain, color: "var(--hot)", dot: 0 },
        { type: "line", name: "+ 5 anchors", values: Lr.losses_anchored, color: "var(--accent)", dot: 0 }],
      xevery: 10, xtitle: "training step", legend: true, label: "training loss",
    });

    const stages = Object.keys(Lr.probes);
    const labels = ["before training", "after training", "after training, with anchors"];
    const caps = [
      "The original model: Paris for the tower, Rome for the Colosseum, and so on.",
      "The edit worked, even for the new wording. But the Statue of Liberty and the Sydney Opera House now say Rome too, and “The capital of France is” broke. Side effects in red.",
      "With anchors: the edit holds, France is Paris again, and Sydney (never in the training data) is back in Sydney. The price: the new wording mostly lost the edit.",
    ];
    const minibar = (p, col) => h("span", { style: "display:inline-flex;align-items:center;gap:.4rem" },
      h("span", { style: `display:inline-block;width:3rem;height:.6rem;background:var(--surface-2);border-radius:2px;overflow:hidden` },
        h("span", { style: `display:block;height:100%;width:${Math.round(p * 100)}%;background:${col}` })), pct(p, 0));
    const before = Lr.probes[stages[0]];
    const drawProbes = (k) => {
      const rows = Lr.probes[stages[k]];
      table($("w13-probes"), ["prompt", "P(Rome)", "P(Paris)", "writes"],
        rows.map(([p, why, pr, pp, txt]) => [h("span", {}, p, h("br"), h("small", { class: "muted", text: why })), minibar(pr, "var(--accent)"), minibar(pp, "var(--hot)"), txt]),
        { rowStyle: (i) => {
          if (k === 0 || i === 0 || i === 1) return "";
          const gotRome = rows[i][4].startsWith(" Rome"), wasRome = before[i][4].startsWith(" Rome");
          const brokeFrance = i === 2 && !rows[i][4].startsWith(" Paris");
          return (gotRome && !wasRome) || brokeFrance ? "color:var(--bad)" : "";
        } });
      $("w13-probe-cap").textContent = caps[k];
    };
    seg($("w13-stage"), stages.map((_, i) => ({ label: labels[i], value: i })), 1, (v) => drawProbes(+v));
    drawProbes(1);

    const dial = Lr.dial;
    range("w13-dial", (v) => pct(dial[v][0], 0), (v) => {
      const [f, pr, pp, txt] = dial[v];
      bars($("w13-dial-bars"), [{ labelNode: h("span", { text: "P(' Rome')" }), value: pr }, { labelNode: h("span", { text: "P(' Paris')" }), value: pp, cls: "hot" }], 1);
      $("w13-dial-text").replaceChildren("The Eiffel Tower is located in the city of", h("span", { class: "gen" + (txt.startsWith(" Paris") ? " hot" : ""), text: txt }));
    })();

    const n8b = 32 * 8 * ((4096 + 4096) * 2 + (4096 + 1024) * 2 + 3 * (4096 + 14336));
    $("w13-final").replaceChildren(
      h("div", { class: "stat" }, h("b", { text: Lr.adapter_mb.toFixed(1) + " MB" }), h("span", { text: `the saved adapter, next to a ${Lr.model_mb.toFixed(0)} MB model` })),
      h("div", { class: "stat" }, h("b", { text: "≈" + Math.round(n8b * 2 / 1e6) + " MB" }), h("span", { text: "same recipe (rank 8) for Llama-3.1-8B, next to 16 GB" })),
      h("div", { class: "stat" }, h("b", { text: `${pct(Lr.qlora.before[0], 0)} → ${pct(Lr.qlora.after[0], 0)}` }), h("span", { text: "P(Rome) on a 4-bit base, before and after QLoRA" })));
  }

  l11(); l12(); l13();
})();
