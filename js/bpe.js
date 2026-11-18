// A complete byte-level BPE tokenizer (the GPT-2 family algorithm, as used by SmolLM2)
// in ~70 lines. Data comes from data/tokenizer.js (window.TOK), exported from tokenizer.json.
(function (root) {
  function bytesToUnicode() {
    // GPT-2 trick: map every byte 0..255 to a printable character, so tokens are always strings.
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const m = {};
    bs.forEach((b, i) => (m[b] = String.fromCodePoint(cs[i])));
    return m;
  }

  function makeTokenizer(T) {
    const byteMap = bytesToUnicode();
    const unByte = {};
    Object.entries(byteMap).forEach(([b, c]) => (unByte[c] = +b));
    const tokToId = new Map();
    T.vocab.forEach((t, i) => t != null && tokToId.set(t, i));
    const rank = new Map();
    T.merges.forEach(([a, b], i) => rank.set(a * 65536 + b, i));
    const added = Object.keys(T.added).sort((a, b) => b.length - a.length);
    const enc = new TextEncoder();
    const dec = new TextDecoder("utf-8", { fatal: false });
    // GPT-2 pre-tokenizer regex: contractions, words, numbers, punctuation, whitespace
    const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    const cache = new Map();

    function bpe(word) {
      if (cache.has(word)) return cache.get(word);
      let ids = Array.from(word, (ch) => tokToId.get(ch));
      while (ids.length > 1) {
        let best = Infinity, at = -1;
        for (let i = 0; i < ids.length - 1; i++) {
          const r = rank.get(ids[i] * 65536 + ids[i + 1]);
          if (r !== undefined && r < best) { best = r; at = i; }
        }
        if (at < 0) break;
        const merged = tokToId.get(T.vocab[ids[at]] + T.vocab[ids[at + 1]]);
        ids.splice(at, 2, merged);
      }
      cache.set(word, ids);
      return ids;
    }

    function preSplit(text) {
      // 1) every digit stands alone   2) GPT-2 regex inside each remaining chunk
      const out = [];
      for (const chunk of text.split(/(\p{N})/u)) {
        if (!chunk) continue;
        if (/^\p{N}$/u.test(chunk)) { out.push(chunk); continue; }
        for (const m of chunk.matchAll(pat)) out.push(m[0]);
      }
      return out;
    }

    function encode(text) {
      // special tokens are matched first and never split
      const parts = [];
      let rest = text;
      while (rest.length) {
        let hit = -1, which = null;
        for (const a of added) { const i = rest.indexOf(a); if (i >= 0 && (hit < 0 || i < hit)) { hit = i; which = a; } }
        if (hit < 0) { parts.push([rest, false]); break; }
        if (hit > 0) parts.push([rest.slice(0, hit), false]);
        parts.push([which, true]);
        rest = rest.slice(hit + which.length);
      }
      const ids = [], words = [];
      for (const [p, special] of parts) {
        if (special) { ids.push(T.added[p]); words.push(p); continue; }
        for (const w of preSplit(p)) {
          const mapped = Array.from(enc.encode(w), (b) => byteMap[b]).join("");
          const wi = bpe(mapped);
          ids.push(...wi);
          words.push(w);
        }
      }
      return { ids, words };
    }

    function decodeOne(id) {
      const t = T.vocab[id];
      if (t == null) return "";
      if (T.added[t] === id) return t;
      return dec.decode(new Uint8Array(Array.from(t, (c) => unByte[c])));
    }
    function rawToken(id) { return T.vocab[id]; }
    return { encode, decodeOne, rawToken, vocabSize: T.vocab.length };
  }

  root.makeTokenizer = makeTokenizer;
  if (typeof module !== "undefined") module.exports = { makeTokenizer };
})(typeof window !== "undefined" ? window : globalThis);
