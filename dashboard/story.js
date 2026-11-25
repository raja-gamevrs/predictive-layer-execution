
// ---- The story's later chapters (dataset `story`, built by scripts/build_story_data.py) ----
const ST_SERIES = [
  { key: 'before', label: 'Ours, before', dash: '6 4' },
  { key: 'after', label: 'Ours, after (fused step)' },
  { key: 'draft', label: 'Ours, after + draft' },
  { key: 'llama', label: 'llama.cpp' },
  { key: 'llama_spec', label: 'llama.cpp + its draft', dash: '6 4' },
];
const ST_COLOR = { before: 'var(--cds-chart-muted)', after: dash.colors[0], draft: dash.colors[2], llama: dash.colors[1], llama_spec: dash.colors[3] };
const ST_CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
const stK = (c) => (c / 1024) + 'k';
PLE.stCtx = PLE.stCtx || 2048;
PLE.stModel = PLE.stModel || 'qwen3_4b';

dash.calc('st_tps', {
  inputs: ['story', 'st_check'],
  title: 'Stable decode tokens/s by model, context and series',
  description: "Per model, context and series: before = our exact step before fusion (L1 + L2), after = the fused exact step (F3a), draft = after + the model's own draft (one setting per model: the fastest at 2k among settings with a clean greedy check, else the fastest at 2k), llama = llama.cpp plain decoding on the same weights, llama_spec = llama.cpp's best speculative setting.",
  fn: (story, checks) => {
    const rows = story.filter((x) => x.kind === 'tps' && x.tps != null);
    const clean = (key, s) => { const c = checks.filter((r) => r.key === key && r.setting === s.replace(/c$/, '')); return c.length > 0 && c.every((r) => r.violations === 0); };
    const pickDraft = new Map();
    for (const key of [...new Set(rows.map((r) => r.key))]) {
      const d2 = rows.filter((r) => r.key === key && r.series === 'draft' && r.context === 2048).sort((a, b) => b.tps - a.tps);
      const pick = d2.find((r) => clean(key, r.setting)) || d2[0];
      if (pick) pickDraft.set(key, pick.setting);
    }
    const best = new Map();
    for (const r of rows) {
      if (r.series === 'draft' && r.setting !== pickDraft.get(r.key)) continue;
      const k = `${r.key}|${r.context}|${r.series}`;
      if (!best.has(k) || r.tps > best.get(k).tps) best.set(k, r);
    }
    return [...best.values()].map((r) => ({ key: r.key, model: r.model, quant: r.quant, context: r.context, ctx: stK(r.context), series: r.series,
      tps: Math.round(r.tps), raw: r.tps, setting: r.setting, accept: r.accept, clean: r.series === 'draft' ? clean(r.key, r.setting) : null, file: r.file }));
  },
});

dash.calc('st_land', {
  inputs: ['st_tps'],
  title: 'Where we landed, per model and context',
  description: "Per model and context: our best exact setting (the fused step, or the model's draft setting when that setting's greedy check had 0 violations) against llama.cpp's best (plain or speculative) on the same weights.",
  fn: (t) => {
    const out = [];
    for (const key of [...new Set(t.map((r) => r.key))]) {
      for (const c of ST_CTX) {
        const at = (s) => t.find((r) => r.key === key && r.context === c && r.series === s);
        const b = at('before'), a = at('after'), d = at('draft'), l = at('llama'), ls = at('llama_spec');
        const ours = [a, d && d.clean ? d : null].filter(Boolean).sort((x, y) => y.raw - x.raw)[0];
        const theirs = [l, ls].filter(Boolean).sort((x, y) => y.raw - x.raw)[0];
        const m = a || b || l || t.find((r) => r.key === key);
        out.push({ id: `${key}|${stK(c)}`, key, model: m.model, quant: m.quant, context: c, ctx: stK(c),
          before: b ? b.tps : null, after: a ? a.tps : null, draft: d ? d.tps : null, draft_setting: d ? d.setting : null, draft_clean: d ? d.clean : null,
          llama: l ? l.tps : null, llama_spec: ls ? ls.tps : null,
          ours: ours ? ours.tps : null, ours_how: ours ? (ours === a ? 'fused step' : 'with draft') : null, theirs: theirs ? theirs.tps : null,
          gain_pct: b && ours ? Math.round((ours.raw / b.raw - 1) * 100) : null,
          vs_llama: ours && theirs ? Math.round((ours.raw / theirs.raw) * 100) / 100 : null });
      }
    }
    return out;
  },
});

dash.calc('st_check', {
  inputs: ['story'],
  title: 'Greedy checks of the timed settings',
  description: 'Violations (the engine picked something other than the model\'s clear top token) and near-ties, 128 tokens per prompt, at the first context of each run.',
  fn: (story) => {
    const g = new Map();
    for (const r of story.filter((x) => x.kind === 'check')) {
      const k = `${r.key}|${r.setting}`;
      const o = g.get(k) || { id: k, key: r.key, model: r.model, setting: r.setting, prompts: 0, violations: 0, near_ties: 0, file: r.file };
      o.prompts += 1; o.violations += r.violations; o.near_ties += r.near_ties;
      g.set(k, o);
    }
    return [...g.values()];
  },
});

dash.calc('st_q2', {
  inputs: ['story'],
  title: '2-bit drafts in llama.cpp',
  description: 'llama.cpp, Q8_0 target with a pure Q2_K draft of the same model, 2k context, three prompts: tokens/s and how often the draft was accepted.',
  fn: (story) => story.filter((r) => r.kind === 'q2').map((r) => ({ id: `${r.model}|${r.k}`, model: r.model, k: r.k, tps: Math.round(r.tps),
    accept_pct: r.accept == null ? null : Math.round(r.accept * 100), file: r.file })),
});

dash.calc('st_inkling', {
  inputs: ['story'],
  title: 'Inkling 2 Mini in llama.cpp',
  description: 'Stable decode tokens/s of Inkling 2 Mini in llama.cpp at each context (our engine does not run this model).',
  fn: (story) => story.filter((r) => r.kind === 'inkling').map((r) => ({ id: `${r.config}|${stK(r.context)}`, config: r.config, k: r.k,
    context: r.context, ctx: stK(r.context), tps: Math.round(r.tps), file: r.file })),
});

dash.draw('st-lines', {
  mount: (el) => PLE.setup(el, 'Stable decode tokens per second from 2k to 128k context: ours before, ours after, ours after with a draft, and llama.cpp'),
  update: (el, made) => {
    const d = dash.data('st_tps');
    if (!PLE.ready(el, d, made)) return;
    const { svg, tip } = made;
    const key = el.dataset.model || PLE.stModel;
    const rows = d.data.filter((r) => r.key === key);
    const W = Math.max(280, el.clientWidth), H = 300, top = 16, bottom = 34;
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
    const shown = ST_SERIES.filter((s) => rows.some((r) => r.series === s.key));
    const ymax = d3.max(rows, (r) => r.tps) * 1.08;
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([H - bottom, top]);
    const ax = PLE.leftAxis(svg, y, d3.format('~s'));
    const left = ax.w + 12;
    ax.g.attr('transform', `translate(${left},0)`);
    const narrow = W < 520;
    const right = narrow ? 12 : 176;
    const x = d3.scalePoint().domain(ST_CTX).range([left + 8, W - right - 8]);
    svg.append('g').selectAll('line').data(y.ticks(5)).join('line').attr('class', 'ple-gridline')
      .attr('x1', left).attr('x2', W - right).attr('y1', (t) => y(t)).attr('y2', (t) => y(t));
    svg.append('g').attr('transform', `translate(0,${H - bottom})`).call(d3.axisBottom(x).tickFormat(stK).tickSizeOuter(0));
    const line = d3.line().x((r) => x(r.context)).y((r) => y(r.tps));
    const ends = [];
    for (const s of shown) {
      const pts = rows.filter((r) => r.series === s.key).sort((a, b) => a.context - b.context);
      svg.append('path').datum(pts).attr('d', line).attr('fill', 'none').attr('stroke', ST_COLOR[s.key]).attr('stroke-width', 2)
        .attr('stroke-dasharray', s.dash || null).attr('stroke-linejoin', 'round');
      svg.append('g').selectAll('circle').data(pts).join('circle').attr('cx', (r) => x(r.context)).attr('cy', (r) => y(r.tps)).attr('r', 4)
        .style('fill', ST_COLOR[s.key]).style('stroke', 'var(--color-panel)').style('stroke-width', 2);
      const last = pts[pts.length - 1];
      if (last) ends.push({ y: y(last.tps), x: x(last.context), text: `${s.label.replace('Ours, ', '')} ${last.tps}`, key: s.key });
    }
    if (!narrow) {
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
      svg.append('g').selectAll('text').data(ends).join('text').attr('class', 'ple-lbl').attr('x', W - right + 10).attr('y', (e) => e.y + 4)
        .text((e) => e.text);
    }
    const guide = svg.append('line').attr('class', 'ple-ref').attr('y1', top).attr('y2', H - bottom).attr('hidden', '');
    svg.append('g').selectAll('rect').data(ST_CTX).join('rect')
      .attr('x', (c) => x(c) - x.step() / 2).attr('width', x.step()).attr('y', top).attr('height', H - bottom - top).style('fill', 'transparent')
      .on('pointerenter pointermove', (ev, c) => {
        guide.attr('x1', x(c)).attr('x2', x(c)).attr('hidden', null);
        const at = rows.filter((r) => r.context === c);
        const lines = [`${stK(c)} context, tokens per second`];
        for (const s of shown) {
          const r = at.find((z) => z.series === s.key);
          lines.push(`${s.label}: ${r ? r.tps + (r.accept != null ? ` (draft accepted ${Math.round(r.accept * 100)}%${r.clean === false ? ', not exact: see the check table' : ''})` : '') : 'not run / does not fit'}`);
        }
        PLE.showTip(el, tip, ev, lines);
      })
      .on('pointerleave', () => { guide.attr('hidden', ''); PLE.hideTip(tip); });
  },
});

dash.draw('st-bars', {
  mount: (el) => PLE.setup(el, 'Tokens per second at the chosen context for each model: ours before, ours after, ours after with a draft, and llama.cpp'),
  update: (el, made) => {
    const d = dash.data('st_land');
    if (!PLE.ready(el, d, made)) return;
    const { svg, tip } = made;
    const c = PLE.stCtx;
    const rows = d.data.filter((r) => r.context === c);
    const keys = ST_SERIES.map((s) => s.key);
    const W = Math.max(280, el.clientWidth), H = 290, top = 22, bottom = 44;
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
    const ymax = d3.max(rows.flatMap((r) => keys.map((k) => r[k]).filter((v) => v != null))) * 1.1;
    const y = d3.scaleLinear().domain([0, ymax]).nice().range([H - bottom, top]);
    const ax = PLE.leftAxis(svg, y, d3.format('~s'));
    const left = ax.w + 12;
    ax.g.attr('transform', `translate(${left},0)`);
    svg.append('g').selectAll('line').data(y.ticks(5)).join('line').attr('class', 'ple-gridline')
      .attr('x1', left).attr('x2', W).attr('y1', (t) => y(t)).attr('y2', (t) => y(t));
    const x0 = d3.scaleBand().domain(rows.map((r) => r.key)).range([left, W]).paddingInner(0.18).paddingOuter(0.05);
    const x1 = d3.scaleBand().domain(keys).range([0, x0.bandwidth()]).paddingInner(0.06);
    const g = svg.append('g').selectAll('g').data(rows).join('g').attr('transform', (r) => `translate(${x0(r.key)},0)`);
    for (const k of keys) {
      const gk = g.filter((r) => r[k] != null);
      gk.append('rect').attr('x', x1(k)).attr('width', x1.bandwidth()).attr('y', (r) => y(r[k])).attr('height', (r) => y(0) - y(r[k]))
        .attr('rx', 3).style('fill', ST_COLOR[k]);
      if (x1.bandwidth() >= 24) gk.append('text').attr('class', 'ple-lbl').attr('text-anchor', 'middle').attr('x', x1(k) + x1.bandwidth() / 2)
        .attr('y', (r) => y(r[k]) - 5).text((r) => r[k]);
    }
    g.append('text').attr('class', 'ple-lbl').attr('text-anchor', 'middle').attr('x', x0.bandwidth() / 2).attr('y', H - bottom + 18).text((r) => r.model);
    g.append('text').attr('text-anchor', 'middle').attr('x', x0.bandwidth() / 2).attr('y', H - bottom + 34).text((r) => r.quant);
    g.append('rect').attr('x', 0).attr('width', x0.bandwidth()).attr('y', top).attr('height', H - bottom - top).style('fill', 'transparent')
      .on('pointerenter pointermove', (ev, r) => PLE.showTip(el, tip, ev, [`${r.model}, ${r.quant}, ${r.ctx} context`,
        ...ST_SERIES.map((s) => `${s.label}: ${r[s.key] == null ? 'not run / does not fit' : r[s.key] + ' tok/s'}`),
        r.vs_llama == null ? 'No head-to-head at this context' : `Our best against llama.cpp's best: ${r.vs_llama}×`]))
      .on('pointerleave', () => PLE.hideTip(tip));
  },
});

dash.draw('st-land-table', (tbody) => {
  const d = dash.data('st_land');
  tbody.replaceChildren();
  if (d.status !== 'ok') { d3.select(tbody).append('tr').append('td').attr('colspan', 8).text(d.status === 'error' ? 'This table could not load its data.' : '…'); return; }
  const f = (v) => v == null ? '–' : v;
  for (const r of d.data) {
    const tr = d3.select(tbody).append('tr').attr('data-key', r.id);
    tr.append('td').text(r.model);
    tr.append('td').attr('class', 'n').text(r.ctx);
    tr.append('td').attr('class', 'n').text(f(r.before));
    tr.append('td').attr('class', 'n').text(f(r.after));
    tr.append('td').attr('class', 'n').text(f(r.draft));
    tr.append('td').attr('class', 'n').text(f(r.llama));
    tr.append('td').attr('class', 'n').text(f(r.llama_spec));
    const v = r.vs_llama;
    tr.append('td').attr('class', 'n').append('span').attr('class', 'ple-pill ' + (v == null ? 'na' : v >= 1.05 ? 'ok' : v >= 0.95 ? 'weak' : 'fail'))
      .text(v == null ? 'n/a' : v.toFixed(2) + '×');
  }
});

dash.draw('st-check-table', (tbody) => {
  const d = dash.data('st_check');
  tbody.replaceChildren();
  if (d.status !== 'ok') { d3.select(tbody).append('tr').append('td').attr('colspan', 5).text('…'); return; }
  const names = { l12: 'Before (L1 + L2)', l12c: 'After (fused step)', l7b: 'INT4 self-draft', mtp1: 'MTP draft, k = 1', mtp2: 'MTP draft, k = 2', mtp3: 'MTP draft, k = 3', mtp4: 'MTP draft, k = 4', mtp3_h4: 'MTP draft, 4-bit head, k = 3', mtp4_h4: 'MTP draft, 4-bit head, k = 4' };
  for (const r of d.data) {
    const tr = d3.select(tbody).append('tr').attr('data-key', r.id);
    tr.append('td').text(r.model);
    tr.append('td').text(names[r.setting] || r.setting);
    tr.append('td').attr('class', 'n').text(r.prompts);
    tr.append('td').attr('class', 'n').text(r.violations);
    tr.append('td').attr('class', 'n').text(r.near_ties);
  }
});

dash.draw('st-q2-table', (tbody) => {
  const d = dash.data('st_q2');
  tbody.replaceChildren();
  if (d.status !== 'ok') { d3.select(tbody).append('tr').append('td').attr('colspan', 4).text('…'); return; }
  for (const r of d.data) {
    const tr = d3.select(tbody).append('tr').attr('data-key', r.id);
    tr.append('td').text(r.model);
    tr.append('td').attr('class', 'n').text(r.k === 0 ? 'no draft' : r.k);
    tr.append('td').attr('class', 'n').text(r.accept_pct == null ? '–' : r.accept_pct + '%');
    tr.append('td').attr('class', 'n').text(r.tps);
  }
});

dash.draw('st-inkling-table', (tbody) => {
  const d = dash.data('st_inkling');
  tbody.replaceChildren();
  if (d.status !== 'ok' || !d.data.length) { d3.select(tbody).append('tr').append('td').attr('colspan', 3).text(d.status === 'ok' ? 'No Inkling runs yet.' : '…'); return; }
  for (const r of [...d.data].sort((a, b) => a.context - b.context || a.k - b.k)) {
    const tr = d3.select(tbody).append('tr').attr('data-key', r.id);
    tr.append('td').text(r.config + (r.k ? `, MTP draft k = ${r.k}` : ''));
    tr.append('td').attr('class', 'n').text(r.ctx);
    tr.append('td').attr('class', 'n').text(r.tps);
  }
});

document.querySelectorAll('#dash-root .ple-st-ctx').forEach((sel) => {
  sel.value = String(PLE.stCtx);
  sel.addEventListener('change', () => { PLE.stCtx = Number(sel.value); dash.draw('st-bars'); });
});
document.querySelectorAll('#dash-root .ple-st-model').forEach((sel) => {
  sel.value = PLE.stModel;
  sel.addEventListener('change', () => {
    PLE.stModel = sel.value;
    document.querySelectorAll('#dash-root [data-draw="st-lines"][data-follow]').forEach((el) => { el.dataset.model = sel.value; });
    dash.draw('st-lines');
  });
});
document.querySelectorAll('#dash-root [data-st-swatch]').forEach((sw) => { sw.style.background = ST_COLOR[sw.dataset.stSwatch]; });
