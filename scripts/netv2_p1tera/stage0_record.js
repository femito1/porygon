'use strict';
// netv2 2026-11-24, council P1 (spec v2.2a = own pre-Tera visibility), Stage 0, JS side. The battle loop is ws/fastenv's
// scripts/fastenv/record_battles.js --v2 (seeded random legal players: 15% switch, 30% tera if legal, else uniform).
//   node stage0_record.js stream N FORMAT SEED OLD_JS_DIR   stdout: one JSON line per battle for stage0_check.py (both
//        sides' message streams; at every decision the new encoder's flag-off observation + aux (sparse) and its flag-on
//        tera / tera_p) + a JS line per battle with the JS-only checks: flag off == OLD_JS_DIR's encoder (today's code)
//        byte for byte (obs + aux); flag on == flag off except tera / tera_p; aux unchanged by the flag; v21_view(flag on)
//        == flag off byte for byte; terastallized == 0 for every own mon that has not terastallized
//   node stage0_record.js leak N FORMAT SEED                stdout: one JSON summary: each battle played twice, the
//        second time with the unrevealed tera types of one side's team (the "victim"; mons that never terastallize
//        in the first pass; not Ogerpon / Terapagos) reassigned; the other side's message stream and flag-on
//        observations must be byte-identical at every decision (and the victim's own observations differ)
//   node stage0_record.js volcarona FORMAT SEED             stdout: one JSON line: our lead Volcarona (Fiery Dance /
//        Giga Drain / Quiver Dance / Tera Blast, Flame Body, Heavy-Duty Boots: the library set) Tera Water vs Tera
//        Ground against an Iron Moth lead (Fire/Poison: Ground x4 > Water x2) and a Skarmory lead (Steel/Flying:
//        Ground immune < Water x1), our first decision encoded flag on and flag off (sparse)
// Env: PORYGON_PS (a built pokemon-showdown); STAGE0_JS overrides the new fastenv/js dir (default ../../fastenv/js).
const fs = require('fs');
const path = require('path');
const JS = process.env.STAGE0_JS || path.join(__dirname, '../../fastenv/js');
const {SimBattle, Teams} = require(path.join(JS, 'battle.js'));
const {SideTracker, PokeEnvError} = require(path.join(JS, 'tracker.js'));
const E = require(path.join(JS, 'encoder_v1.js'));
const E2 = require(path.join(JS, 'encoder_v2.js'));
const SPEC = JSON.parse(fs.readFileSync(path.join(JS, 'tables', 'spec_v2.json'), 'utf8'));
const T_COL = SPEC.MON_NUM.indexOf('terastallized'), MND = SPEC.MON_NUM.length, NT = SPEC.TYPES.length + 1;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function sparseTyped(a) {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) n++;
  const ix = new Uint16Array(n), vs = new a.constructor(n);
  for (let i = 0, k = 0; i < a.length; i++) if (a[i] !== 0) { ix[k] = i; vs[k++] = a[i]; }
  return [Buffer.from(ix.buffer).toString('base64'), Buffer.from(vs.buffer, vs.byteOffset, vs.byteLength).toString('base64')];
}
const sp = o => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, sparseTyped(v)]));
const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
const eqArr = (a, b) => bytes(a).equals(bytes(b));
function diffFields(a, b, skip = new Set()) { const out = []; for (const k in a) if (!skip.has(k) && !eqArr(a[k], b[k])) out.push(k); return out; }
function pickAction(rng, mask, noTera) {
  const legal = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] && !(noTera && i >= 22)) legal.push(i);
  if (!legal.length) return -2;
  const sw = legal.filter(a => a < 6), tera = legal.filter(a => a >= 22);
  const r = rng();
  if (r < 0.15 && sw.length) return sw[Math.floor(rng() * sw.length)];
  if (r < 0.45 && tera.length) return tera[Math.floor(rng() * tera.length)];
  return legal[Math.floor(rng() * legal.length)];
}
const clearCache = t => { for (const m of t.oppTeam.values()) { m.v2key = ''; m.v2view = null; } };

/** One seeded battle; hooks: teams(teams) mutates the teams (no rng draws), decide(s, S, O, e, d) at every decision,
 * stopAt(s, S) -> true ends the battle at that decision (fixtures). Same seed + same masks => same rng draws. */
function playBattle(format, id, seed, hooks = {}) {
  const rng = mulberry32(seed * 1000003 + id * 7919);
  const s4 = () => [0, 0, 0, 0].map(() => Math.floor(rng() * 65536));
  const teams = [Teams.generate(format, {seed: s4()}), Teams.generate(format, {seed: s4()})];
  if (hooks.teams) hooks.teams(teams);
  const gen = parseInt(format[3], 10), nA = E.nActions(gen);
  const sides = [0, 1].map(s => ({s, t: new SideTracker({format, side: s}), ev: [], dec: [], crash: null, mask: new Uint8Array(nA)}));
  const queue = [];
  const b = new SimBattle(format, s4(), teams, {
    onLog: (s, text) => queue.push([s, 'L', text, null]),
    onRequest: (s, req, json) => queue.push([s, 'R', json, req]),
    onError: (s, text) => queue.push([s, 'E', text, null]),
  });
  let crashed = false, stopped = false;
  const decide = (s, e) => {
    const S = sides[s], O = sides[1 - s];
    E.actionMask(S.t, S.mask, 0);
    const d = {e, k: O.ev.length, mask: Array.from(S.mask), a: null, done: false};
    if (hooks.decide) hooks.decide(s, S, O, e, d);
    S.dec.push(d);
    if (hooks.stopAt && hooks.stopAt(s, S)) stopped = true;
  };
  const drain = () => {
    const due = [];
    while (queue.length) {
      const [s, kind, text, req] = queue.shift();
      const S = sides[s];
      S.ev.push([kind, text]);
      if (S.crash) continue;
      try {
        if (kind === 'L') S.t.onLog(text);
        else if (kind === 'R') S.t.onRequest(req);
        else S.t.onError(text);
      } catch (err) {
        if (!(err instanceof PokeEnvError)) throw err;
        S.crash = {e: S.ev.length - 1, msg: err.message};
        crashed = true;
        continue;
      }
      if (S.t.decisionPending && kind !== 'L') due.push([s, S.ev.length - 1]);
    }
    for (const [s, e] of due) {
      if (sides[s].crash || !sides[s].t.decisionPending) continue;
      if (sides[s].ev.length - 1 !== e) throw new Error('events after a decision within one update');
      decide(s, e);
    }
  };
  b.start();
  drain();
  let guard = 0;
  while (!b.ended && !crashed && !stopped && guard++ < 5000) {
    const pend = b.pendingSides();
    if (!pend.length) throw new Error(`battle ${id}: no pending side`);
    for (const s of pend) {
      const S = sides[s];
      const d = S.dec[S.dec.length - 1];
      let choice = 'default';
      if (d && !d.done && S.t.decisionPending) {
        d.a = pickAction(rng, d.mask, hooks.noTera && hooks.noTera(s));
        choice = E.actionToChoice(S.t, d.a);
        d.done = true;
        S.t.markChosen();
      }
      const ok = b.choose(s, choice);
      drain();
      if (!ok && !S.t.decisionPending && !b.ended && b.pendingSides().includes(s)) { b.choose(s, 'default'); drain(); }
      if (b.ended || crashed || stopped) break;
    }
  }
  if (!b.ended && !crashed && !stopped) throw new Error(`battle ${id} did not end`);
  for (const S of sides) for (const d of S.dec) delete d.done;
  return {format, id, seed, winner: b.winner, turns: b.turn, teams, sides};
}

// ---- stream: parity input for Python + JS-only checks
function stream(n, format, seed, oldDir) {
  const OLD = require(path.join(path.resolve(oldDir), 'encoder_v2.js'));
  const gen = parseInt(format[3], 10);
  const F = E2.fields(gen), A = E2.auxFields(gen);
  const tot = {battles: 0, decisions: 0, crashes: 0, off_vs_old_obs: 0, off_vs_old_aux: 0, on_vs_off_other: 0,
    aux_on_vs_off: 0, v21_view_vs_off: 0, tera_pre_nonzero: 0, own_pre_tera_slots: 0, own_pre_tera_filled: 0,
    decisions_on_ne_off: 0, examples: []};
  const ex = msg => { if (tot.examples.length < 20) tot.examples.push(msg); };
  for (let id = 0; id < n; id++) {
    const bufs = [0, 1].map(() => ({on: E2.allocate(F, 1), off: E2.allocate(F, 1), old: E2.allocate(F, 1),
      aon: E2.allocate(A, 1), aoff: E2.allocate(A, 1), aold: E2.allocate(A, 1)}));
    const r = playBattle(format, id, seed, {decide: (s, S, O, e, d) => {
      const B = bufs[s], t = S.t, ot = O.crash ? null : O.t;
      clearCache(t); E2.setOwnTeraVisible(true); E2.encodeFields(t, B.on, 0); E2.encodeAux(t, ot, null, B.aon, 0);
      clearCache(t); E2.setOwnTeraVisible(false); E2.encodeFields(t, B.off, 0); E2.encodeAux(t, ot, null, B.aoff, 0);
      clearCache(t); OLD.encodeFields(t, B.old, 0); OLD.encodeAux(t, ot, null, B.aold, 0);
      const tag = `battle ${id} p${s + 1} ev ${e} turn ${t.turn}`;
      let x = diffFields(B.off, B.old); if (x.length) { tot.off_vs_old_obs++; ex(`${tag}: flag off != today's encoder in ${x}`); }
      x = diffFields(B.aoff, B.aold); if (x.length) { tot.off_vs_old_aux++; ex(`${tag}: flag-off aux != today's in ${x}`); }
      x = diffFields(B.on, B.off, new Set(['tera', 'tera_p'])); if (x.length) { tot.on_vs_off_other++; ex(`${tag}: flag on changes ${x}`); }
      x = diffFields(B.aon, B.aoff); if (x.length) { tot.aux_on_vs_off++; ex(`${tag}: flag on changes aux ${x}`); }
      // v21_view(on) (own slots 0-5 zeroed where terastallized == 0) == off
      const tv = Int16Array.from(B.on.tera), pv = Float32Array.from(B.on.tera_p);
      const mons = [...t.team.values()].slice(0, 6);
      for (let i = 0; i < 6; i++) {
        if (B.on.mon_num[MND * i + T_COL] === 0) { tv[i] = 0; pv.fill(0, NT * i, NT * (i + 1)); }
        if (i < mons.length && !mons[i].tera) {
          tot.own_pre_tera_slots++;
          if (B.on.mon_num[MND * i + T_COL] !== 0) { tot.tera_pre_nonzero++; ex(`${tag}: own slot ${i} terastallized != 0 pre-Tera`); }
          if (gen >= 9 && B.on.tera[i] !== 0) tot.own_pre_tera_filled++;
        }
      }
      if (!eqArr(tv, B.off.tera) || !eqArr(pv, B.off.tera_p)) { tot.v21_view_vs_off++; ex(`${tag}: v21_view(flag on) != flag off`); }
      if (!eqArr(B.on.tera, B.off.tera) || !eqArr(B.on.tera_p, B.off.tera_p)) tot.decisions_on_ne_off++;
      d.off = sp(B.off); d.aux = sp(B.aoff); d.on = {tera: sparseTyped(B.on.tera), tera_p: sparseTyped(B.on.tera_p)};
      tot.decisions++;
    }});
    tot.battles++;
    if (r.sides.some(S => S.crash)) tot.crashes++;
    process.stdout.write(JSON.stringify({format, id, seed, sides: r.sides.map(S => ({ev: S.ev, dec: S.dec, crash: S.crash}))}) + '\n');
  }
  process.stdout.write(JSON.stringify({js_checks: tot}) + '\n');
}

// ---- leak: tau = reassign the victim's unrevealed tera types; the observer must not see any difference
const STD_TYPES = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground',
  'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'];
function leak(n, format, seed) {
  const gen = parseInt(format[3], 10), F = E2.fields(gen);
  E2.setOwnTeraVisible(true);
  const tot = {battles: 0, compared_battles: 0, decisions_compared: 0, mons_changed: 0, stream_diverged: 0, obs_mismatch: 0,
    victim_obs_differs: 0, skipped_crash: 0, examples: []};
  const run = (id, v, mutate) => {
    const obs = [], vic = [];
    const r = playBattle(format, id, seed, {
      teams: mutate,
      decide: (s, S) => {
        const o = E2.allocate(F, 1);
        clearCache(S.t); E2.encodeFields(S.t, o, 0);
        (s === v ? vic : obs).push(Buffer.concat(Object.keys(o).sort().map(k => bytes(o[k]))));
      },
    });
    return {r, obs, vic};
  };
  for (let id = 0; id < n; id++) {
    const v = id % 2, o = 1 - v;
    tot.battles++;
    const p1 = run(id, v, null);
    if (p1.r.sides.some(S => S.crash)) { tot.skipped_crash++; continue; }
    const tera = new Set();
    const pre = `|-terastallize|p${v + 1}a: `;
    for (const [kind, text] of p1.r.sides[o].ev) if (kind === 'L') for (const l of text.split('\n')) if (l.startsWith(pre)) tera.add(l.slice(pre.length).split('|')[0]);
    let changed = 0;
    const mutate = teams => {
      for (const [j, set] of teams[v].entries()) {
        const name = set.name || set.species;
        if (tera.has(name) || /^(Ogerpon|Terapagos)/.test(set.species)) continue;
        const k = STD_TYPES.indexOf(set.teraType);
        set.teraType = STD_TYPES[(k + 1 + ((id + j) % 16) + STD_TYPES.length) % STD_TYPES.length];
        changed++;
      }
    };
    const p2 = run(id, v, mutate);
    tot.mons_changed += changed;
    if (!changed) continue;
    tot.compared_battles++;
    // the |t:| lines are wall-clock timestamps (ignored by the tracker and poke-env): not part of the comparison
    const noT = ev => ev.map(([kind, text]) => [kind, kind === 'L' ? String(text).split('\n').filter(l => !l.startsWith('|t:|')).join('\n') : text]);
    p1.r.sides[o].ev = noT(p1.r.sides[o].ev); p2.r.sides[o].ev = noT(p2.r.sides[o].ev);
    const s1 = JSON.stringify(p1.r.sides[o].ev), s2 = JSON.stringify(p2.r.sides[o].ev);
    if (s1 !== s2) {
      tot.stream_diverged++;
      const a = p1.r.sides[o].ev, b = p2.r.sides[o].ev;
      let where = `lengths ${a.length} / ${b.length}`;
      for (let i = 0; i < Math.min(a.length, b.length); i++) {
        if (a[i][0] === b[i][0] && a[i][1] === b[i][1]) continue;
        const la = String(a[i][1]).split('\n'), lb = String(b[i][1]).split('\n');
        let j = 0; while (j < la.length && la[j] === lb[j]) j++;
        where = `event ${i} (${a[i][0]}/${b[i][0]}) line ${j}: '${String(la[j]).slice(0, 160)}' vs '${String(lb[j]).slice(0, 160)}'`;
        break;
      }
      if (tot.examples.length < 20) tot.examples.push(`battle ${id}: observer p${o + 1}'s stream differs at ${where}`);
    }
    const m = Math.min(p1.obs.length, p2.obs.length);
    let bad = p1.obs.length !== p2.obs.length ? 1 : 0;
    for (let i = 0; i < m; i++) { tot.decisions_compared++; if (!p1.obs[i].equals(p2.obs[i])) bad++; }
    if (bad) { tot.obs_mismatch++; if (tot.examples.length < 20) tot.examples.push(`battle ${id}: observer p${o + 1}'s observations differ at ${bad} decisions`); }
    if (p1.vic.length && p2.vic.length && !p1.vic[0].equals(p2.vic[0])) tot.victim_obs_differs++;
  }
  process.stdout.write(JSON.stringify({leak: tot}) + '\n');
}

// ---- volcarona: Tera Water vs Tera Ground, flag on / off, first decision of p1
function volcarona(format, seed) {
  const gen = parseInt(format[3], 10), F = E2.fields(gen);
  const out = {format, seed, cases: []};
  const OPP = {ironmoth: {name: 'Iron Moth', species: 'Iron Moth', level: 78, ability: 'Quark Drive', item: 'Heavy-Duty Boots',
    moves: ['Fiery Dance', 'Sludge Wave', 'Energy Ball', 'Dazzling Gleam'], teraType: 'Grass', gender: ''},
  skarmory: {name: 'Skarmory', species: 'Skarmory', level: 80, ability: 'Sturdy', item: 'Leftovers',
    moves: ['Body Press', 'Brave Bird', 'Roost', 'Spikes'], teraType: 'Fighting', gender: ''}};
  for (const [opp, oset] of Object.entries(OPP)) for (const tt of ['Water', 'Ground']) {
    const r = playBattle(format, 0, seed, {
      teams: teams => {
        const v = teams[0][0], h = teams[1][0];
        Object.assign(v, {name: 'Volcarona', species: 'Volcarona', level: 77, ability: 'Flame Body', item: 'Heavy-Duty Boots',
          moves: ['Fiery Dance', 'Giga Drain', 'Quiver Dance', 'Tera Blast'], teraType: tt, gender: ''});
        Object.assign(h, oset);
        for (const t of teams) for (const s of t.slice(1)) if (s.species === 'Volcarona' || s.species === oset.species) s.species = s.name = 'Pikachu';
      },
      decide: (s, S, O, e, d) => {
        if (s !== 0 || d.on) return;
        const on = E2.allocate(F, 1), off = E2.allocate(F, 1);
        clearCache(S.t); E2.setOwnTeraVisible(true); E2.encodeFields(S.t, on, 0);
        clearCache(S.t); E2.setOwnTeraVisible(false); E2.encodeFields(S.t, off, 0);
        const act = S.t.activePokemon;
        out.cases.push({opp, tera: tt, active: act ? act.species : null, opp_active: S.t.oppTeam.size ? [...S.t.oppTeam.values()][0].species : null,
          moves: act ? [...act.movesMap().keys()] : [], on: sp(on), off: sp(off)});
      },
      stopAt: s => s === 0,
    });
    if (r.sides[0].crash) throw new Error('volcarona fixture crashed: ' + r.sides[0].crash.msg);
  }
  process.stdout.write(JSON.stringify(out) + '\n');
}

const [mode, ...rest] = process.argv.slice(2);
if (mode === 'stream') stream(parseInt(rest[0], 10), rest[1], parseInt(rest[2], 10), rest[3]);
else if (mode === 'leak') leak(parseInt(rest[0], 10), rest[1], parseInt(rest[2], 10));
else if (mode === 'volcarona') volcarona(rest[0], parseInt(rest[1], 10));
else { console.error('usage: stage0_record.js stream N FORMAT SEED OLD_JS_DIR | leak N FORMAT SEED | volcarona FORMAT SEED'); process.exit(2); }
