// Plays N in-process Showdown battles with seeded random legal players driven by the JS tracker + encoder, and
// records per side the exact message stream (logs, request JSON, errors) plus, at every decision point, the JS
// encode() (sparse), actionMask(), actionToChoice() for every action and stateValue(), for scripts/fastenv/parity_v1.py.
//
//   node scripts/fastenv/record_battles.js N FORMAT OUT.jsonl [SEED=1] [--bench] [--v2]     (OUT '-' = stdout)
//   --v2: decisions carry encoder_v2 fields (o2) and aux targets (a2, vs the other side's tracker) as sparse
//   [indices, values in the field dtype], and k = the other side's event count at the decision (for aux)
//
// Player policy (deterministic given SEED): uniform over legal actions, except 15% "switch if any switch is
// legal" and 30% "terastallize if tera is legal".  One JSON line per battle:
//   {format, id, seed, winner, turns, sides: [{ev: [[kind, text], ...], dec: [...], final, crash}, x2]}
//   ev kind: 'L' log chunk, 'R' request JSON, 'E' error text; dec: {e: index into ev of the triggering event,
//   obs: [base64 uint16 indices, base64 float32 values], mask: [...], ch: [choice per action], v: state value, a: action}
// --bench: no output file; prints timing of tracker updates and encode() only.
'use strict';
const fs = require('fs');
const path = require('path');
const JS = path.join(__dirname, '../../fastenv/js');
const {SimBattle, Teams} = require(path.join(JS, 'battle.js'));
const {SideTracker, PokeEnvError} = require(path.join(JS, 'tracker.js'));
const E = require(path.join(JS, 'encoder_v1.js'));
const E2 = require(path.join(JS, 'encoder_v2.js'));
let V2 = false;  // --v2: record encoder_v2 fields + aux (sparse per field) instead of v1 obs / choices

function sparseTyped(a) {  // [b64 uint16 indices, b64 values in the array's own dtype]
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) n++;
  const ix = new Uint16Array(n), vs = new a.constructor(n);
  for (let i = 0, k = 0; i < a.length; i++) if (a[i] !== 0) { ix[k] = i; vs[k++] = a[i]; }
  return [Buffer.from(ix.buffer).toString('base64'), Buffer.from(vs.buffer, vs.byteOffset, vs.byteLength).toString('base64')];
}

const SHAPING = {fainted: 0.05, hp: 0.05, status: 0, victory: 1};

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function sparse(obs) {
  let n = 0;
  for (let i = 0; i < obs.length; i++) if (obs[i] !== 0) n++;
  const ix = new Uint16Array(n), vs = new Float32Array(n);
  for (let i = 0, k = 0; i < obs.length; i++) if (obs[i] !== 0) { ix[k] = i; vs[k++] = obs[i]; }
  return [Buffer.from(ix.buffer).toString('base64'), Buffer.from(vs.buffer).toString('base64')];
}

const timing = {trackNs: 0n, trackCalls: 0, encNs: 0n, encCalls: 0, auxNs: 0n, turns: 0, decisions: 0, errors: 0};

function pickAction(rng, mask, gen) {
  const legal = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
  if (!legal.length) return -2;
  const sw = legal.filter(a => a < 6), tera = legal.filter(a => a >= 22);
  const r = rng();
  if (r < 0.15 && sw.length) return sw[Math.floor(rng() * sw.length)];
  if (r < 0.45 && tera.length) return tera[Math.floor(rng() * tera.length)];
  return legal[Math.floor(rng() * legal.length)];
}

function playBattle(format, id, seed, record) {
  const rng = mulberry32(seed * 1000003 + id * 7919);
  const s4 = () => [0, 0, 0, 0].map(() => Math.floor(rng() * 65536));
  const teams = [Teams.generate(format, {seed: s4()}), Teams.generate(format, {seed: s4()})];
  const gen = parseInt(format[3], 10), nA = E.nActions(gen);
  const sides = [0, 1].map(s => ({t: new SideTracker({format, side: s}), ev: [], dec: [], final: null, crash: null,
    obs: new Float32Array(E.OBS_DIM), mask: new Uint8Array(nA),
    o2: V2 ? E2.allocate(E2.fields(gen), 1) : null, a2: V2 ? E2.allocate(E2.auxFields(gen), 1) : null}));
  const queue = [];
  const b = new SimBattle(format, s4(), teams, {
    onLog: (s, text) => queue.push([s, 'L', text, null]),
    onRequest: (s, req, json) => queue.push([s, 'R', json, req]),
    onError: (s, text) => { timing.errors++; queue.push([s, 'E', text, null]); },
  });
  let crashed = false;
  // computed after the whole update is drained (both players' requests in), as the host does
  const decide = (s, e) => {
    const S = sides[s], O = sides[1 - s], t = S.t;
    E.actionMask(t, S.mask, 0);
    const d = {e, k: O.ev.length, mask: Array.from(S.mask), v: E.stateValue(t, SHAPING), a: null, done: false};
    let t0 = process.hrtime.bigint();
    if (V2) {
      E2.encodeFields(t, S.o2, 0);
      const t1 = process.hrtime.bigint();
      timing.encNs += t1 - t0; timing.encCalls++;
      E2.encodeAux(t, O.crash ? null : O.t, null, S.a2, 0);
      timing.auxNs += process.hrtime.bigint() - t1;
      if (record) {
        d.o2 = {}; d.a2 = {};
        for (const k in S.o2) d.o2[k] = sparseTyped(S.o2[k]);
        for (const k in S.a2) d.a2[k] = sparseTyped(S.a2[k]);
      }
    } else {
      E.encode(t, S.obs, 0);
      timing.encNs += process.hrtime.bigint() - t0; timing.encCalls++;
      if (record) { d.obs = sparse(S.obs); d.ch = []; for (let a = 0; a < nA; a++) d.ch.push(E.actionToChoice(t, a)); }
    }
    S.dec.push(d);
    timing.decisions++;
  };
  const drain = () => {
    const due = [];
    while (queue.length) {
      const [s, kind, text, req] = queue.shift();
      const S = sides[s];
      S.ev.push([kind, text]);
      if (S.crash) continue;
      const t0 = process.hrtime.bigint();
      try {
        if (kind === 'L') { S.t.onLog(text); if (text.indexOf('|turn|') >= 0) timing.turns++; }
        else if (kind === 'R') S.t.onRequest(req);
        else S.t.onError(text);
      } catch (err) {
        if (!(err instanceof PokeEnvError)) throw err;
        S.crash = {e: S.ev.length - 1, msg: err.message};
        crashed = true;
        continue;
      } finally {
        timing.trackNs += process.hrtime.bigint() - t0; timing.trackCalls++;
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
  let guard = 0, defaults = 0;
  while (!b.ended && !crashed && guard++ < 5000) {
    const pend = b.pendingSides();
    if (!pend.length) throw new Error(`battle ${id}: no pending side`);
    for (const s of pend) {
      const S = sides[s];
      const d = S.dec[S.dec.length - 1];
      let choice = 'default';
      if (d && !d.done && S.t.decisionPending) {
        d.a = pickAction(rng, d.mask, gen);
        choice = d.ch ? d.ch[d.a] : E.actionToChoice(S.t, d.a);
        d.done = true;
        S.t.markChosen();
      } else defaults++;
      const ok = b.choose(s, choice);
      drain();
      if (!ok && !S.t.decisionPending && !b.ended && b.pendingSides().includes(s)) { b.choose(s, 'default'); drain(); defaults++; }
      if (b.ended || crashed) break;
    }
  }
  if (!b.ended && !crashed) throw new Error(`battle ${id} did not end`);
  for (const S of sides) {
    if (!S.crash) S.final = E.stateValue(S.t, SHAPING);
    for (const d of S.dec) delete d.done;
  }
  return {format, id, seed, winner: b.winner, turns: b.turn, defaults,
    sides: sides.map(S => ({ev: S.ev, dec: S.dec, final: S.final, crash: S.crash}))};
}

function main() {
  const args = process.argv.slice(2);
  const bench = args.includes('--bench');
  V2 = args.includes('--v2');
  const pos = args.filter(a => !a.startsWith('--'));
  const n = parseInt(pos[0] || '10', 10), format = pos[1] || 'gen9randombattle';
  const out = bench ? null : pos[2], seed = parseInt(pos[3] || '1', 10);
  const fd = out ? (out === '-' ? 1 : fs.openSync(out, 'w')) : null;
  let crashes = 0, defaults = 0;
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const r = playBattle(format, i, seed, !bench);
    crashes += r.sides.some(S => S.crash) ? 1 : 0;
    defaults += r.defaults;
    if (fd !== null) fs.writeSync(fd, JSON.stringify(r) + '\n');
  }
  if (fd !== null && fd !== 1) fs.closeSync(fd);
  const us = (ns, c) => (Number(ns) / 1e3 / Math.max(c, 1)).toFixed(2);
  console.error(`${format}: ${n} battles in ${((Date.now() - t0) / 1e3).toFixed(1)}s, ${timing.decisions} decisions, ` +
    `${crashes} battles with a PokeEnvError, ${defaults} 'default' fallbacks, ${timing.errors} Showdown choice errors`);
  console.error(`tracker: ${us(timing.trackNs, timing.trackCalls)} us/message-chunk, ` + (V2 ? `aux ${us(timing.auxNs, timing.encCalls)} us/call, ` : '') +
    `${us(timing.trackNs, timing.turns * 2)} us per side-turn; encode: ${us(timing.encNs, timing.encCalls)} us/call`);
}

main();
