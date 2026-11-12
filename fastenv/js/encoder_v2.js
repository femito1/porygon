'use strict';
// Observation encoder for spec v2 (owner netv2; .orchestra/CONTRACTS.md "Observation spec v2"): a JS port of the
// reference `porygon.v2.encoder.encode(battle)` and `aux_targets(battle, opp_battle, obs)` (branch ws/netv2) on the
// poke-env-equivalent SideTracker state (tracker.js).  Static data (set library, set_species, move types, spec
// field lists) come from the reference via scripts/fastenv/export_tables_v2.py -> tables/{spec,gen4_lib,gen9_lib}_v2.json.
//
//   const E2 = require('./encoder_v2');
//   E2.fields(gen)      [[name, dtype, shape], ...] = spec.fields(gen) in spec order (incl. action_mask, f32)
//   E2.auxFields(gen)   [[name, dtype, shape], ...] = spec.aux_fields(gen)
//   E2.encodeFields(t, outs, row)                writes row `row` of every field into outs[name] (typed array of
//                                                rows x prod(shape)); == encode(battle)
//   E2.encodeAux(t, oppTracker, oppTrueTeam, outs, row)   == aux_targets(battle, opp_battle, obs): oppTracker = the
//                                                other side's SideTracker (poke-env's opp_battle) at the time of the
//                                                call; oppTrueTeam is unused (null oppTracker -> all "no target",
//                                                as the reference with opp_battle=None); aux_opp_target is zeros
//                                                (the trainer fills it)
//   E2.nActions, actionMask, actionToChoice, stateValue, validChoices: re-exported from encoder_v1
//   E2.SPEC_VERSION
//
// Set posterior: computed by `posterior(gen, species, level, revealed, ability, item, tera)` exactly as the reference
// (same candidate filtering / relaxation order, same accumulation order, ties by name) and memoised per opponent mon
// (recomputed only when its key changes) plus a global LRU-ish memo keyed like the reference's lru_cache.
// Parity: python scripts/fastenv/parity_v2.py --record 300 gen9randombattle (streaming vs the reference) and
//         python scripts/fastenv/parity_v2.py --fixture DIR (netv2's recorded fixtures, `check --candidate`).
//
// Matched: netv2 ws/netv2 HEAD 5f60a4f (spec.py / encoder.py last changed in 1d2574e, SPEC_VERSION 2.1).
// Reference behaviours reproduced as is (reported to the spec owner, not "fixed" here):
//  * own `tera` / `tera_p` stay 0 until the mon terastallizes: the reference reads poke-env's mon.tera_type, which
//    poke-env never fills from the request's teraType (the spec text says ours are known in gen9);
//  * `aux_tera` is -1 unless the opponent mon has terastallized (same poke-env field on the opponent's own team);
//  * an opponent item that was removed (knocked off / consumed: poke-env item None) is encoded as unknown (posterior
//    candidates, item_known 0), not as "removed = id 0, p = 1" as the spec text says;
//  * weather_elapsed is ~0 because poke-env resets the weather start turn on every [upkeep] line (as v1);
//  * aux_opp_class reads the opponent tracker's *current* request: call encodeAux after both players' requests of
//    the update are parsed (the host does), else it describes the opponent's previous decision.
// Speed (laptop i5-1235U, one P-core, 32 battles in lockstep with the simulator): encodeFields ~32 us (gen9) /
// ~28 us (gen4) incl. the memoised posterior, encodeAux ~10-11 us.

const fs = require('fs');
const path = require('path');
const E1 = require('./encoder_v1');
const {loadCommon, loadGen} = require('./tracker');

const TABLES = path.join(__dirname, 'tables');
const S = JSON.parse(fs.readFileSync(path.join(TABLES, 'spec_v2.json'), 'utf8'));
const C = loadCommon();
const KM = S.KM, KI = S.KI, KA = S.KA, TERA_OFFSET = S.TERA_OFFSET;
const PAD = 0, UNK = 1;
const TYPE_ID = new Map(S.TYPES.map((t, i) => [t, i + 1]));
const N_TYPES = S.TYPES.length + 1;
const COL = Object.fromEntries(S.MON_NUM.map((n, i) => [n, i]));
const MON_NUM_DIM = S.MON_NUM.length, FIELD_NUM_DIM = S.FIELD_NUM.length;
const EFFECT0 = COL.effect_0;
const EFFECT_IDX = new Map(C.effects.map((e, i) => [e, i]));
const WEATHER_INDEX = new Map(S.WEATHERS.map((w, i) => [w, i]));
const TERRAIN_INDEX = new Map([['ELECTRIC_TERRAIN', 0], ['GRASSY_TERRAIN', 1], ['MISTY_TERRAIN', 2], ['PSYCHIC_TERRAIN', 3]]);
const STATUS_COL = new Map(S.STATUSES.map(s => [s.toUpperCase(), COL[`status_${s}`]]));
const BOOSTS = S.BOOSTS, STATS = S.STATS;
const idx = E1.idx;
const prod = sh => sh.reduce((a, b) => a * b, 1);

const LIB = {};
function lib(gen) {
  if (LIB[gen]) return LIB[gen];
  const raw = JSON.parse(fs.readFileSync(path.join(TABLES, `gen${gen}_lib_v2.json`), 'utf8'));
  const species = new Map();
  for (const [sp, [level, sets]] of Object.entries(raw.species)) {
    species.set(sp, {level, sets: sets.map(([moves, ability, item, evs, ivs, n, tera]) => ({moves, ability, item, evs, ivs, n, tera}))});
  }
  LIB[gen] = {species, setSpecies: new Map(Object.entries(raw.setSpecies)), moveType: new Map(Object.entries(raw.moveType)),
    memo: new Map()};
  return LIB[gen];
}

const TYPE_ID_UP = new Map(S.TYPES.map((t, i) => [t.toUpperCase(), i + 1]));  // poke-env enum names
const typeIdOf = upper => (upper === null || upper === undefined ? 0 : TYPE_ID_UP.get(upper) || 0);
const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, '');
function stripDigits(s) { let e = s.length; while (e > 0 && s.charCodeAt(e - 1) >= 48 && s.charCodeAt(e - 1) <= 57) e--; return s.slice(0, e); }

/** reference move_type_of(gen, move_id) */
function moveTypeOf(gen, G, id) {
  const L = lib(gen);
  const v = L.moveType.get(id);
  if (v !== undefined) return v;
  if (id.startsWith('hiddenpower') && id.length > 11) {
    const t = stripDigits(id.slice(11));
    if (TYPE_ID.has(t)) return TYPE_ID.get(t);
  }
  const e = G.moves.get(id) || G.moves.get(stripDigits(id));
  return e ? (TYPE_ID.get(e.type.toLowerCase()) || 0) : 0;
}

function setSpecies(gen, species) {
  const L = lib(gen);
  if (L.species.has(species)) return species;
  const s = L.setSpecies.get(species);
  return s === undefined ? species : s;
}

const matches = (r, m) => m === r || (r === 'hiddenpower' && m.startsWith('hiddenpower'));
function movesOk(s, moves) {
  for (const r of moves) {
    let ok = false;
    for (const m of s.moves) if (matches(r, m)) { ok = true; break; }
    if (!ok) return false;
  }
  return true;
}
/** reference consistent_revealed */
function consistentRevealed(sets, revealed, ability, item, tera) {
  const levels = [
    s => movesOk(s, revealed) && (tera === null || s.tera === tera) && (ability === null || s.ability === ability) && (item === null || s.item === item),
    s => movesOk(s, revealed) && (tera === null || s.tera === tera) && (ability === null || s.ability === ability),
    s => movesOk(s, revealed) && (tera === null || s.tera === tera),
    s => movesOk(s, revealed),
  ];
  for (const ok of levels) {
    const found = sets.filter(ok);
    if (found.length) return found;
  }
  const known = revealed.filter(r => sets.some(s => s.moves.some(m => matches(r, m))));
  const f = sets.filter(s => movesOk(s, known));
  return f.length ? f : sets;
}
function calcStats(G, sp, level, evs, ivs) {
  const b = G.dex.get(sp).bs;
  const out = [Math.floor((2 * b[0] + ivs[0] + Math.floor(evs[0] / 4)) * level / 100) + level + 10];
  for (let k = 1; k < 6; k++) out.push(Math.floor((2 * b[k] + ivs[k] + Math.floor(evs[k] / 4)) * level / 100) + 5);
  return out;
}
const ranked = m => [...m.entries()].sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

/** reference posterior(): {pm, pi, pa, pt, st} (ranked [key, p] lists, expected stats or null) or null */
function posterior(gen, G, species, level, revealed, ability, item, tera) {
  const L = lib(gen);
  const key = `${species}|${level}|${revealed.join(',')}|${ability}|${item}|${tera}`;
  let r = L.memo.get(key);
  if (r !== undefined) return r;
  r = null;
  const sp = setSpecies(gen, species);
  const e = L.species.get(sp);
  if (e && e.sets.length) {
    const cands = consistentRevealed(e.sets, revealed, ability, item, tera);
    let tot = 0;
    for (const s of cands) tot += s.n;
    const moves = new Map(), items = new Map(), abilities = new Map(), teras = new Map();
    const st = [0, 0, 0, 0, 0, 0];
    const dexOk = G.dex.has(sp);
    for (const s of cands) {
      const p = s.n / tot;
      for (const m of s.moves) moves.set(m, (moves.get(m) || 0) + p);
      items.set(s.item, (items.get(s.item) || 0) + p);
      abilities.set(s.ability, (abilities.get(s.ability) || 0) + p);
      if (s.tera) teras.set(s.tera, (teras.get(s.tera) || 0) + p);
      if (dexOk) { const c = calcStats(G, sp, level, s.evs, s.ivs); for (let k = 0; k < 6; k++) st[k] += p * c[k]; }
    }
    r = {pm: ranked(moves), pi: ranked(items), pa: ranked(abilities), pt: ranked(teras), st: dexOk ? st : null};
  }
  if (L.memo.size > 200000) L.memo.clear();
  L.memo.set(key, r);
  return r;
}

function estimateStats(mon, level) {
  const b = mon.baseStatsP;
  const out = [Math.floor((2 * b[0] + 31 + 21) * level / 100) + level + 10];
  for (let k = 1; k < 6; k++) out.push(Math.floor((2 * b[k] + 31 + 21) * level / 100) + 5);
  return out;
}

// ---------------------------------------------------------------- field layout
const FIELDS = {4: S.fields[4], 9: S.fields[9]};
const AUX = {4: S.auxFields[4], 9: S.auxFields[9]};
const WG = {4: {}, 9: {}};  // gen -> name -> width per row (action_mask / aux_opp_class differ by gen)
for (const g of [4, 9]) for (const [k, , sh] of [...FIELDS[g], ...AUX[g]]) WG[g][k] = prod(sh);
function fields(gen) { return FIELDS[gen >= 9 ? 9 : 4].map(([k, d, sh]) => [k, d, sh.slice()]); }
function auxFields(gen) { return AUX[gen >= 9 ? 9 : 4].map(([k, d, sh]) => [k, d, sh.slice()]); }

const elapsed = (start, turn) => Math.min(Math.max(turn - start, 0), 8) / 8;

function encodeSide(a, o, conds, turn) {
  for (let i = 0; i < C.screens.length; i++) {
    const v = conds.get(C.screens[i]);
    if (v !== undefined) { a[o + i] = 1; a[o + C.screens.length + i] = elapsed(v, turn); }
  }
  for (let i = 0; i < C.hazards.length; i++) {
    const [name, mx] = C.hazards[i];
    a[o + 2 * C.screens.length + i] = Math.min(conds.get(name) || 0, mx) / mx;
  }
}

function encodeFieldNum(a, o, t, act) {
  const turn = t.turn;
  let wEl = 0, tEl = 0;
  for (const [w, st] of t.weather) {
    const name = w === 'SNOWSCAPE' ? 'snow' : w.toLowerCase();
    const wi = WEATHER_INDEX.get(name);
    if (wi !== undefined) { a[o + wi] = 1; wEl = elapsed(st, turn); }
  }
  let p = o + 8;
  a[p++] = wEl;
  for (const [f, st] of t.fields) {
    const ti = TERRAIN_INDEX.get(f);
    if (ti !== undefined) { a[p + ti] = 1; tEl = elapsed(st, turn); }
  }
  p += 4;
  a[p++] = tEl;
  const tr = t.fields.get('TRICK_ROOM');
  a[p++] = tr !== undefined ? 1 : 0;
  a[p++] = tr !== undefined ? elapsed(tr, turn) : 0;
  a[p++] = t.fields.has('GRAVITY') ? 1 : 0;
  let alive = 0, oppF = 0;
  for (const m of t.team.values()) if (!m.fainted) alive++;
  for (const m of t.oppTeam.values()) if (m.fainted) oppF++;
  a[p++] = Math.min(turn, 100) / 100;
  a[p++] = (6 - t.oppTeam.size) / 6;
  a[p++] = alive / 6;
  a[p++] = (6 - oppF) / 6;
  a[p++] = t.forceSwitch ? 1 : 0;
  a[p++] = t.trapped ? 1 : 0;
  a[p++] = t.maybeTrapped ? 1 : 0;
  a[p++] = t.canTera ? 1 : 0;
  a[p++] = t.usedTera ? 1 : 0;
  a[p++] = t.oppUsedTera ? 1 : 0;
  a[p++] = t.wait ? 1 : 0;
  a[p++] = act === null ? 1 : 0;
  a[p++] = t.availableSwitches.length / 5;
  a[p++] = t.availableMoves.length / 4;
  if (p - o !== FIELD_NUM_DIM) throw new Error('encoder_v2 field_num layout mismatch');
}

const speciesKey = mon => (C.vocabMaps.species.has(mon.species) ? mon.species : mon.baseSpecies);

function monCommon(O, R, i, mon, t, mine) {
  O.species[R.species + i] = idx('species', speciesKey(mon));
  const tt = mon.tempTypes.length ? mon.tempTypes : (mon.type2 === null ? [mon.type1] : [mon.type1, mon.type2]);
  O.types[R.types + 2 * i] = tt.length ? typeIdOf(tt[0]) : 0;
  O.types[R.types + 2 * i + 1] = tt.length > 1 ? typeIdOf(tt[1]) : 0;
  const bo = R.boosts + 7 * i;
  for (let b = 0; b < 7; b++) O.boosts[bo + b] = mon.boosts[BOOSTS[b]] || 0;
  const n = O.mon_num, o = R.mon_num + MON_NUM_DIM * i;
  n[o + COL.present] = 1;
  n[o + COL.mine] = mine ? 1 : 0;
  n[o + COL.active] = mon.active ? 1 : 0;
  n[o + COL.fainted] = mon.fainted ? 1 : 0;
  n[o + COL.revealed] = 1;
  n[o + COL.hp_frac] = mon.hpFraction;
  if (mon.status !== null) { const c = STATUS_COL.get(mon.status); if (c !== undefined) n[o + c] = 1; }
  n[o + COL.status_counter] = Math.min(mon.statusCounter, 10) / 10;
  n[o + COL.terastallized] = mon.tera ? 1 : 0;
  n[o + COL.first_turn] = mon.activeTurns === 1 ? 1 : 0;
  n[o + COL.protect_counter] = mon.protectCounter / 3;
  n[o + COL.must_recharge] = mon.mustRecharge ? 1 : 0;
  n[o + COL.preparing] = mon.preparing ? 1 : 0;
  n[o + COL.trapped] = (mine && mon.active && t.trapped) ? 1 : 0;
  n[o + COL.log_weight] = Math.log10(Math.max(mon.weight, 0.1)) / 3;
  for (const e of mon.effects.keys()) { const k = EFFECT_IDX.get(e); if (k !== undefined) n[o + EFFECT0 + k] = 1; }
  return o;
}

function encodeOwn(O, R, i, mon, t, availIds) {
  const gen = t.gen, G = t.G;
  const o = monCommon(O, R, i, mon, t, true);
  const n = O.mon_num;
  const level = mon.level || 80;
  n[o + COL.level] = level / 100;
  if (gen >= 9 && mon.teraType !== null) {
    const ty = typeIdOf(mon.teraType);
    O.tera[R.tera + i] = ty;
    O.tera_p[R.tera_p + N_TYPES * i + ty] = 1;
  }
  const item = mon.item === C.unknownItem ? null : mon.item;
  O.item[R.item + KI * i] = idx('items', item);
  O.item_p[R.item_p + KI * i] = 1;
  n[o + COL.item_known] = 1;
  n[o + COL.item_none] = item === '' ? 1 : 0;
  O.ability[R.ability + KA * i] = idx('abilities', mon.abilityP);
  O.ability_p[R.ability_p + KA * i] = 1;
  n[o + COL.ability_known] = 1;
  let j = 0;
  for (const mv of mon.movesMap().values()) {
    if (j >= 4) break;
    const s = KM * i + j;
    O.moves[R.moves + s] = idx('moves', mv.id);
    O.move_p[R.move_p + s] = 1;
    O.move_type[R.move_type + s] = mv.id.startsWith('hiddenpower') ? moveTypeOf(gen, G, mv.id) : typeIdOf(mv.e.typeName);
    const q = R.move_num + 3 * s;
    O.move_num[q] = 1;
    O.move_num[q + 1] = mv.pp / Math.max(mv.maxPP, 1);
    O.move_num[q + 2] = mon.active && availIds.includes(mv.id) ? 1 : 0;
    j++;
  }
  n[o + COL.n_moves_known] = j / 4;
  const st = mon.stats;
  let stats;
  if (st.atk && st.def && st.spa && st.spd && st.spe && mon.maxHpP) stats = [mon.maxHpP, st.atk, st.def, st.spa, st.spd, st.spe];
  else stats = estimateStats(mon, level);
  const so = R.stats + 6 * i;
  for (let k = 0; k < 6; k++) O.stats[so + k] = stats[k];
  O.hp[R.hp + 2 * i] = mon.hp || 0;
  O.hp[R.hp + 2 * i + 1] = mon.maxHpP || O.stats[so];
}

/** Opponent mon view: the reference's derived quantities, memoised on the mon while its key is unchanged. */
function oppView(t, mon) {
  const gen = t.gen, G = t.G;
  const revealed = [];
  const mm = mon.movesMap();
  for (const k of mm.keys()) if (k !== 'struggle') revealed.push(k);  // keys are ids: norm(k) === k
  let ability = mon.abilityP;
  if (ability === null && mon.possibleAbilities.length === 1) ability = norm(mon.possibleAbilities[0]);
  const item = (mon.item === C.unknownItem || mon.item === null) ? null : mon.item;
  const tera = gen >= 9 && mon.tera && mon.teraType !== null ? mon.teraType.toLowerCase() : null;
  const level = mon.level || (lib(gen).species.get(setSpecies(gen, mon.species)) || {level: 80}).level || 80;
  const key = `${mon.species}|${level}|${revealed.join(',')}|${ability}|${item}|${tera}`;
  if (mon.v2key === key && mon.v2view !== null) return mon.v2view;
  const post = posterior(gen, G, mon.species, level, revealed, ability, item, tera);
  const pm = post ? post.pm : [];
  // move slots: revealed (order of revelation) then candidates by marginal probability
  const ids = [], probs = [], types = [], known = [], pp = [];
  const used = new Set();
  for (const mid of revealed.slice(0, KM)) {
    const mv = mm.get(mid);
    let ty = typeIdOf(mv.e.typeName);
    if (mid === 'hiddenpower') {
      const hp = pm.filter(([m]) => m.startsWith('hiddenpower')).map(([m]) => m);
      if (hp.length) { ty = moveTypeOf(gen, G, hp[0]); for (const h of hp) used.add(h); }
    }
    used.add(mid);
    ids.push(idx('moves', mid)); probs.push(1); types.push(ty); known.push(1); pp.push(mid);
  }
  for (const [mid, p] of pm) {
    if (ids.length >= KM) break;
    if (used.has(mid)) continue;
    ids.push(idx('moves', mid)); probs.push(p); types.push(moveTypeOf(gen, G, mid)); known.push(0); pp.push(null);
  }
  const view = {level, revealed, ability, item, tera, post, ids, probs, types, known, pp};
  mon.v2key = key;
  mon.v2view = view;
  return view;
}

function encodeOpp(O, R, i, mon, t) {
  const gen = t.gen;
  const o = monCommon(O, R, i, mon, t, false);
  const n = O.mon_num;
  const v = oppView(t, mon);
  const mm = mon.movesMap();
  n[o + COL.level] = v.level / 100;
  if (v.tera !== null) {
    const ty = typeIdOf(mon.teraType);
    O.tera[R.tera + i] = ty;
    O.tera_p[R.tera_p + N_TYPES * i + ty] = 1;
  }
  const post = v.post;
  const io = KI * i, ao = KA * i;
  if (v.item !== null) {
    O.item[R.item + io] = idx('items', v.item); O.item_p[R.item_p + io] = 1;
    n[o + COL.item_known] = 1;
    n[o + COL.item_none] = v.item === '' ? 1 : 0;
  } else if (post && post.pi.length) {
    for (let k = 0; k < Math.min(KI, post.pi.length); k++) {
      O.item[R.item + io + k] = idx('items', post.pi[k][0]); O.item_p[R.item_p + io + k] = post.pi[k][1];
    }
  } else { O.item[R.item + io] = UNK; O.item_p[R.item_p + io] = 1; }
  if (v.ability !== null) {
    O.ability[R.ability + ao] = idx('abilities', v.ability); O.ability_p[R.ability_p + ao] = 1;
    n[o + COL.ability_known] = 1;
  } else if (post && post.pa.length) {
    for (let k = 0; k < Math.min(KA, post.pa.length); k++) {
      O.ability[R.ability + ao + k] = idx('abilities', post.pa[k][0]); O.ability_p[R.ability_p + ao + k] = post.pa[k][1];
    }
  } else {
    let poss = mon.possibleAbilities.map(norm).slice(0, KA);
    if (!poss.length) poss = [null];
    for (let k = 0; k < poss.length; k++) {
      O.ability[R.ability + ao + k] = idx('abilities', poss[k]); O.ability_p[R.ability_p + ao + k] = 1 / poss.length;
    }
  }
  if (v.tera === null && gen >= 9 && post && post.pt.length) {
    const to = R.tera_p + N_TYPES * i;
    for (const [ty, p] of post.pt) { const c = TYPE_ID.get(ty) || 0; O.tera_p[to + c] = Math.fround(O.tera_p[to + c] + p); }
    O.tera_p[to] = 0;
  }
  for (let j = 0; j < v.ids.length; j++) {
    const s = KM * i + j;
    O.moves[R.moves + s] = v.ids[j];
    O.move_p[R.move_p + s] = v.probs[j];
    O.move_type[R.move_type + s] = v.types[j];
    const q = R.move_num + 3 * s;
    O.move_num[q] = v.known[j];
    if (v.pp[j] !== null) { const mv = mm.get(v.pp[j]); O.move_num[q + 1] = mv.pp / Math.max(mv.maxPP, 1); }
    else O.move_num[q + 1] = 1;
  }
  n[o + COL.n_moves_known] = Math.min(v.revealed.length, 4) / 4;
  const stats = post && post.st !== null ? post.st : estimateStats(mon, v.level);  // estimate: current base stats
  const so = R.stats + 6 * i;
  for (let k = 0; k < 6; k++) O.stats[so + k] = stats[k];
  O.hp[R.hp + 2 * i] = mon.hpFraction * stats[0];
  O.hp[R.hp + 2 * i + 1] = stats[0];
}

const MASK = new Uint8Array(32);
const AVAIL = [];
/** encode(battle) -> row `row` of outs[name] */
function encodeFields(t, outs, row) {
  const F = FIELDS[t.gen >= 9 ? 9 : 4], W = WG[t.gen >= 9 ? 9 : 4];
  const R = {};
  for (const [k] of F) {
    const w = W[k];
    R[k] = row * w;
    outs[k].fill(0, row * w, (row + 1) * w);
  }
  const act = t.activePokemon;
  encodeFieldNum(outs.field_num, R.field_num, t, act);
  encodeSide(outs.side_num, R.side_num, t.sideConds, t.turn);
  encodeSide(outs.side_num, R.side_num + W.side_num / 2, t.oppSideConds, t.turn);
  AVAIL.length = 0;
  for (const m of t.availableMoves) AVAIL.push(m.id);
  let i = 0;
  for (const m of t.team.values()) { if (i >= 6) break; encodeOwn(outs, R, i, m, t, AVAIL); i++; }
  i = 0;
  for (const m of t.oppTeam.values()) { if (i >= 6) break; encodeOpp(outs, R, 6 + i, m, t); i++; }
  const nA = E1.nActions(t.gen);
  E1.actionMask(t, MASK, 0);
  const ma = outs.action_mask, mo = R.action_mask;
  for (let a = 0; a < nA && a < W.action_mask; a++) ma[mo + a] = MASK[a];
}

/** aux_targets(battle, opp_battle, obs) -> row `row` of outs[name] (opp = the other side's SideTracker) */
function encodeAux(t, opp, oppTrueTeam, outs, row) {
  const gen = t.gen;
  const A = AUX[gen >= 9 ? 9 : 4], W = WG[gen >= 9 ? 9 : 4];
  const R = {};
  for (const [k] of A) { const w = W[k]; R[k] = row * w; outs[k].fill(k === 'aux_opp_target' ? 0 : -1, row * w, (row + 1) * w); }
  if (!opp) return;
  const theirs = opp.team;
  const seen = [...t.oppTeam.keys()];
  const ourOpp = [...t.oppTeam.values()];
  for (let k = 0; k < Math.min(6, seen.length); k++) {
    let tr = theirs.get(seen[k]);
    if (!tr) { for (const m of theirs.values()) if (m.species === ourOpp[k].species) { tr = m; break; } }
    if (!tr) continue;
    outs.aux_item[R.aux_item + k] = idx('items', tr.item !== C.unknownItem ? tr.item : null);
    outs.aux_ability[R.aux_ability + k] = idx('abilities', tr.abilityP);
    if (gen >= 9 && tr.teraType !== null) outs.aux_tera[R.aux_tera + k] = typeIdOf(tr.teraType);
    const trueIds = [];
    for (const m of tr.movesMap().keys()) { if (trueIds.length >= 4) break; trueIds.push(idx('moves', m)); }
    for (let j = 0; j < trueIds.length; j++) outs.aux_moves[R.aux_moves + 4 * k + j] = trueIds[j];
    const ids = oppView(t, ourOpp[k]).ids;
    for (let j = 0; j < ids.length; j++) {
      if (ids[j] === PAD) continue;
      outs.aux_move_in_set[R.aux_move_in_set + KM * k + j] = trueIds.includes(ids[j]) ? 1 : 0;
    }
  }
  let u = 0;
  for (const [key, m] of theirs) {
    if (t.oppTeam.has(key)) continue;
    if (u >= 6) break;
    outs.aux_unseen[R.aux_unseen + u++] = idx('species', speciesKey(m));
  }
  // opponent's action (its frame) -> class in our frame
  if (opp.wait) return;
  const nA = E1.nActions(opp.gen);
  E1.actionMask(opp, MASK, 0);
  const team = [...opp.team.keys()];
  let slot = null;
  for (let k = 0; k < Math.min(6, seen.length); k++) if (ourOpp[k].active) { slot = k; break; }
  const act = opp.activePokemon;
  const known = [];
  if (act !== null) for (const m of act.movesMap().values()) { if (known.length >= 4) break; known.push(m); }
  const slotIds = slot !== null ? oppView(t, ourOpp[slot]).ids : null;
  const oc = outs.aux_opp_class, oo = R.aux_opp_class;
  for (let a = 0; a < nA; a++) {
    if (!MASK[a]) continue;
    if (a < 6) {
      const key = a < team.length ? team[a] : null;
      const si = key === null ? -1 : seen.indexOf(key);
      oc[oo + a] = si >= 0 && si < 6 ? si : 6;
    } else if (slot !== null && ((a >= 6 && a < 10) || (a >= TERA_OFFSET && a < TERA_OFFSET + 4))) {
      const j = (a - 6) % 4;
      if (j >= known.length) continue;
      const mid = idx('moves', known[j].id);
      const s = slotIds.indexOf(mid);
      if (s >= 0) oc[oo + a] = 7 + s + (a >= TERA_OFFSET ? KM : 0);
    }
  }
}

/** Typed-array dict for `rows` rows of the given field list (helper for tests / tools). */
function allocate(fieldList, rows) {
  const T = {f32: Float32Array, i32: Int32Array, u32: Uint32Array, i16: Int16Array, u16: Uint16Array, u8: Uint8Array, i8: Int8Array};
  const o = {};
  for (const [k, d, sh] of fieldList) o[k] = new T[d](rows * prod(sh));
  return o;
}

module.exports = {SPEC_VERSION: S.version, fields, auxFields, encodeFields, encodeAux, allocate,
  nActions: E1.nActions, actionMask: E1.actionMask, actionToChoice: E1.actionToChoice, validChoices: E1.validChoices,
  stateValue: E1.stateValue, posterior};
