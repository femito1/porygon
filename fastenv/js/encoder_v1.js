'use strict';
// Observation encoder v1: a JS port of porygon.encoder.embed_battle (src/porygon/encoder.py, OBS_DIM 4079), plus
// poke-env 0.16.1's SinglesEnv.get_action_mask / action_to_order(strict=False) and
// PokeEnv.reward_computing_helper's state value, all as pure functions of a SideTracker's state (tracker.js) and the
// static tables exported from Python (scripts/fastenv/export_tables_v1.py -> fastenv/js/tables/).
//
//   const E = require('./encoder_v1');
//   E.OBS_DIM                      4079
//   E.nActions(gen)                SinglesEnv.get_action_space_size(gen): 10 for gen4, 26 for gen9
//   E.encode(t, f32, off)          writes OBS_DIM floats at f32[off..] == embed_battle(battle)
//   E.actionMask(t, u8, off)       writes nActions(gen) 0/1 bytes == get_action_mask(battle)
//   E.actionToChoice(t, a)         Showdown choice == str(action_to_order(a, battle, strict=False)) minus '/choose '
//                                  ('move irontail', 'move dracometeor terastallize', 'switch Moltres', 'move 1' for
//                                  recharge, 'default').  Where poke-env falls back to a RANDOM valid order (action
//                                  not convertible / not in battle.valid_orders) this returns 'default'; where
//                                  poke-env raises IndexError (switch index >= team size) it also returns 'default'.
//   E.validChoices(t)              battle.valid_orders as choice strings
//   E.stateValue(t, {fainted, hp, status, victory})   reward_computing_helper's current_value (number_of_pokemons=6)
//
// Token layout (as encoder.py): [field (95) | 6 own mons (battle.team order) | 6 opponent mons (opponent_team
// order)], mon = [7 categorical ids | 24 movepool ids | 125 numerics | 4 x 44 move features].
//
// Known poke-env behaviours reproduced on purpose (v1 = parity with the trained network; fix in a spec v2):
//  * Weather start turn is reset by every |-weather|...|[upkeep]| line, so weather "elapsed" is ~always 0..1/8.
//  * Stealth Rock / Sticky Web "layers" store the start TURN (non-stackable), so the hazard slot is min(turn,1).
//  * |-heal| / |-sethp| / Illusion-end hp strings without a status clear the mon's status.
//  * Own mons' tera_type is never read from the request (only from |-terastallize|/details 'tera:'), so the own
//    tera one-hot is zero until terastallized; can_tera comes from the request.
//  * Opponent movepool slots use the randbats set prior for the species (hidden info is not leaked; it is a prior).
//  * Hidden Power: own mons' move id is the typed id from the request (e.g. 'hiddenpowerfire', power override from
//    the digits); the opponent's revealed Hidden Power is plain 'hiddenpower' (Normal type).
//  * -enditem compares the raw item name with 'powerherb' (never matches), boosts dict of -copyboost is a copy.
//  * Exceptions poke-env raises in its handler (unknown messages, team-size overflow on Illusion, etc.) are raised
//    as PokeEnvError by the tracker.
//  * Transform overwrites the mon's weight and it is not restored on switch-out (poke-env bug); Illusion end copies
//    the status and then clears it (set_hp without status); a knocked-off/consumed item and an unknown item both
//    encode as UNK; Regenerator adds max_hp/3 on switch-out also for the opponent's percentage HP.
// See tracker.js for the state schema.
//
// Tables: python scripts/fastenv/export_tables_v1.py (re-run if poke-env, vocab.json or the sets files change).
// Parity: python scripts/fastenv/parity_v1.py --record 400 gen9randombattle --seed 21   (streams node -> python,
//         no temp files; exit 0 iff zero mismatches); quick: pytest tests/test_fastenv_parity.py
// Speed (laptop i5-1235U, one P-core): encode ~11 us hot / ~23 us batched over 32 battles interleaved with the
//         simulator; tracker ~4.5 us per message, ~10 us per side-turn (see final report of the ws-fastenv task).

const {loadCommon, loadSets, PokeEnvError} = require('./tracker');

const C = loadCommon();
const D = C.dims;
const OBS_DIM = D.OBS_DIM, FIELD_DIM = D.FIELD_DIM, MON_DIM = D.MON_DIM, MON_NUM_DIM = D.MON_NUM_DIM;
const MOVE_DIM = D.MOVE_DIM, N_CAT = D.N_CAT, MOVEPOOL = D.MOVEPOOL, N_MOVES = D.N_MOVES;
const PAD = 0, UNK = 1;
const NT = C.types.length;
const TYPE_IDX = new Map(C.types.map((t, i) => [t, i]));
const STATUS_IDX = new Map(C.statuses.map((s, i) => [s, i]));
const EFFECT_IDX = new Map(C.effects.map((e, i) => [e, i]));
const WEATHER_IDX = new Map(C.weathers.map((w, i) => [w, i]));
const FIELD_IDX = new Map(C.fields.map((f, i) => [f, i]));
const NW = C.weathers.length, NF = C.fields.length, NS = C.screens.length;
const BOOSTS = C.boosts, STATS = C.stats;
const V = C.vocabMaps;
const HP_MOVE_IDX = V.moves.get('hiddenpower');

// move feature slots (encoder.encode_move order)
const MV_BP = 0, MV_PP = 3, MV_AVAIL = 4, MV_CAT_STATUS = 7, MV_STAB = 8 + NT, MV_LOG = 9 + NT, MV_IMMUNE = 10 + NT;
// mon numerics offsets
const NUM0 = N_CAT + MOVEPOOL;
const N_HP = 0, N_FNT = 1, N_BASE = 2, N_STATS = 8, N_LEVEL = 14, N_TYPES = 15, N_TERA = 15 + NT, N_ISTERA = 15 + 2 * NT;
const N_BOOSTS = N_ISTERA + 1, N_STATUS = N_BOOSTS + 7, N_SCOUNT = N_STATUS + C.statuses.length;
const N_EFF = N_SCOUNT + 1, N_TAIL = N_EFF + C.effects.length;
if (N_TAIL + 10 !== MON_NUM_DIM || MV_IMMUNE + 16 !== MOVE_DIM) throw new Error('encoder_v1 layout mismatch');
const MOVE0 = NUM0 + MON_NUM_DIM;

function stripDigits(s) { let e = s.length; while (e > 0 && s.charCodeAt(e - 1) >= 48 && s.charCodeAt(e - 1) <= 57) e--; return s.slice(0, e); }
/** encoder.idx */
function idx(kind, key) {
  if (key === null || key === undefined) return UNK;
  if (key === '') return PAD;
  const t = V[kind];
  const v = t.get(key);
  if (v !== undefined) return v;
  if (kind === 'moves' && key.startsWith('hiddenpower')) return HP_MOVE_IDX;
  const w = t.get(stripDigits(key));
  return w === undefined ? UNK : w;
}
const elapsed = (start, turn) => Math.min(Math.max(turn - start, 0), 8) / 8;
function estimateStat(base, level, stat) {
  const core = Math.floor((2 * base + 31 + 21) * level / 100);
  return stat === 'hp' ? core + level + 10 : core + 5;
}
function nActions(gen) {
  const g = gen === 6 ? 1 : gen === 7 ? 2 : gen === 8 ? 3 : gen === 9 ? 4 : 0;
  return 6 + 4 * (g + 1);
}

function damageMultiplier(moveType, t1, t2, chart) {  // PokemonType.damage_multiplier
  if (moveType === 'THREE_QUESTION_MARKS' || moveType === 'STELLAR' || t1 === 'THREE_QUESTION_MARKS' || t1 === 'STELLAR') return 1;
  const r1 = chart[t1];
  if (!r1 || !(moveType in r1)) throw new PokeEnvError(`KeyError type chart ${t1} ${moveType}`);
  let m = r1[moveType];
  if (t2 !== null) {
    const r2 = chart[t2];
    if (!r2 || !(moveType in r2)) throw new PokeEnvError(`KeyError type chart ${t2} ${moveType}`);
    m *= r2[moveType];
  }
  return m;
}

function encodeMove(out, o, move, user, defender, chart, avail) {
  const e = move.e;
  if (e.s === null) throw new PokeEnvError(`encode_move failed for ${move.id}`);
  const s = e.s;
  if (e.s32 === undefined) e.s32 = Float32Array.from(s);
  out.set(e.s32, o);
  if (move.bpOverride !== null) out[o + MV_BP] = move.bpOverride / 150;
  out[o + MV_PP] = move.pp / Math.max(move.maxPP, 1);
  let a = 0;
  for (let i = 0; i < avail.length; i++) if (avail[i] === move.id) { a = 1; break; }
  out[o + MV_AVAIL] = a;
  const mt = e.typeName;
  out[o + MV_STAB] = (user.t1 === mt || user.t2 === mt) ? 1 : 0;
  let mult = 1;
  if (defender !== null && s[MV_CAT_STATUS] !== 1) mult = damageMultiplier(mt, defender.t1, defender.t2, chart);
  out[o + MV_LOG] = mult > 0 ? Math.log2(mult) / 2 : -1.5;
  out[o + MV_IMMUNE] = mult === 0 ? 1 : 0;
}

const EMPTY = [];
const moveIdx = m => (m.vidx >= 0 ? m.vidx : (m.vidx = idx('moves', m.id)));
// ctx: {act, oppAct, avail (own available move ids)} computed once per encode()
function encodeMon(out, o, mon, t, mine, sets, ctx) {
  if (mon === null) { out[o + NUM0 + MON_NUM_DIM - 1] = 1; return; }
  const G = t.G;
  const defender = mine ? ctx.oppAct : ctx.act;
  const mm = mon.movesMap();
  const avail = mine && mon.active ? ctx.avail : EMPTY;
  let ability = mon.abilityP;
  if (ability === null && mon.possibleAbilities.length === 1) ability = mon.possibleAbilities[0];
  const item = mon.item === C.unknownItem ? null : mon.item;
  const species = V.species.has(mon.species) ? mon.species : mon.baseSpecies;
  out[o] = idx('species', species);
  out[o + 1] = idx('items', item);
  out[o + 2] = idx('abilities', ability);
  // first 4 moves (categorical + features) and own movepool
  let i = 0;
  for (const m of mm.values()) {
    if (i < N_MOVES) {
      out[o + 3 + i] = moveIdx(m);
      encodeMove(out, o + MOVE0 + i * MOVE_DIM, m, mon, defender, G.chart, avail);
    } else if (!mine) break;
    if (mine && i < MOVEPOOL) out[o + N_CAT + i] = moveIdx(m);
    i++;
  }
  if (!mine) {
    const pool = sets.get(species);
    if (pool) for (let j = 0; j < MOVEPOOL; j++) out[o + N_CAT + j] = pool[j];
  }
  // numerics
  const n = o + NUM0;
  const level = mon.level;
  out[n + N_HP] = mon.hpFraction;
  out[n + N_FNT] = mon.fainted ? 1 : 0;
  const bs = mon.baseStatsP;
  for (let s = 0; s < 6; s++) {
    out[n + N_BASE + s] = bs[s] / 255;
    const v = mon.stats[STATS[s]];
    out[n + N_STATS + s] = (v ? v : estimateStat(bs[s], level, STATS[s])) / 500;
  }
  out[n + N_LEVEL] = level / 100;
  const t1 = mon.t1, t2 = mon.t2;
  let ti = TYPE_IDX.get(t1);
  if (ti !== undefined) out[n + N_TYPES + ti] = 1;
  if (t2 !== null && t2 !== t1) { ti = TYPE_IDX.get(t2); if (ti !== undefined) out[n + N_TYPES + ti] = 1; }
  if (mon.teraType !== null) { ti = TYPE_IDX.get(mon.teraType); if (ti !== undefined) out[n + N_TERA + ti] = 1; }
  out[n + N_ISTERA] = mon.tera ? 1 : 0;
  for (let b = 0; b < 7; b++) out[n + N_BOOSTS + b] = (mon.boosts[BOOSTS[b]] || 0) / 6;
  if (mon.status !== null) { const si = STATUS_IDX.get(mon.status); if (si !== undefined) out[n + N_STATUS + si] = 1; }
  out[n + N_SCOUNT] = Math.min(mon.statusCounter, 10) / 10;
  for (const e of mon.effects.keys()) { const ei = EFFECT_IDX.get(e); if (ei !== undefined) out[n + N_EFF + ei] = 1; }
  const tl = n + N_TAIL;
  out[tl] = mon.active ? 1 : 0;
  out[tl + 1] = mine ? 1 : 0;
  out[tl + 2] = mon.activeTurns === 1 ? 1 : 0;
  out[tl + 3] = mon.protectCounter / 3;
  out[tl + 4] = mon.mustRecharge ? 1 : 0;
  out[tl + 5] = mon.preparing ? 1 : 0;
  out[tl + 6] = Math.log10(Math.max(mon.weight, 0.1)) / 3;
  out[tl + 7] = (mine && mon.active && t.trapped) ? 1 : 0;
  out[tl + 8] = ctx.oppAct === mon ? 1 : 0;
  out[tl + 9] = 0;
}

function encodeSide(out, o, conds, turn) {
  for (let i = 0; i < NS; i++) {
    const v = conds.get(C.screens[i]);
    if (v !== undefined) { out[o + i] = 1; out[o + NS + i] = elapsed(v, turn); }
  }
  for (let i = 0; i < C.hazards.length; i++) {
    const [name, mx] = C.hazards[i];
    out[o + 2 * NS + i] = Math.min(conds.get(name) || 0, mx) / mx;
  }
  return o + 2 * NS + C.hazards.length;
}

function encodeField(out, o, t, ctx) {
  const turn = t.turn;
  for (const [w, st] of t.weather) {
    const wi = WEATHER_IDX.get(w);
    if (wi !== undefined) { out[o + wi] = 1; out[o + NW + wi] = elapsed(st, turn); }
  }
  out[o + 2 * NW] = t.weather.size === 0 ? 1 : 0;
  let p = o + 2 * NW + 1;
  for (const [f, st] of t.fields) {
    const fi = FIELD_IDX.get(f);
    if (fi !== undefined) { out[p + fi] = 1; out[p + NF + fi] = elapsed(st, turn); }
  }
  p += 2 * NF;
  p = encodeSide(out, p, t.sideConds, turn);
  p = encodeSide(out, p, t.oppSideConds, turn);
  let alive = 0, oppFainted = 0;
  for (const m of t.team.values()) if (!m.fainted) alive++;
  for (const m of t.oppTeam.values()) if (m.fainted) oppFainted++;
  out[p++] = Math.min(turn, 100) / 100;
  out[p++] = (6 - t.oppTeam.size) / 6;
  out[p++] = alive / 6;
  out[p++] = (6 - oppFainted) / 6;
  out[p++] = t.forceSwitch ? 1 : 0;
  out[p++] = t.trapped ? 1 : 0;
  out[p++] = t.maybeTrapped ? 1 : 0;
  out[p++] = t.canTera ? 1 : 0;
  out[p++] = t.usedTera ? 1 : 0;
  out[p++] = t.oppUsedTera ? 1 : 0;
  out[p++] = t.wait ? 1 : 0;
  out[p++] = ctx.act === null ? 1 : 0;
  out[p++] = t.availableSwitches.length / 5;
  out[p++] = t.availableMoves.length / 4;
  if (p - o !== FIELD_DIM) throw new Error('encoder_v1 layout mismatch (field)');
}

/** embed_battle(battle) -> out[off .. off+OBS_DIM) */
const CTX = {act: null, oppAct: null, avail: []};
function encode(t, out, off = 0) {
  out.fill(0, off, off + OBS_DIM);
  const ctx = CTX;
  ctx.act = t.activePokemon; ctx.oppAct = t.oppActivePokemon;
  ctx.avail.length = 0;
  for (const m of t.availableMoves) ctx.avail.push(m.id);
  encodeField(out, off, t, ctx);
  const sets = loadSets(t.format || '');
  let o = off + FIELD_DIM, i = 0;
  for (const m of t.team.values()) { if (i++ >= 6) break; encodeMon(out, o, m, t, true, sets, ctx); o += MON_DIM; }
  for (; i < 6; i++) { encodeMon(out, o, null, t, true, sets, ctx); o += MON_DIM; }
  i = 0;
  for (const m of t.oppTeam.values()) { if (i++ >= 6) break; encodeMon(out, o, m, t, false, sets, ctx); o += MON_DIM; }
  for (; i < 6; i++) { encodeMon(out, o, null, t, false, sets, ctx); o += MON_DIM; }
  return out;
}

const SPECIAL = C.specialSet;
function knownMoves(a) { const r = []; for (const m of a.movesMap().values()) { if (r.length >= 4) break; r.push(m); } return r; }

/** SinglesEnv.get_action_mask (gen4/gen9: no mega / z / dynamax in random battles; they are still honoured). */
function actionMask(t, out, off = 0) {
  const n = nActions(t.gen);
  out.fill(0, off, off + n);
  if (t.wait) { out[off] = 1; return out; }
  if (!t.trapped) {
    let i = 0;
    for (const p of t.team.values()) {
      const bs = p.baseSpecies;
      for (const s of t.availableSwitches) if (s.baseSpecies === bs) { if (i < n) out[off + i] = 1; break; }
      i++;
    }
  }
  const a = t.activePokemon;
  if (a === null) return out;
  const known = knownMoves(a);
  const am = t.availableMoves;
  const ms = [];
  for (let i = 0; i < known.length; i++) {
    for (const m of am) if (m.id === known[i].id) { ms.push(i + 6); break; }
  }
  if (!ms.length && am.length === 1 && !SPECIAL.has(am[0].id)) ms.push(6);
  const set = x => { if (x < n) out[off + x] = 1; };
  if (t.canMega) for (const x of ms) set(x + 4);
  if (t.canZ) throw new PokeEnvError('z-moves not supported in v1');
  if (t.canDynamax) for (const x of ms) set(x + 12);
  if (t.canTera) for (const x of ms) set(x + 16);
  if (!ms.length && am.length === 1 && SPECIAL.has(am[0].id)) ms.push(6);
  for (const x of ms) set(x);
  return out;
}

function moveMsg(m, gimmick) {
  if (m.id === 'recharge') return 'move 1';
  return gimmick ? `move ${m.id} ${gimmick}` : `move ${m.id}`;
}
/** Battle.valid_orders as choice strings */
function validChoices(t) {
  if (t.wait) return ['default'];
  const out = [];
  if (!t.trapped) for (const p of t.availableSwitches) out.push(`switch ${p.name}`);
  if (t.activePokemon === null || t.forceSwitch) return out;
  for (const m of t.availableMoves) out.push(moveMsg(m, ''));
  if (t.canMega) for (const m of t.availableMoves) out.push(moveMsg(m, 'mega'));
  if (t.canZ) throw new PokeEnvError('z-moves not supported in v1');
  if (t.canDynamax) for (const m of t.availableMoves) out.push(moveMsg(m, 'dynamax'));
  if (t.canTera) for (const m of t.availableMoves) out.push(moveMsg(m, 'terastallize'));
  return out;
}

/** SinglesEnv.action_to_order(action, battle, strict=False) -> choice ('default' where poke-env goes random). */
function actionToChoice(t, action) {
  if (action === -2) return 'default';
  if (action === -1) return 'forfeit';
  let order;
  if (action < 6) {
    let i = 0, mon = null;
    for (const p of t.team.values()) { if (i++ === action) { mon = p; break; } }
    if (mon === null) return 'default';  // poke-env: IndexError (never a masked-legal action)
    order = `switch ${mon.name}`;
  } else {
    const a = t.activePokemon;
    if (a === null) return 'default';
    const known = knownMoves(a);
    const am = t.availableMoves;
    let useAvail = false;
    if (am.length === 1) { useAvail = true; for (const k of known) if (k.id === am[0].id) { useAvail = false; break; } }
    const mvs = useAvail ? am : known;
    const j = (action - 6) % 4;
    if (j >= mvs.length) return 'default';
    const g = action >= 10 && action < 14 ? 'mega' : action >= 14 && action < 18 ? 'zmove'
      : action >= 18 && action < 22 ? 'dynamax' : action >= 22 && action < 26 ? 'terastallize' : '';
    order = moveMsg(mvs[j], g);
  }
  return validChoices(t).includes(order) ? order : 'default';
}

/** reward_computing_helper's current_value */
function stateValue(t, {fainted = 0, hp = 0, status = 0, victory = 1} = {}) {
  let v = 0;
  for (const m of t.team.values()) {
    v += m.hpFraction * hp;
    if (m.fainted) v -= fainted; else if (m.status !== null) v -= status;
  }
  v += (6 - t.team.size) * hp;
  for (const m of t.oppTeam.values()) {
    v -= m.hpFraction * hp;
    if (m.fainted) v += fainted; else if (m.status !== null) v += status;
  }
  v -= (6 - t.oppTeam.size) * hp;
  if (t.won === true) v += victory; else if (t.won === false) v -= victory;
  return v;
}

module.exports = {OBS_DIM, FIELD_DIM, MON_DIM, MOVE_DIM, nActions, encode, actionMask, actionToChoice, validChoices,
  stateValue, idx};
