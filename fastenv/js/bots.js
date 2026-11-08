// Baseline bots on the tracker (no Python, no poke-env): ports of poke-env 0.16.1's baselines, used by host.js
// (--bot) as env-side opponents and by porygon.fastenv_eval.
//   heuristic(t)  SimpleHeuristicsPlayer.choose_singles_move (singles; never dynamaxes in gen4/9; never
//                 terastallizes because poke-env never learns its own tera type from the request; never uses
//                 setup moves: poke-env's `move.target == "self"` compares an enum with a string)
//   maxPower(t)   MaxBasePowerPlayer
// Each returns a Showdown choice string ('move <id>', 'switch <name>'), or null where poke-env would pick a random
// valid order (the caller picks one: E.validChoices). Move static data: the encoder tables (accuracy, category,
// expected hits, type) plus Showdown's dex for boosts / target (poke-env's data is the same Showdown source).
'use strict';
const {Dex, toID} = require('./ps');
const {loadGen} = require('./tracker');

const ENTRY_HAZARDS = {spikes: 'SPIKES', stealthrock: 'STEALTH_ROCK', stickyweb: 'STICKY_WEB', toxicspikes: 'TOXIC_SPIKES'};
const ANTI_HAZARDS = new Set(['rapidspin', 'defog']);
const SETUP_BRANCH = false;
const SPEED_TIER = 0.1, HP_FRACTION = 0.4, SWITCH_OUT_THRESHOLD = -2;
const MV_ACC = 1, MV_CAT = 5, MV_HITS = 41;  // encoder_v1 move-feature slots (static part of the table row)

function chartOf(t) { return loadGen(t.gen).chart; }

function dmg(chart, moveType, mon) {  // Pokemon.damage_multiplier(type)
  if (moveType === 'THREE_QUESTION_MARKS' || moveType === 'STELLAR') return 1;
  let m = chart[mon.t1][moveType];
  if (mon.t2 !== null) m *= chart[mon.t2][moveType];
  return m;
}
const types = mon => [mon.t1, mon.t2].filter(x => x !== null);

function matchup(chart, mon, opp) {  // SimpleHeuristicsPlayer._estimate_matchup
  let score = Math.max(...types(mon).map(x => dmg(chart, x, opp)));
  score -= Math.max(...types(opp).map(x => dmg(chart, x, mon)));
  const ms = mon.baseStatsP[5], os = opp.baseStatsP[5];
  if (ms > os) score += SPEED_TIER; else if (os > ms) score -= SPEED_TIER;
  score += mon.hpFraction * HP_FRACTION;
  score -= opp.hpFraction * HP_FRACTION;
  return score;
}

const STAT = {atk: 1, def: 2, spa: 3, spd: 4, spe: 5};
function statEstimation(mon, stat) {
  const b = mon.boosts[stat];
  const boost = b > 1 ? (2 + b) / 2 : 2 / (2 - b);
  return ((2 * mon.baseStatsP[STAT[stat]] + 31) + 5) * boost;
}

function shouldSwitchOut(t, chart) {
  const a = t.activePokemon, o = t.oppActivePokemon;
  if (!t.availableSwitches.some(m => matchup(chart, m, o) > 0)) return false;
  if (a.boosts.def <= -3 || a.boosts.spd <= -3) return true;
  if (a.boosts.atk <= -3 && a.stats.atk >= a.stats.spa) return true;
  if (a.boosts.spa <= -3 && a.stats.atk <= a.stats.spa) return true;
  return matchup(chart, a, o) < SWITCH_OUT_THRESHOLD;
}

const dexCache = new Map();
function dexMove(gen, id) {
  const k = `${gen}:${id}`;
  if (!dexCache.has(k)) {
    const m = Dex.forGen(gen).moves.get(id);
    dexCache.set(k, {boosts: m && m.exists && m.boosts ? m.boosts : null, target: m && m.exists ? m.target : ''});
  }
  return dexCache.get(k);
}

function moveMsg(m) { return m.id === 'recharge' ? 'move 1' : `move ${m.id}`; }

function heuristic(t) {
  const a = t.activePokemon, o = t.oppActivePokemon;
  if (a === null || o === null) return null;
  const chart = chartOf(t);
  const phys = statEstimation(a, 'atk') / statEstimation(o, 'def');
  const spec = statEstimation(a, 'spa') / statEstimation(o, 'spd');
  const avail = t.availableMoves;
  if (avail.length && (!shouldSwitchOut(t, chart) || !t.availableSwitches.length)) {
    let nMine = 0, oppFainted = 0;
    for (const m of t.team.values()) if (!m.fainted) nMine++;
    for (const m of t.oppTeam.values()) if (m.fainted) oppFainted++;
    const nOpp = 6 - oppFainted;
    for (const m of avail) {
      if (nOpp >= 3 && m.id in ENTRY_HAZARDS && !t.oppSideConds.has(ENTRY_HAZARDS[m.id])) return moveMsg(m);
      else if (t.sideConds.size && ANTI_HAZARDS.has(m.id) && nMine >= 2) return moveMsg(m);
    }
    // Setup moves: poke-env compares move.target (a Target enum) with the string "self", which is never equal,
    // so SimpleHeuristicsPlayer never takes this branch; reproduced (SETUP_BRANCH = false) for parity.
    if (SETUP_BRANCH && a.hpFraction === 1 && matchup(chart, a, o) > 0) {
      for (const m of avail) {
        const d = dexMove(t.gen, m.id);
        if (!d.boosts || m.e.tgt !== 'self') continue;
        const vals = Object.entries(d.boosts);
        if (vals.reduce((s, [, v]) => s + v, 0) < 2) continue;
        const pos = vals.filter(([, v]) => v > 0).map(([k]) => a.boosts[k]);
        if (pos.length && Math.min(...pos) < 6) return moveMsg(m);
      }
    }
    let best = null, bestScore = -Infinity;
    const at = types(a);
    for (const m of avail) {
      const s = m.e.s;
      const acc = s ? s[MV_ACC] : 1, hits = s ? s[MV_HITS] * 5 : 1, physical = s ? s[MV_CAT] === 1 : false;
      const ty = m.e.typeName;
      const score = m.basePower * (at.includes(ty) ? 1.5 : 1) * (physical ? phys : spec) * acc * hits
        * dmg(chart, ty, o);
      if (score > bestScore) { best = m; bestScore = score; }
    }
    return moveMsg(best);
  }
  if (t.availableSwitches.length) {
    let best = null, bestScore = -Infinity;
    for (const m of t.availableSwitches) {
      const s = matchup(chart, m, o);
      if (s > bestScore) { best = m; bestScore = s; }
    }
    return `switch ${best.name}`;
  }
  return null;
}

function maxPower(t) {
  let best = null;
  for (const m of t.availableMoves) if (best === null || m.basePower > best.basePower) best = m;
  return best === null ? null : moveMsg(best);
}

module.exports = {heuristic, maxPower};
