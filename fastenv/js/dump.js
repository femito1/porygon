// Full simulator battle state for host.js --dump-states (NNUE gate 1: porygon.nnue.convert_full builds the
// poke-engine State of the true position from it). Plain JSON only: effect states keep their primitive fields
// (Pokemon references become {side, slot}, effects their id).
//
// state = {turn, gen, side (the side to move, 0/1), sides: [p1, p2], field}
//   side  = {n (0 = p1, 1 = p2), active (index into mons, -1 none), mons: [mon x team size, in the team's initial order (= poke-env's
//            battle.team order)], conditions: {id: effect}, slot: {id: effect} (active slot: wish, futuremove, ...),
//            request: {forceSwitch, wait, trapped, maybeTrapped, canTera, disabled: [move id]} | null, totalFainted}
//   mon   = {species, base, level, types, added, tera, terastallized, hp, maxhp, stats {atk, def, spa, spd, spe},
//            item, ability, base_ability, moves: [{id, pp, maxpp, disabled}], base_moves: [id], hp_type, status,
//            status_state, boosts, volatiles: {id: effect}, last_move, active_turns, fainted, active, trapped (incl. by a hidden ability), weightkg,
//            transformed, illusion (species shown | null)}
//   field = {weather, weather_state, terrain, terrain_state, pseudo: {id: effect}}
// A mon's volatiles also carry _start (the turn it most likely started: the first dump that saw it is at the start
// of turn T => T - 1; at a forced switch within turn T => T) and _time0 (its `time` then, e.g. confusion's attempts
// left): elapsed counts as poke-env (turn - _start) / search.convert pass them to the engine.
'use strict';

function ref(x, battle) {
  if (x === null || x === undefined) return null;
  if (typeof x !== 'object') return x;
  if (x.side && x.species && typeof x.position === 'number') {  // a Pokemon
    return {side: x.side.n, slot: x.side.pokemon.indexOf(x), species: x.species.id};
  }
  if (typeof x.id === 'string' && x.effectType) return x.id;  // an effect (move, item, ability, condition)
  return undefined;
}

function effect(es, battle) {
  const out = {};
  for (const [k, v] of Object.entries(es)) {
    if (k === 'effectOrder' || k === 'target') continue;
    if (v === null || typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') out[k] = v;
    else if (Array.isArray(v)) { if (v.every(y => y === null || typeof y !== 'object')) out[k] = v; }
    else { const r = ref(v, battle); if (r !== undefined) out[k] = r; }
  }
  return out;
}

const effects = (obj, battle) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, effect(v, battle)]));
const SEEN = new WeakMap();  // volatile EffectState -> {turn, time} at its first dump
function volatiles(obj, battle) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    let first = SEEN.get(v);
    if (!first) {
      const mid = battle.sides.some(s => s.activeRequest && s.activeRequest.forceSwitch);
      first = {turn: battle.turn - (mid ? 0 : 1), time: v.time};
      SEEN.set(v, first);
    }
    out[k] = {...effect(v, battle), _start: first.turn, _time0: first.time ?? null};
  }
  return out;
}

function statusState(es, battle) {  // + _time0: its `time` at the first dump that saw it (gen4 sleep length)
  let first = SEEN.get(es);
  if (!first) { first = {time: es.time}; SEEN.set(es, first); }
  return {...effect(es, battle), _time0: first.time ?? null};
}

function mon(p, battle) {
  const st = p.storedStats;
  return {
    species: p.species.id, base: p.baseSpecies.id, level: p.level,
    types: p.getTypes(true, true), added: p.addedType || '', tera: p.teraType ? p.teraType.toLowerCase() : '',
    terastallized: p.terastallized ? p.terastallized.toLowerCase() : '',
    hp: p.hp, maxhp: p.maxhp, stats: {atk: st.atk, def: st.def, spa: st.spa, spd: st.spd, spe: st.spe},
    item: p.item || '', ability: p.ability || '', base_ability: p.baseAbility || '',
    moves: p.moveSlots.map(m => ({id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled})),
    base_moves: p.baseMoveSlots.map(m => m.id), hp_type: (p.hpType || '').toLowerCase(),
    status: p.status || '', status_state: statusState(p.statusState, battle), boosts: {...p.boosts},
    volatiles: volatiles(p.volatiles, battle), last_move: p.lastMove ? p.lastMove.id : '', active_turns: p.activeTurns,
    fainted: !!p.fainted, active: !!p.isActive, trapped: !!p.trapped, weightkg: p.weighthg / 10, transformed: !!p.transformed,
    illusion: p.illusion ? p.illusion.species.id : null,
  };
}

function request(side) {
  const r = side.activeRequest;
  if (!r) return null;
  const a = r.active && r.active[0];
  return {
    forceSwitch: !!(r.forceSwitch && r.forceSwitch[0]), wait: !!r.wait,
    trapped: !!(a && a.trapped), maybeTrapped: !!(a && a.maybeTrapped), canTera: !!(a && a.canTerastallize),
    disabled: a && a.moves ? a.moves.filter(m => m.disabled).map(m => m.id) : [],
  };
}

/** Per side, the team's Pokemon objects in their initial order (call right after battle start). */
function initialOrder(battle) {
  return battle.sides.map(s => s.pokemon.slice());
}

function battleState(battle, side, order) {
  const f = battle.field;
  return {
    turn: battle.turn, gen: battle.gen, side,
    sides: battle.sides.map((s, i) => ({
      n: i, active: s.active[0] ? order[i].indexOf(s.active[0]) : -1,
      mons: order[i].map(p => mon(p, battle)),
      conditions: effects(s.sideConditions, battle),
      slot: effects(s.slotConditions[0] || {}, battle),
      request: request(s), totalFainted: s.totalFainted,
    })),
    field: {weather: f.weather || '', weather_state: effect(f.weatherState, battle), terrain: f.terrain || '',
      terrain_state: effect(f.terrainState, battle), pseudo: effects(f.pseudoWeather, battle)},
  };
}

module.exports = {battleState, initialOrder};
