'use strict';
// First-person battle-state tracker: a line-by-line port of poke-env 0.16.1's message handling (Player.
// _handle_battle_message -> AbstractBattle.parse_message / Battle.parse_request, Pokemon, Move, MoveSet) for
// singles, restricted to what porygon's encoder (encoder_v1.js == porygon.encoder.embed_battle), the action mask /
// action->choice mapping (poke-env SinglesEnv) and the reward state value read.  poke-env quirks and bugs are
// reproduced on purpose (parity with the trained network); they are listed in encoder_v1.js's header.
//
//   const {SideTracker} = require('./tracker');
//   const t = new SideTracker({format: 'gen9randombattle', side: 0});   // side 0 = p1, 1 = p2
//   t.onLog(text)       every SimBattle onLog chunk for this side, in order (lines joined by \n)
//   t.onRequest(req)    every request object for this side (Showdown's side.activeRequest; not mutated)
//   t.onError(text)     '|error|...' lines; '[Invalid choice]' re-opens the decision (poke-env re-asks the agent)
//   t.decisionPending   true when poke-env would call choose_move now; cleared by t.markChosen() (or the next
//                       onLog/onRequest)
//   t.teamIdents() / t.opponentIdents()   poke-env dict keys ('p1: Name', position stripped) in battle.team /
//                       battle.opponent_team order (= encoder token order)
//   t.won               true / false / null (poke-env battle.won), t.finished
//
// A poke-env exception (KeyError, AssertionError, NotImplementedError, ... raised inside its message handler) is
// raised here as PokeEnvError at the same message; poke-env's battle would be stuck/broken at that point too.
//
// ---------------------------------------------------------------------------------------------------------------
// STATE SCHEMA (plain data; every name below is the poke-env attribute it mirrors).  Enum values are poke-env enum
// member NAMES (strings: 'FIRE', 'TOX', 'SUBSTITUTE', 'SUNNYDAY', 'ELECTRIC_TERRAIN', 'STEALTH_ROCK', ...).
// SideTracker (AbstractBattle / Battle):
//   gen, format (from |tier|, e.g. 'gen9randombattle'), username ('p1'/'p2'), role ('p1'/'p2'), turn
//   team: Map ident -> Mon       own team, poke-env insertion order (first request = Showdown team order)
//   oppTeam: Map ident -> Mon    opponent mons in order of first appearance
//   teamSize {p1, p2}
//   weather: Map name -> start turn (at most one; reset on every |-weather| incl. [upkeep])
//   fields: Map name -> start turn;  sideConds / oppSideConds: Map name -> start turn (layers for SPIKES /
//   TOXIC_SPIKES)
//   request-derived: availableMoves [Move], availableSwitches [Mon], canTera, canMega, canZ, canDynamax,
//   forceSwitch, trapped, maybeTrapped, wait, teampreview, reviving, lastRequest
//   usedTera, oppUsedTera, won (true/false/null), finished
// Mon (Pokemon):
//   key (dict key), name, species (stored species id), baseStats [hp,atk,def,spa,spd,spe] (current forme),
//   type1, type2 (current forme), weight, possibleAbilities [ids], ability_ (_ability), tempAbility,
//   formeAbility, item ('unknown_item' | id | '' | null), level, active, hp (_current_hp, may be null),
//   maxHp, status (null | 'BRN'..'TOX' | 'FNT'), statusCounter, boosts {accuracy,atk,def,evasion,spa,spd,spe},
//   effects: Map name -> counter, activeTurns (first_turn == activeTurns === 1), tera (is_terastallized),
//   teraType, mustRecharge, prepMove, prepTarget, protectCounter, revealed, stats {hp..spe} (own: from request),
//   tempBaseStats, tempTypes [], moves: MoveSet, lastRequest, lastDetails, dancing
// MoveSet: base: Map key -> Move, mimic: Move|null, transform: MoveSet|null (moves() = poke-env Pokemon.moves)
// Move: id, bpOverride (Hidden Power power from request ids), pp (current), fromMimic, fromTransform, e (static
//   table row from tables/gen{N}_v1.json or the special recharge/fight entry)
// ---------------------------------------------------------------------------------------------------------------

const fs = require('fs');
const path = require('path');

const TABLE_DIR = path.join(__dirname, 'tables');
let COMMON = null;
const GEN_TABLES = {}, SETS = {};

function loadCommon() {
  if (COMMON) return COMMON;
  const c = JSON.parse(fs.readFileSync(path.join(TABLE_DIR, 'common_v1.json'), 'utf8'));
  c.allEffectsSet = new Set(c.allEffects);
  c.effTurnCountableSet = new Set(c.effTurnCountable);
  c.effActionCountableSet = new Set(c.effActionCountable);
  c.effEndsOnTurnSet = new Set(c.effEndsOnTurn);
  c.effBreaksProtectSet = new Set(c.effBreaksProtect);
  c.allTypesSet = new Set(c.allTypes);
  c.allStatusesSet = new Set(c.allStatuses);
  c.protectCounterSet = new Set(c.protectCounterMoves);
  c.specialSet = new Set(c.specialMoves);
  c.vocabMaps = {};
  for (const k of Object.keys(c.vocab)) c.vocabMaps[k] = new Map(Object.entries(c.vocab[k]));
  COMMON = c;
  return c;
}

function loadGen(gen) {
  if (GEN_TABLES[gen]) return GEN_TABLES[gen];
  const t = JSON.parse(fs.readFileSync(path.join(TABLE_DIR, `gen${gen}_v1.json`), 'utf8'));
  const dex = new Map();
  for (const [k, v] of Object.entries(t.dex)) {
    dex.set(k, {bs: v[0], types: v[1], abilities: v[2], w: v[3], baseSpecies: v[4], fca: v[5], ab0: v[6]});
  }
  const moves = new Map();
  for (const [k, v] of Object.entries(t.moves)) {
    v.typeName = typeFromName(v.type);
    v.s = v.s ? Float64Array.from(v.s) : null;
    moves.set(k, v);
  }
  GEN_TABLES[gen] = {gen, dex, moves, chart: t.chart};
  return GEN_TABLES[gen];
}

function loadSets(format) {
  if (format in SETS) return SETS[format];
  const p = path.join(TABLE_DIR, `${format}_sets_v1.json`);
  SETS[format] = fs.existsSync(p) ? new Map(Object.entries(JSON.parse(fs.readFileSync(p, 'utf8')))) : new Map();
  return SETS[format];
}

class PokeEnvError extends Error {}
function fail(msg) { throw new PokeEnvError(msg); }

// ---------------------------------------------------------------- string helpers (Python semantics)
const ID_CACHE = new Map();
const ALNUM_RE = /[\p{L}\p{N}]/u;
/** poke_env.data.to_id_str: keep str.isalnum() chars, lower(). */
function toId(s) {
  let r = ID_CACHE.get(s);
  if (r !== undefined) return r;
  r = '';
  for (const ch of s) {
    const c = ch.charCodeAt(0);
    if ((c >= 48 && c <= 57) || (c >= 97 && c <= 122)) r += ch;
    else if (c >= 65 && c <= 90) r += String.fromCharCode(c + 32);
    else if (c > 127 && ALNUM_RE.test(ch)) r += ch.toLowerCase();
  }
  if (ID_CACHE.size > 50000) ID_CACHE.clear();
  ID_CACHE.set(s, r);
  return r;
}
const rep = (s, a, b) => (s.indexOf(a) < 0 ? s : s.split(a).join(b));  // Python str.replace (all occurrences)
const last = a => a[a.length - 1];
const lastSplit = (s, sep) => { const i = s.lastIndexOf(sep); return i < 0 ? s : s.slice(i + sep.length); };
const pyStrip = s => s.trim();

/** Move.retrieve_id */
function retrieveId(name) {
  const m = toId(name);
  if (m.startsWith('return')) return 'return';
  if (m.startsWith('frustration')) return 'frustration';
  if (m.startsWith('hiddenpower')) return 'hiddenpower';
  return m;
}
function typeFromName(name) {  // PokemonType.from_name
  if (name === '???') return 'THREE_QUESTION_MARKS';
  const n = name.toUpperCase();
  if (!loadCommon().allTypesSet.has(n)) fail(`KeyError PokemonType ${name}`);
  return n;
}
function statusFromName(s) {  // Status[s.upper()]
  const n = s.toUpperCase();
  if (!loadCommon().allStatusesSet.has(n)) fail(`KeyError Status ${s}`);
  return n;
}
function effectFromMsg(msg) {  // Effect.from_showdown_message
  let m = rep(rep(rep(msg, 'item: ', ''), 'move: ', ''), 'ability: ', '');
  m = rep(rep(m, ' ', '_'), '-', '_').toUpperCase();
  if (m === 'FALLENUNDEFINED') m = 'FALLEN';
  return loadCommon().allEffectsSet.has(m) ? m : 'UNKNOWN';
}
function fieldFromMsg(msg) {  // Field.from_showdown_message
  let m = rep(rep(rep(msg, 'ability: ', ''), 'move: ', ''), ' ', '_');
  if (m.endsWith('terrain') && !m.endsWith('_terrain')) m = rep(m, 'terrain', '_terrain');
  return loadCommon().fieldMembers[m.toUpperCase()] || 'UNKNOWN';
}
function weatherFromMsg(msg) {  // Weather.from_showdown_message (SNOW is an alias of SNOWSCAPE)
  const m = rep(rep(rep(msg, 'move: ', ''), ' ', '_'), '-', '_').toUpperCase();
  return loadCommon().weatherMembers[m] || 'UNKNOWN';
}
function sideFromMsg(msg) {  // SideCondition.from_showdown_message
  const m = rep(rep(rep(msg, 'move: ', ''), ' ', '_'), '-', '_').toUpperCase();
  return loadCommon().sideMembers[m] || 'UNKNOWN';
}
const isTerrain = f => f.endsWith('_TERRAIN');

// ---------------------------------------------------------------- Move / MoveSet
const SPECIAL_ENTRY = {pp: 1, type: 'normal', typeName: 'NORMAL', bp: 0, tgt: undefined, mp: 0, st: 0, err: 1, s: null};

class Move {
  constructor(moveId, G, rawId = null, fromMimic = false, fromTransform = false) {
    this.id = moveId;
    this.bpOverride = null;
    this.fromMimic = fromMimic;
    this.fromTransform = fromTransform;
    this.gen = G.gen;
    this.vidx = -1;  // encoder cache: vocab index of this.id
    if (moveId.startsWith('hiddenpower') && rawId !== null) {
      let digits = '', id = '';
      for (const ch of rawId) if (ch >= '0' && ch <= '9') digits += ch;
      for (const ch of toId(rawId)) if (!(ch >= '0' && ch <= '9')) id += ch;
      this.id = id;
      if (digits) this.bpOverride = parseInt(digits, 10);
    }
    this.e = moveEntry(this.id, G);
    this.pp = fromTransform ? Math.min(5, this.maxPP) : this.maxPP;
  }
  get maxPP() {
    let m = Math.floor(this.e.pp * 8 / 5);
    if (this.gen >= 5 && this.fromTransform) return Math.min(5, m);
    if (this.gen < 3) m = Math.min(m, 61);
    return m;
  }
  get basePower() { return this.bpOverride !== null ? this.bpOverride : this.e.bp; }
  use(pressure = false, overridden = false) {
    let d = 1;
    if (pressure) d += 1;
    if (overridden) d -= 1;
    this.pp = Math.max(this.pp - d, 0);
  }
}
function moveEntry(id, G) {  // Move.entry
  let e = G.moves.get(id);
  if (e) return e;
  if (id.startsWith('z') && (e = G.moves.get(id.slice(1)))) return e;
  if (id === 'recharge' || id === 'fight') return SPECIAL_ENTRY;
  fail(`ValueError Unknown move: ${id}`);
}
function shouldBeStored(id, G) {
  const e = G.moves.get(id);
  return !!(e && e.st);
}

class MoveSet {
  constructor(base) { this.base = base; this.mimic = null; this.transform = null; }
  resolved() { let s = this; while (s.transform !== null) s = s.transform; return s; }
  get baseMoves() { return this.resolved().base; }
  get mimicMove() { return this.resolved().mimic; }
  set mimicMove(m) { this.resolved().mimic = m; }
  /** Pokemon.moves: base moves, with the 'mimic' slot replaced by the mimicked move. */
  moves() {
    const r = this.resolved();
    if (r.mimic === null) return r.base;
    const out = new Map();
    for (const [k, v] of r.base) {
      if (k === 'mimic') out.set(r.mimic.id, r.mimic); else out.set(k, v);
    }
    return out;
  }
}

// ---------------------------------------------------------------- Mon (poke-env Pokemon)
class Mon {
  constructor(G, {species = null, name = null, request = null, details = null} = {}) {
    this.G = G;
    this.gen = G.gen;
    this.key = null;
    this.species = '';
    this.baseStats = null; this.type1 = null; this.type2 = null; this.weight = 0;
    this.possibleAbilities = [];
    this.ability_ = null; this.tempAbility = null; this.formeAbility = null;
    this.gender = null; this.shiny = false;
    this.level = 100;
    this.maxHp = 0; this.hp = 0;
    this.moves = new MoveSet(new Map());
    this.name = null;
    this.active = false;
    this.boosts = {accuracy: 0, atk: 0, def: 0, evasion: 0, spa: 0, spd: 0, spe: 0};
    this.effects = new Map();
    this.activeTurns = 0;
    this.tera = false; this.teraType = null;
    this.item = loadCommon().unknownItem;
    this.lastRequest = {};
    this.lastDetails = '';
    this.mustRecharge = false;
    this.prepMove = null; this.prepTarget = null;
    this.protectCounter = 0;
    this.revealed = false;
    this.stats = {hp: null, atk: null, def: null, spa: null, spd: null, spe: null};
    this.status = null; this.statusCounter = 0;
    this.tempBaseStats = null; this.tempTypes = [];
    this.dancing = false;
    if (request) this.updateFromRequest(request);
    else if (details) this.updateFromDetails(details);
    else if (species) this.updateFromPokedex(species);
    if (name !== null) this.name = name;
  }

  // ---- properties
  get abilityP() {  // Pokemon.ability
    if (this.tempAbility !== null) return this.tempAbility;
    if (this.formeAbility !== null) return this.formeAbility;
    return this.ability_;
  }
  setAbility(a) {  // Pokemon.ability setter
    if (this.ability_ === null) this.ability_ = toId(a); else this.tempAbility = toId(a);
  }
  get baseSpecies() {
    const e = this.G.dex.get(this.species);
    if (!e) fail(`KeyError pokedex ${this.species}`);
    return e.baseSpecies;
  }
  get fainted() { return this.status === 'FNT'; }
  get curHp() { return this.hp || 0; }
  get maxHpP() { return this.maxHp || 0; }
  get hpFraction() { return this.curHp ? this.curHp / this.maxHpP : 0; }
  get transformed() { return this.moves.resolved() !== this.moves; }
  get isDynamaxed() { return this.effects.has('DYNAMAX'); }
  get preparing() { return !!this.prepTarget || !!this.prepMove; }
  get t1() {
    if (this.tera && this.teraType !== null) return this.teraType;
    if (this.tempTypes.length > 0) return this.tempTypes[0];
    return this.type1;
  }
  get t2() {
    if (this.tera || this.tempTypes.length === 1) return null;
    if (this.tempTypes.length > 1) return this.tempTypes[1];
    return this.type2;
  }
  types() { const t2 = this.t2; return t2 === null ? [this.t1] : [this.t1, t2]; }
  get baseStatsP() { return this.tempBaseStats !== null ? this.tempBaseStats : this.baseStats; }
  movesMap() { return this.moves.moves(); }
  identifiesAs(ident) {
    const bs = this.baseSpecies;
    if (bs === toId(ident)) return true;
    for (const sub of ident.split('-')) if (toId(sub) === bs) return true;
    return false;
  }

  // ---- methods
  addMove(moveId) {
    const id = retrieveId(moveId);
    const mv = this.movesMap();
    if (mv.has(id)) return mv.get(id);
    if (!shouldBeStored(id, this.G)) return null;
    const base = this.moves.baseMoves;
    if (!base.has(id)) base.set(id, new Move(id, this.G, moveId, false, this.transformed));
    return base.get(id);
  }
  boost(stat, amount) {
    if (!(stat in this.boosts)) fail(`KeyError boost ${stat}`);
    let v = this.boosts[stat] + amount;
    if (v > 6) v = 6; else if (v < -6) v = -6;
    this.boosts[stat] = v;
  }
  cantMove() {
    this.dancing = false;
    this.protectCounter = 0;
    if (this.status === 'SLP') this.statusCounter += 1;
  }
  clearBoosts() { for (const k in this.boosts) this.boosts[k] = 0; }
  clearEffects() { for (const e of [...this.effects.keys()]) this.endEffect(e); }
  cureStatus(status = null) {
    if (status && statusFromName(status) === this.status) { this.status = null; this.statusCounter = 0; }
    else if (status === null && !this.fainted) this.status = null;
  }
  endEffect(effectStr) {
    const e = effectFromMsg(effectStr);
    this.effects.delete(e);
    if (e === 'TYPECHANGE') this.tempTypes = [];
    else if (e === 'SKILL_SWAP') this.tempAbility = null;
    else if (e === 'QUARK_DRIVE') {
      for (const x of ['QUARKDRIVEATK', 'QUARKDRIVEDEF', 'QUARKDRIVESPA', 'QUARKDRIVESPD', 'QUARKDRIVESPE']) this.effects.delete(x);
    } else if (e === 'PROTOSYNTHESIS') {
      for (const x of ['PROTOSYNTHESISATK', 'PROTOSYNTHESISDEF', 'PROTOSYNTHESISSPA', 'PROTOSYNTHESISSPD', 'PROTOSYNTHESISSPE']) this.effects.delete(x);
    }
  }
  endItem(item) {
    this.item = null;
    if (item === 'powerherb') { this.prepMove = null; this.prepTarget = false; }
  }
  endTurn() {
    const C = loadCommon();
    this.activeTurns += 1;
    if (this.status === 'TOX') this.statusCounter += 1;
    for (const e of [...this.effects.keys()]) {
      if (C.effTurnCountableSet.has(e)) {
        if (!this.effects.has(e)) fail(`KeyError effect ${e}`);
        this.effects.set(e, this.effects.get(e) + 1);
      }
      if (C.effEndsOnTurnSet.has(e)) this.endEffect(e);
    }
  }
  faint() {
    this.hp = 0;
    this.status = 'FNT';
    this.tempAbility = null;
    this.tempBaseStats = null;
    this.moves.transform = null;
    this.moves.mimicMove = null;
    this.clearEffects();
  }
  formeChange(species) { this.updateFromPokedex(species.split(',')[0], false); }
  moved(moveId, failed = false, use = true, reveal = true, pressure = false) {
    this.mustRecharge = false;
    this.prepMove = null; this.prepTarget = null;
    let move = null;
    if (reveal) move = this.addMove(moveId);
    if (use && move !== null) move.use(pressure);
    if (move !== null && loadCommon().protectCounterSet.has(move.id) && !failed) this.protectCounter += 1;
    else this.protectCounter = 0;
    if (this.status === 'SLP') this.statusCounter += 1;
    if (this.effects.has('GLAIVE_RUSH')) this.endEffect('Glaive Rush');
    else if (this.effects.has('CHARGE') && move !== null && move.basePower > 0 && move.e.typeName === 'ELECTRIC' && use) this.endEffect('Charge');
    else if (this.effects.has('FLASH_FIRE') && move !== null && move.basePower > 0 && move.e.typeName === 'FIRE' && use) this.endEffect('Flash Fire');
  }
  prepare(moveId, target) {
    const id = retrieveId(moveId);
    const mv = this.movesMap();
    this.prepMove = mv.has(id) ? mv.get(id) : new Move(id, this.G, moveId);
    this.prepTarget = target;
  }
  setBoost(stat, amount) {
    if (Math.abs(amount) > 6) fail(`AssertionError set_boost ${stat} ${amount}`);
    this.boosts[stat] = amount;
  }
  setHpStatus(hpStatus, store = false) {
    if (hpStatus === '0 fnt') { this.hp = 0; this.status = 'FNT'; return; }
    let hp;
    if (hpStatus.indexOf(' ') >= 0) {
      const parts = hpStatus.split(' ');
      if (parts.length !== 2) fail(`ValueError hp_status ${hpStatus}`);
      hp = parts[0];
      this.status = statusFromName(parts[1]);
      if (this.status === 'SLP' && this.effects.has('YAWN')) this.endEffect('yawn');
    } else {
      hp = hpStatus;
      this.status = null;
    }
    // keep only digits and '/', then split on '/' into exactly two non-empty ints (Python semantics)
    let a = 0, b = 0, nA = 0, nB = 0, slashes = 0;
    for (let i = 0; i < hp.length; i++) {
      const c = hp.charCodeAt(i);
      if (c >= 48 && c <= 57) { if (slashes === 0) { a = a * 10 + c - 48; nA++; } else { b = b * 10 + c - 48; nB++; } }
      else if (c === 47) slashes++;
    }
    if (slashes !== 1 || !nA || !nB) fail(`ValueError hp ${hpStatus}`);
    this.hp = a;
    this.maxHp = b;
    if (store) this.stats.hp = this.maxHp;
  }
  startEffect(effectStr, details = null) {
    const C = loadCommon();
    const e = effectFromMsg(effectStr);
    if (!this.effects.has(e)) this.effects.set(e, 0);
    else if (C.effActionCountableSet.has(e)) this.effects.set(e, this.effects.get(e) + 1);
    if (C.effBreaksProtectSet.has(e)) this.protectCounter = 0;
    if (e === 'TYPECHANGE' && details !== null) {
      this.tempTypes = details.split('/').map(typeFromName);
    } else if (e === 'SKILL_SWAP' && details && details.length) {
      if (this.abilityP === null) this.setAbility(details[1]);
      this.tempAbility = toId(details[0]);
    }
  }
  switchIn(details = null) {
    this.active = true;
    if (details) this.updateFromDetails(details);
    this.revealed = true;
  }
  switchOut(fields) {
    if (this.abilityP === 'regenerator' && (this.item === 'abilityshield' || !fields.has('NEUTRALIZING_GAS'))
        && this.status !== 'FNT') {
      this.hp = Math.min(Math.trunc(this.curHp + this.maxHpP / 3), this.maxHpP);
    }
    this.active = false;
    this.clearBoosts();
    this.clearEffects();
    this.activeTurns = 0;
    this.mustRecharge = false;
    this.prepMove = null; this.prepTarget = null;
    this.protectCounter = 0;
    this.tempAbility = null;
    this.tempBaseStats = null;
    this.moves.transform = null;
    this.moves.mimicMove = null;
    this.tempTypes = [];
    if (this.status === 'TOX') this.statusCounter = 0;
  }
  terastallize(type) {
    this.teraType = typeFromName(type);
    this.tera = true;
    this.tempTypes = [];
  }
  transform(into) {
    const e = this.G.dex.get(into.species);
    if (!e) fail(`KeyError pokedex ${into.species}`);
    this.weight = e.w;
    this.tempBaseStats = e.bs;
    const ab = into.abilityP;
    if (ab !== null) this.tempAbility = ab;
    this.tempTypes = e.types.map(typeFromName);
    const tm = new Map();
    for (const m of into.movesMap().values()) tm.set(m.id, new Move(m.id, this.G, null, false, true));
    this.moves.transform = new MoveSet(tm);
    this.boosts = {...into.boosts};
  }
  updateFromPokedex(species, store = true) {
    species = toId(species);
    const e = this.G.dex.get(species);
    if (!e) fail(`KeyError pokedex ${species}`);
    if (store) this.species = species;
    this.baseStats = e.bs;
    this.type1 = typeFromName(e.types[0]);
    this.type2 = e.types.length === 1 ? null : typeFromName(e.types[1]);
    if (e.fca !== null) this.formeAbility = e.fca;
    else if (this.formeAbility === null) {
      this.possibleAbilities = e.abilities;
      if (e.abilities.length === 1 && this.gen >= 3) this.ability_ = e.abilities[0];
    } else this.formeAbility = null;
    this.weight = e.w;
  }
  updateFromDetails(details) {
    if (details === this.lastDetails) return;
    this.lastDetails = details;
    if (details.indexOf(', shiny') >= 0) { this.shiny = true; details = rep(details, ', shiny', ''); }
    else this.shiny = false;
    const sp = details.split(', ');
    for (let i = 0; i < sp.length; i++) {
      if (sp[i].startsWith('tera:')) { this.teraType = typeFromName(sp[i].slice(5)); sp.splice(i, 1); break; }
    }
    let species, level = null, gender = null;
    if (sp.length === 3) [species, level, gender] = sp;
    else if (sp.length === 2) {
      if (sp[1].startsWith('L')) [species, level] = sp; else [species, gender] = sp;
    } else species = toId(sp[0]);
    this.gender = gender || 'NEUTRAL';
    this.level = level ? parseInt(level.slice(1), 10) : 100;
    if (species !== this.species) this.updateFromPokedex(species);
  }
  updateFromRequest(r) {
    this.active = r.active;
    if (this.abilityP === null) this.setAbility(r.baseAbility);
    if ('ability' in r && r.ability !== r.baseAbility) this.tempAbility = toId(r.ability);
    this.lastRequest = r;
    this.setHpStatus(r.condition, true);
    this.name = r.ident.slice(4);
    this.item = r.item;
    this.updateFromDetails(r.details);
    for (const m of r.moves) this.addMove(m);
    if (r.stats) for (const s in r.stats) this.stats[s] = r.stats[s];
  }
  wasIllusioned(fields) {
    this.hp = null; this.maxHp = null; this.status = null;
    const lr = this.lastRequest;
    this.lastRequest = null;
    if (lr && Object.keys(lr).length) this.updateFromRequest(lr);
    this.switchOut(fields);
  }
  availableMovesFromRequest(req) {
    const C = loadCommon();
    const out = [];
    if (this.effects.has('COMMANDER')) return out;
    for (const rm of req.moves) {
      if (rm.disabled) continue;
      const id = rm.id;
      const mv = this.movesMap();
      if (mv.has(id)) {
        if (this.isDynamaxed) fail('NotImplemented dynamax move');
        out.push(mv.get(id));
      } else if (C.specialSet.has(id)) out.push(new Move(id, this.G));
      else {
        let hp = null, n = 0;
        for (const [k, v] of mv) if (k.startsWith('hiddenpower')) { n++; hp = v; }
        if (id === 'hiddenpower' && n === 1) out.push(hp);
        else {
          let ok = this.abilityP === 'dancer';
          for (const k of ['copycat', 'metronome', 'mefirst', 'mirrormove', 'assist']) if (mv.has(k)) ok = true;
          if (!ok) fail(`AssertionError Error with move ${id}`);
          out.push(new Move(id, this.G));
        }
      }
    }
    return out;
  }
}

// ---------------------------------------------------------------- SideTracker (poke-env Battle)
const MSG_IGNORE = new Set(['-anim', '-block', '-burst', '-center', '-combine', '-crit', '-fail', '-fieldactivate',
  '-hint', '-hitcount', '-miss', '-notarget', '-nothing', '-ohko', '-resisted', '-supereffective', '-waiting',
  '-zbroken', 'J', 'L', 'askreg', 'badge', 'c', 'chat', 'crit', 'debug', 'deinit', 'gametype', 'hidelines', 'html',
  'immune', 'inactiveoff', 'init', 'j', 'join', 'l', 'leave', 'n', 'name', 'rated', 'resisted', 'sentchoice',
  'split', 'supereffective', 'teampreview', 'upkeep', 'uhtml', 'zbroken', '']);
const PLAYER_IGNORE = new Set(['t:', 'expire', 'uhtmlchange', 'tempnotify', 'tempnotifyoff']);
const PRESSURE_TARGETS = new Set(['all', 'allAdjacent', 'allAdjacentFoes', 'any', 'normal', 'randomNormal', 'scripted']);
const CALYREX_IGNORE = new Set(['As One', 'Unnerve', 'Chilling Neigh', 'Grim Neigh']);
const LOCKED = new Set(['[from] lockedmove', '[from]lockedmove', '[from] Sky Attack']);
const PURSUIT = new Set(['[from] Pursuit', '[from]Pursuit', '[zeffect]']);
const CALLERS = new Set(['Sleep Talk', 'Copycat', 'Metronome', 'Nature Power']);

function genFromFormat(format) {
  const m = /^gen(\d)/.exec(format);
  return m ? parseInt(m[1], 10) : 9;
}

class SideTracker {
  constructor({format = 'gen9randombattle', side = 0, username = null} = {}) {
    this.gen = genFromFormat(format);
    this.G = loadGen(this.gen);
    this.C = loadCommon();
    this.side = side;
    this.username = username || (side ? 'p2' : 'p1');
    this.formatHint = format;
    this.format = null;
    this.role = null;
    this.turn = 0;
    this.team = new Map();
    this.oppTeam = new Map();
    this.teamSize = {};
    this.weather = new Map();
    this.fields = new Map();
    this.sideConds = new Map();
    this.oppSideConds = new Map();
    this.availableMoves = [];
    this.availableSwitches = [];
    this.canMega = false; this.canZ = false; this.canDynamax = false; this.canTera = false;
    this.forceSwitch = false; this.maybeTrapped = false; this.trapped = false;
    this.wait = false; this.teampreview = false; this.reviving = false;
    this.lastRequest = {};
    this.usedTera = false; this.oppUsedTera = false;
    this.won = null; this.finished = false;
    this.anybodyInactive = false; this.reconnected = true;
    this.inTeamPreview = false; this.usedMega = false; this.oppUsedMega = false;
    this.decisionPending = false;
  }

  // ---- public API
  onLog(text) {
    this.decisionPending = false;
    let i = 0;
    const n = text.length;
    while (i <= n) {
      let j = text.indexOf('\n', i);
      if (j < 0) j = n;
      if (j > i) this.handleLine(text.slice(i, j));
      i = j + 1;
    }
  }
  onRequest(req) {
    this.decisionPending = false;
    this.parseRequest(req);
    if (!this.wait && !this.teampreview) this.decisionPending = true;
  }
  onError(text) {
    // Player: '[Unavailable choice]' -> wait for the updated request; '[Invalid choice]' -> ask the agent again
    // (poke-env then sends Showdown's default with probability DEFAULT_CHOICE_CHANCE instead).
    let t = text.startsWith('|error|') ? text.slice(7) : text;
    if (t.startsWith('[Invalid choice]') && !this.wait && !this.teampreview) this.decisionPending = true;
  }
  markChosen() { this.decisionPending = false; }
  teamIdents() { return [...this.team.keys()]; }
  opponentIdents() { return [...this.oppTeam.keys()]; }

  get activePokemon() { for (const m of this.team.values()) if (m.active) return m; return null; }
  get oppActivePokemon() { for (const m of this.oppTeam.values()) if (m.active) return m; return null; }

  /** Player._handle_battle_message for one battle-room line. */
  handleLine(line) {
    const ev = line.split('|');
    if (ev.length === 1) return;
    const k = ev[1];
    if (k === '') return this.parseMessage(ev);
    if (PLAYER_IGNORE.has(k)) return;
    if (k === 'request') { if (ev[2]) this.onRequest(JSON.parse(ev.slice(2).join('|'))); return; }
    if (k === 'win') { this.won = ev[2] === this.username; this.finished = true; return; }
    if (k === 'tie') { this.finished = true; return; }
    if (k === 'error') { this.onError(ev.slice(2).join('|')); return; }
    if (k === 'bigerror' || k === 'showteam') return;
    this.parseMessage(ev);
  }

  // ---- AbstractBattle.get_pokemon
  getPokemon(identifier, forceSelf = false, details = '', request = null) {
    if (identifier.length < 4) fail(`IndexError identifier ${identifier}`);
    if (identifier[3] !== ' ') {
      identifier = identifier.slice(0, 2) + identifier.slice(3);
      if (identifier.slice(0, 2) === this.role && this.team.has(identifier)) {
        const believed = this.activePokemon;
        if (believed !== null) {
          const m = this.team.get(identifier);
          if (believed === m) return believed;
          if (m.ability_ === 'illusion') return m;
          if (believed.ability_ === 'illusion') return believed;
        }
      }
    }
    let m = this.team.get(identifier);
    if (m) return m;
    m = this.oppTeam.get(identifier);
    if (m) return m;
    const role = identifier.slice(0, 2);
    const name = pyStrip(identifier.slice(3));
    const own = role === this.role || forceSelf;
    let team = own ? this.team : this.oppTeam;
    const nameDet = details.split(', ')[0];
    const matches = [];
    let i = 0;
    for (const p of team.values()) { if (p.identifiesAs(nameDet)) matches.push(i); i++; }
    if (matches.length >= 2) fail('AssertionError get_pokemon matches');
    if (!team.has(identifier) && matches.length) {
      const items = [...team.entries()];
      const mi = matches[0];
      items[mi] = [identifier, items[mi][1]];
      items[mi][1].name = identifier.slice(4);
      items[mi][1].key = identifier;
      const nt = new Map(items);
      if (own) this.team = nt; else this.oppTeam = nt;
    }
    team = own ? this.team : this.oppTeam;
    if (team.has(identifier)) return team.get(identifier);
    if (Object.keys(this.teamSize).length && team.size >= this.teamSize[role]) {
      fail(`ValueError ${role}'s team already has ${this.teamSize[role]} pokemons: cannot add ${identifier}`);
    }
    if (request) m = new Mon(this.G, {request, name});
    else if (details) m = new Mon(this.G, {details, name});
    else m = new Mon(this.G, {species: identifier.slice(4), name});
    m.key = identifier;
    team.set(identifier, m);
    return m;
  }

  // ---- Battle methods
  clearAllBoosts() {
    const a = this.activePokemon, o = this.oppActivePokemon;
    if (a) a.clearBoosts();
    if (o) o.clearBoosts();
  }
  switch(pokemonStr, details, hpStatus) {
    const ident = pokemonStr.split(':')[0].slice(0, 2);
    const cur = ident === this.role ? this.activePokemon : this.oppActivePokemon;
    if (cur) cur.switchOut(this.fields);
    const p = this.getPokemon(pokemonStr, false, details);
    p.switchIn(details);
    p.setHpStatus(hpStatus);
  }
  endIllusion(pokemonName, details) {
    const active = pokemonName.slice(0, 2) === this.role ? this.activePokemon : this.oppActivePokemon;
    if (active === null) fail('ValueError Cannot end illusion without an active pokemon.');
    const ill = this.getPokemon(pokemonName, false, details);
    if (ill === active) return;
    ill.switchIn(details);
    ill.status = active.status;
    ill.setHpStatus(`${active.curHp}/${active.maxHpP}`);
    active.wasIllusioned(this.fields);
  }
  fieldEnd(s) {
    const f = fieldFromMsg(s);
    if (f !== 'UNKNOWN') {
      if (f === 'NEUTRALIZING_GAS') this.fields.delete(f);
      else { if (!this.fields.has(f)) fail(`KeyError field ${f}`); this.fields.delete(f); }
    }
  }
  fieldStart(s) {
    const f = fieldFromMsg(s);
    if (isTerrain(f)) for (const k of [...this.fields.keys()]) if (isTerrain(k)) this.fields.delete(k);
    this.fields.set(f, this.turn);
  }
  sideEnd(side, cond) {
    const c = side.slice(0, 2) === this.role ? this.sideConds : this.oppSideConds;
    const sc = sideFromMsg(cond);
    if (sc !== 'UNKNOWN') { if (!c.has(sc)) fail(`KeyError side condition ${sc}`); c.delete(sc); }
  }
  sideStart(side, cond) {
    const c = side.slice(0, 2) === this.role ? this.sideConds : this.oppSideConds;
    const sc = sideFromMsg(cond);
    if (sc in this.C.stackable) c.set(sc, (c.get(sc) || 0) + 1);
    else if (!c.has(sc)) c.set(sc, this.turn);
  }
  endTurn(turn) {
    this.turn = turn;
    const a = this.activePokemon, o = this.oppActivePokemon;
    if (a) a.endTurn();
    if (o) o.endTurn();
  }
  getTargetMon(pokemon, targetType, targetStr) {  // Battle._get_target_mon
    if (targetType !== 'all' && targetStr !== null) return this.getPokemon(targetStr);
    if (this.role === pokemon.slice(0, 2)) return this.oppActivePokemon;
    return this.activePokemon;
  }
  pressureOn(pokemon, move, targetStr) {
    const md = this.G.moves.get(retrieveId(move));
    if (!md) return false;
    const target = this.getTargetMon(pokemon, md.tgt, targetStr);
    if (target === null) return false;
    return target.abilityP === 'pressure' && !target.fainted && (PRESSURE_TARGETS.has(md.tgt) || md.mp === 1);
  }
  inTeam(mon) { for (const m of this.team.values()) if (m === mon) return true; return false; }
  inOppTeam(mon) { for (const m of this.oppTeam.values()) if (m === mon) return true; return false; }

  // ---- AbstractBattle.parse_message
  parseMessage(event) {
    const k = event[1];
    if (MSG_IGNORE.has(k)) return;
    const n = event.length;
    switch (k) {
      case 'drag': case 'switch':
        if (n < 5) fail('ValueError switch unpack');
        this.switch(event[2], event[3], event[4]);
        return;
      case '-damage': {
        if (n < 4) fail('ValueError -damage unpack');
        this.getPokemon(event[2]).setHpStatus(event[3]);
        this.checkDamageItem(event);
        this.checkDamageAbility(event);
        return;
      }
      case 'move': return this.parseMove(event.slice());
      case 'cant':
        if (n < 4) fail('ValueError cant unpack');
        this.getPokemon(event[2]).cantMove();
        return;
      case 'turn': this.endTurn(parseInt(event[2], 10)); return;
      case '-heal':
        if (n < 4) fail('ValueError -heal unpack');
        this.getPokemon(event[2]).setHpStatus(event[3]);
        this.checkHealAbility(event);
        this.checkHealItem(event);
        return;
      case '-boost': case '-unboost': {
        if (n < 5) fail('ValueError boost unpack');
        const amt = parseInt(event[4], 10);
        this.getPokemon(event[2]).boost(event[3], k === '-boost' ? amt : -amt);
        return;
      }
      case '-weather':
        if (event[2] === 'none') { this.weather = new Map(); return; }
        this.weather = new Map([[weatherFromMsg(event[2]), this.turn]]);
        return;
      case 'faint': this.getPokemon(event[2]).faint(); return;
      case '-ability': {
        if (n < 4) fail('ValueError -ability unpack');
        const cause = event[3];
        const mon = this.getPokemon(event[2]);
        if (mon.baseSpecies.indexOf('calyrex') >= 0 && CALYREX_IGNORE.has(cause)) return;
        if ((n > 4 && event[4].startsWith('[from] ability: Trace')) || (n > 5 && event[5].startsWith('[from] ability: Trace'))) {
          if (mon.abilityP !== 'trace') {
            if (mon.tempAbility !== null) mon.tempAbility = null;
            else if (mon.abilityP !== null) mon.ability_ = null;
            mon.setAbility('trace');
          }
          mon.setAbility(cause);
        } else if (cause === 'Neutralizing Gas') this.fieldStart(cause);
        else mon.setAbility(cause);
        return;
      }
      case '-start': {
        if (n < 4) fail('ValueError -start unpack');
        const effect = event[3];
        const mon = this.getPokemon(event[2]);
        if (effect === 'typechange') {
          let types;
          if (n > 5 && event[5].startsWith('[of] ')) types = this.getPokemon(event[5].slice(5)).types().join('/');
          else { if (n < 5) fail('IndexError typechange'); types = event[4]; }
          mon.startEffect(effect, types);
        } else {
          if (effect === 'Mimic') {
            if (n < 5) fail('IndexError Mimic');
            mon.moves.mimicMove = new Move(retrieveId(event[4]), this.G, null, true);
          }
          mon.startEffect(effect);
        }
        return;
      }
      case '-activate': return this.parseActivate(event);
      case '-status':
        if (n < 4) fail('ValueError -status unpack');
        this.getPokemon(event[2]).status = statusFromName(event[3]);
        return;
      case 'rule': return;
      case '-clearallboost': this.clearAllBoosts(); return;
      case '-clearboost': this.getPokemon(event[2]).clearBoosts(); return;
      case '-clearnegativeboost': {
        const b = this.getPokemon(event[2]).boosts;
        for (const s in b) if (b[s] < 0) b[s] = 0;
        return;
      }
      case '-clearpositiveboost': {
        const b = this.getPokemon(event[2]).boosts;
        for (const s in b) if (b[s] > 0) b[s] = 0;
        return;
      }
      case '-copyboost': {
        if (n < 4) fail('ValueError -copyboost unpack');
        const src = this.getPokemon(event[2]);
        this.getPokemon(event[3]).boosts = {...src.boosts};
        return;
      }
      case '-curestatus':
        if (n < 4) fail('ValueError -curestatus unpack');
        this.getPokemon(event[2]).cureStatus(event[3]);
        return;
      case '-cureteam': {
        const team = event[2].slice(0, 2) === this.role ? this.team : this.oppTeam;
        for (const m of team.values()) m.cureStatus();
        return;
      }
      case '-end':
        if (n < 4) fail('ValueError -end unpack');
        if (event[3] === 'ability: Neutralizing Gas') this.fieldEnd(event[3]);
        else this.getPokemon(event[2]).endEffect(event[3]);
        return;
      case '-endability': this.getPokemon(event[2]).tempAbility = null; return;
      case '-enditem':
        if (n < 4) fail('ValueError -enditem unpack');
        this.getPokemon(event[2]).endItem(event[3]);
        return;
      case '-fieldend': this.fieldEnd(event[2]); return;
      case '-fieldstart': this.fieldStart(event[2]); return;
      case '-formechange': case 'detailschange':
        if (n < 4) fail('ValueError formechange unpack');
        this.getPokemon(event[2]).formeChange(event[3]);
        return;
      case '-invertboost': {
        const m = this.getPokemon(event[2]);
        const b = {};
        for (const s in m.boosts) b[s] = -m.boosts[s];
        m.boosts = b;
        return;
      }
      case '-item': return this.parseItem(event);
      case '-mega': {
        if (event[2].startsWith(this.role)) this.usedMega = true; else this.oppUsedMega = true;
        fail('NotImplemented -mega');
        return;
      }
      case '-mustrecharge': this.getPokemon(event[2]).mustRecharge = true; return;
      case '-prepare': {
        let attacker, move, defMon = null;
        if (n >= 5) {
          attacker = event[2]; move = event[3];
          const defender = event[4];
          defMon = defender !== '[premajor]' ? this.getPokemon(defender) : null;
          if (defMon !== null && toId(move) === 'skydrop') defMon.startEffect('Sky Drop');
        } else {
          if (n < 4) fail('ValueError -prepare unpack');
          attacker = event[2]; move = event[3];
        }
        this.getPokemon(attacker).prepare(move, defMon);
        return;
      }
      case '-primal': fail('NotImplemented -primal'); return;
      case '-setboost':
        if (n < 5) fail('ValueError -setboost unpack');
        this.getPokemon(event[2]).setBoost(event[3], parseInt(event[4], 10));
        return;
      case '-sethp':
        if (n < 4) fail('ValueError -sethp unpack');
        this.getPokemon(event[2]).setHpStatus(event[3]);
        return;
      case '-sideend': if (n < 4) fail('ValueError'); this.sideEnd(event[2], event[3]); return;
      case '-sidestart': if (n < 4) fail('ValueError'); this.sideStart(event[2], event[3]); return;
      case '-singleturn': case '-singlemove':
        if (n < 4) fail('ValueError singleturn unpack');
        this.getPokemon(event[2]).startEffect(rep(event[3], 'move: ', ''));
        return;
      case '-swapboost': {
        if (n < 5) fail('ValueError -swapboost unpack');
        const s = this.getPokemon(event[2]), t = this.getPokemon(event[3]);
        const stats = event[4].indexOf('[from]') >= 0
          ? ['accuracy', 'atk', 'def', 'evasion', 'spa', 'spd', 'spe'] : event[4].split(', ');
        for (const st of stats) { const x = s.boosts[st]; s.boosts[st] = t.boosts[st]; t.boosts[st] = x; }
        return;
      }
      case '-transform': {
        if (n < 4) fail('ValueError -transform unpack');
        const mon = this.getPokemon(event[2]);
        if (n > 4 && event[4] === '[from] ability: Imposter') { mon.addMove('transform'); mon.setAbility('imposter'); }
        mon.transform(this.getPokemon(event[3]));
        return;
      }
      case '-zpower': return;
      case 'clearpoke': this.inTeamPreview = true; for (const m of this.team.values()) m.active = false; return;
      case 'gen':
        if (this.gen !== parseInt(event[2], 10)) fail(`RuntimeError Battle Initiated with gen ${this.gen} but got ${event[2]}`);
        return;
      case 'tier': this.format = event[2].toLowerCase().replace(/[^a-z0-9]+/g, ''); return;
      case 'inactive':
        if (event[2].indexOf('disconnected') >= 0) this.anybodyInactive = true;
        else if (event[2].indexOf('reconnected') >= 0) { this.anybodyInactive = false; this.reconnected = true; }
        return;
      case 'player': {
        let player, username;
        if (n === 6 || n === 5) { player = event[2]; username = event[3]; }
        else if (n === 4) { if (event[3] !== '') fail('RuntimeError Invalid player message'); return; }
        else {
          if (!this.anybodyInactive) {
            if (this.reconnected) this.reconnected = false; else fail('RuntimeError Invalid player message');
          }
          return;
        }
        if (username === this.username) this.role = player; else this.role = player === 'p2' ? 'p1' : 'p2';
        return;
      }
      case 'poke': return;  // teampreview registration of opponent mons (not in random battles)
      case 'raw': return;
      case 'replace':
        if (n < 4) fail('IndexError replace');
        this.endIllusion(event[2], event[3]);
        return;
      case 'start': this.inTeamPreview = false; return;
      case 'swap': return;
      case 'teamsize': this.teamSize[event[2]] = parseInt(event[3], 10); return;
      case 'message': case '-message': return;
      case '-immune':
        if (n === 4 && event[3].startsWith('[from] ability:')) {
          this.getPokemon(event[2]).setAbility(rep(event[3], '[from] ability:', ''));
        }
        return;
      case '-swapsideconditions': { const x = this.sideConds; this.sideConds = this.oppSideConds; this.oppSideConds = x; return; }
      case 'title': return;
      case '-terastallize': {
        if (n !== 4) fail('ValueError -terastallize unpack');
        const mon = this.getPokemon(event[2]);
        mon.terastallize(event[3]);
        if (mon.tera) {
          if (this.inTeam(mon)) this.usedTera = true;
          else if (this.inOppTeam(mon)) this.oppUsedTera = true;
        }
        return;
      }
      default:
        fail(`NotImplementedError ${event.join('|')}`);
    }
  }

  parseMove(event) {
    const pokemon0 = event[2];
    let mon = this.getPokemon(pokemon0);
    let use = !mon.dancing, failed = false, reveal = !mon.dancing, overridden = null, spread = false;
    mon.dancing = false;
    for (const s of ['[miss]', '[still]', '[notarget]']) if (last(event) === s) { event.pop(); failed = true; }
    if (last(event) === '[notarget]') event.pop();
    while (last(event).startsWith('[spread]')) { spread = true; event.pop(); }
    while (last(event) === '[still]') event.pop();
    if (LOCKED.has(last(event))) { use = false; reveal = false; event.pop(); }
    if (PURSUIT.has(last(event))) event.pop();
    if (last(event) === '[from] Sleep Talk') event[event.length - 1] = '[from] move: Sleep Talk';
    if (last(event).startsWith('[anim]')) event.pop();
    if (last(event).startsWith('[from] move: ') || last(event).startsWith('[from]move: ')) {
      overridden = lastSplit(event.pop(), ': ');
      if (overridden === 'Sleep Talk') { /* pass */ }
      else if (overridden === 'Copycat' || overridden === 'Metronome' || overridden === 'Nature Power' || overridden === 'Round') reveal = false;
      else if (overridden === 'Grass Pledge' || overridden === 'Water Pledge' || overridden === 'Fire Pledge') overridden = null;
    }
    if (last(event) === 'null') event.pop();
    if (last(event).startsWith('[from] ability: ') || last(event).startsWith('[from]ability: ')) {
      const ab = lastSplit(event.pop(), ': ');
      this.getPokemon(event[2]).setAbility(ab);
      if (ab === 'Magic Bounce') { use = false; reveal = false; }
      else if (ab === 'Dancer') return;
    }
    if (last(event) === '[from] Magic Coat' || last(event) === '[from] Mirror Move') { use = false; reveal = false; event.pop(); }
    let pokemon, move, target = null;
    if (event.length === 4) { pokemon = event[2]; move = event[3]; }
    else {
      if (event.length < 5) fail('ValueError move unpack');
      pokemon = event[2]; move = event[3]; target = event[4];
    }
    if (pyStrip(move.toUpperCase()) === 'MINIMIZE') this.getPokemon(pokemon).startEffect('MINIMIZE');
    if (spread || target === '') target = null;
    const pressure = this.pressureOn(pokemon, move, target);
    mon = this.getPokemon(pokemon);
    if (overridden) {
      mon.moved(move, failed, false, reveal);
      const ov = mon.movesMap().get(retrieveId(overridden));
      if (!ov) fail(`KeyError overridden move ${overridden}`);
      ov.use(pressure, true);
    } else if (!failed && CALLERS.has(move)) mon.moved(move, failed, use, reveal);
    else mon.moved(move, failed, use, reveal, pressure);
  }

  parseActivate(event) {
    const n = event.length;
    if (n < 4) fail('ValueError -activate unpack');
    let target = event[2];
    const effect = event[3];
    const strip = a => rep(rep(a, '[ability] ', ''), '[ability2] ', '');
    if (target && rep(effect, 'move: ', '') === 'Skill Swap') {
      if (n > 4) {
        if (last(event).startsWith('[of] ')) {
          const tm = this.getPokemon(target);
          const am = this.getPokemon(rep(last(event), '[of] ', ''));
          const abilities = event.slice(4, -1).map(strip).filter(a => a);
          if (abilities.length >= 2) {
            tm.startEffect(effect, abilities.slice(0, 2));
            am.tempAbility = toId(abilities[1]);
          } else if (tm.abilityP !== null && am.abilityP !== null) {
            const ta = tm.abilityP;
            tm.tempAbility = toId(am.abilityP);
            am.tempAbility = toId(ta);
          }
        } else {
          const am = this.getPokemon(event[4]);
          const abilities = event.slice(5).map(strip).filter(a => a);
          if (abilities.length >= 2) {
            this.getPokemon(target).startEffect(effect, abilities.slice(0, 2));
            am.tempAbility = toId(abilities[1]);
          }
        }
      }
    } else if (effect === 'ability: Dancer') this.getPokemon(target).dancing = true;
    else if (effect === 'ability: Mummy') {
      if (n < 6) fail('IndexError Mummy');
      target = event[5].indexOf('[of] ') >= 0 ? rep(event[5], '[of] ', '') : event[4];
      this.getPokemon(target).tempAbility = 'mummy';
    } else if (effect === 'ability: Wandering Spirit') {
      if (n < 7) fail('IndexError Wandering Spirit');
      const actor = event[2], tg = rep(event[6], '[of] ', '');
      this.getPokemon(actor).tempAbility = toId(event[4]);
      this.getPokemon(tg).tempAbility = 'wanderingspirit';
    } else if (effect === 'ability: Symbiosis') {
      if (n < 6) fail('IndexError Symbiosis');
      this.getPokemon(rep(event[5], '[of] ', '')).item = toId(rep(event[4], '[item] ', ''));
      this.getPokemon(target).item = null;
    } else if (effect === 'item: Leppa Berry') {
      const mon = this.getPokemon(target);
      const mv = mon.movesMap().get(toId(event[4]));
      if (!mv) fail('KeyError Leppa Berry move');
      mv.pp = Math.min(mv.pp + 10, mv.maxPP);
    } else if (effect === 'move: Mimic') {
      const mon = this.getPokemon(target);
      mon.moves.mimicMove = new Move(retrieveId(event[4]), this.G, null, true);
    } else if (effect === 'move: Trick') {
      const mon = this.getPokemon(target);
      const mon2 = this.getPokemon(rep(event[4], '[of] ', ''));
      const a = mon.item, b = mon2.item;
      mon.item = b; mon2.item = a;
    } else if (target !== '') this.getPokemon(target).startEffect(effect);
  }

  parseItem(event) {
    const n = event.length;
    if (n === 6) {
      const item = event[3], cause = event[4];
      let pokemon = event[5];
      if (cause === '[from] ability: Frisk') {
        pokemon = lastSplit(pokemon, '[of] ');
        const mon = this.getPokemon(pokemon);
        const a = this.activePokemon, o = this.oppActivePokemon;
        if (mon === a) { if (o === null) fail('AttributeError Frisk'); o.item = toId(item); }
        else if (mon === o) { if (a === null) fail('AttributeError Frisk'); a.item = toId(item); }
        mon.setAbility('frisk');
      } else if (cause === '[from] ability: Pickpocket' || cause === '[from] ability: Magician') {
        const thief = event[2], victim = rep(event[5], '[of] ', '');
        this.getPokemon(thief).item = toId(event[3]);
        this.getPokemon(thief).setAbility(cause === '[from] ability: Pickpocket' ? 'pickpocket' : 'magician');
        this.getPokemon(victim).item = null;
      } else if (cause === '[from] move: Thief' || cause === '[from] move: Covet') {
        const thief = event[2], victim = rep(event[5], '[of] ', '');
        this.getPokemon(thief).item = toId(event[3]);
        this.getPokemon(victim).item = null;
      } else fail(`ValueError Unhandled item message: ${event.join('|')}`);
    } else {
      if (n < 4) fail('ValueError -item unpack');
      const pokemon = event[2], item = event[3];
      if (n > 4 && (event[4] === '[from] ability: Magician' || event[4] === '[from] move: Switcheroo' || event[4] === '[from] move: Trick')) {
        const mon = this.getPokemon(pokemon);
        if (mon.item === 'unknown_item') mon.item = toId(item);
        return;
      }
      this.getPokemon(pokemon).item = toId(item);
    }
  }

  checkDamageItem(ev) {
    if (ev.length === 6 && ev[4].startsWith('[from] item:') && ev[5].startsWith('[of]')) {
      const item = lastSplit(ev[4], 'item:');
      this.getPokemon(pyStrip(lastSplit(ev[5], '[of]'))).item = toId(item);
    } else if (ev.length === 5 && ev[4].startsWith('[from] item:')) {
      this.getPokemon(ev[2]).item = toId(lastSplit(ev[4], 'item:'));
    }
  }
  checkDamageAbility(ev) {
    if (ev.length === 6 && ev[4].startsWith('[from] ability:') && ev[5].startsWith('[of]')) {
      const ab = lastSplit(ev[4], 'ability:');
      this.getPokemon(pyStrip(lastSplit(ev[5], '[of]'))).setAbility(ab);
    }
  }
  checkHealAbility(ev) {
    if (ev.length === 6 && ev[4].startsWith('[from] ability:')) {
      const ab = toId(lastSplit(ev[4], 'ability:'));
      if (ab === 'hospitality') this.getPokemon(pyStrip(rep(ev[5], '[of] ', ''))).setAbility(ab);
      else this.getPokemon(ev[2]).setAbility(ab);
    }
  }
  checkHealItem(ev) {
    if (ev.length === 5 && ev[4].startsWith('[from] item:')) {
      const p = this.getPokemon(ev[2]);
      const item = toId(lastSplit(ev[4], 'item:'));
      if (p.item !== null && item.indexOf('berry') < 0 && item.indexOf('herb') < 0) p.item = item;
    }
  }

  // ---- Battle.parse_request
  parseRequest(req) {
    this.wait = !!req.wait;
    const side = req.side;
    this.availableMoves = [];
    this.availableSwitches = [];
    this.canMega = false; this.canZ = false; this.canDynamax = false; this.canTera = false;
    this.maybeTrapped = false;
    this.reviving = false;
    for (const m of side.pokemon || []) if ('reviving' in m && m.reviving) this.reviving = true;
    this.trapped = false;
    this.forceSwitch = !!(req.forceSwitch ? req.forceSwitch[0] : false);
    this.lastRequest = req;
    this.teampreview = !!req.teamPreview;
    if (side.pokemon.length) this.role = side.pokemon[0].ident.slice(0, 2);
    this.updateTeamFromRequest(side);
    if (req.active) {
      const ar = req.active[0];
      if (ar.trapped) this.trapped = true;
      const a = this.activePokemon;
      if (a !== null) for (const m of a.availableMovesFromRequest(ar)) this.availableMoves.push(m);
      if (ar.canMegaEvo) this.canMega = true;
      if (ar.canZMove) this.canZ = true;
      if (ar.canDynamax) this.canDynamax = true;
      if (ar.canTerastallize) this.canTera = true;
      if (ar.maybeTrapped) this.maybeTrapped = true;
    }
    if (!this.trapped) {
      for (const pj of side.pokemon) {
        const p = this.team.get(pj.ident);
        if (!p) fail(`KeyError team ${pj.ident}`);
        if (this.reviving) { if (p.fainted) this.availableSwitches.push(p); }
        else if (!p.active && !p.fainted) this.availableSwitches.push(p);
      }
    }
  }
  updateTeamFromRequest(side) {
    const falsely = [], truly = [];
    for (const pj of side.pokemon) {
      if (!this.team.has(pj.ident)) this.getPokemon(pj.ident, true, pj.details, pj);
      const mon = this.team.get(pj.ident);
      if (!mon) fail(`KeyError team ${pj.ident}`);
      if (pj.active && !mon.active) truly.push(mon);
      else if (!pj.active && mon.active) falsely.push(mon);
    }
    for (const m of falsely) m.wasIllusioned(this.fields);
    for (const m of truly) m.switchIn();
    for (const pj of side.pokemon) {
      const mon = this.team.get(pj.ident);
      if (mon) mon.updateFromRequest(pj);
      else this.getPokemon(pj.ident, true, pj.details, pj);
    }
  }
}

module.exports = {SideTracker, Mon, Move, MoveSet, PokeEnvError, loadCommon, loadGen, loadSets, toId, retrieveId};
