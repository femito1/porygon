// How far the current month's random-battle sets are from the pinned ones, from teams made by gen_teams.js.
//   node analyze.js FORMAT DIR PINNED_PS CURRENT_PS VOCAB.json LIBRARY.json.gz POKE_ENV_STATIC OUT.json
// DIR holds FORMAT_{pinned,current}_s{1,2}.jsonl.gz (same seeds in s1 of both builds, independent seeds in s2).
// Reports (all over generated mons; a team is 6 mons):
//   paired     pinned s1 vs current s1 mon by mon (same seeds): identical / level-only / other field changes
//   static     share of current mons whose species' sets.json entry changed (level, movepool, abilities, tera, roles)
//   dist       species-frequency TV and per-species TV of the move / item / ability / tera / level / whole-set
//              marginals, weighted by species frequency, for pinned s1 vs current s1 (paired), pinned s1 vs current s2
//              (independent) and pinned s1 vs pinned s2 (the sampling-noise floor of the same estimate)
//   support    share of current mons whose (moves, ability[, tera]) under their role is impossible under the pinned
//              sets.json (static necessary condition), and whose (species, moves, item, ability) or full set is unseen
//              in a pinned sample / in our set library (sampled estimate; pinned s2 gives the false-positive floor)
//   vocab      species / moves / items / abilities of current (and pinned) teams missing from vocab.json (raw and
//              after the encoder's fallbacks), new relative to the pinned output, missing from poke-env's data
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const [FORMAT, DIR, PINNED, CURRENT, VOCAB, LIBRARY, PE, OUT] = process.argv.slice(2);
if (!OUT) {
  console.error('usage: node analyze.js FORMAT DIR PINNED_PS CURRENT_PS VOCAB.json LIBRARY.json.gz POKE_ENV_STATIC OUT');
  process.exit(2);
}
const GEN = FORMAT.match(/^gen(\d)/)[1];
const {Dex} = require(path.join(path.resolve(CURRENT), 'dist/sim'));
const dex = Dex.forGen(+GEN);
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const F = {sid: 0, sp: 1, level: 2, item: 3, ability: 4, tera: 5, moves: 6, evs: 7, ivs: 8, role: 9, gender: 10, shiny: 11};
const FIELDS = Object.keys(F);

function load(name) {
  const file = path.join(DIR, `${FORMAT}_${name}.jsonl.gz`);
  const teams = zlib.gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(l => JSON.parse(l));
  return {name, teams, mons: teams.flat()};
}
const P1 = load('pinned_s1'), P2 = load('pinned_s2'), C1 = load('current_s1'), C2 = load('current_s2');
const pct = (x, n) => n ? +(100 * x / n).toFixed(3) : 0;
const out = {format: FORMAT, pinned: PINNED, current: CURRENT, teams: {}, paired: {}, static: {}, dist: {},
  support: {}, vocab: {}, library: {}};
for (const r of [P1, P2, C1, C2]) out.teams[r.name] = {teams: r.teams.length, mons: r.mons.length};

// ---- paired: same seeds, mon by mon --------------------------------------------------------------------------
function paired(A, B) {
  const r = {teams: Math.min(A.teams.length, B.teams.length), teamsIdentical: 0, teamsIdenticalExceptLevel: 0,
    teamsSameSpecies: 0, mons: 0, monsIdentical: 0, monsLevelOnly: 0, monsOtherChange: 0, fieldDiffers: {}};
  for (const f of FIELDS) r.fieldDiffers[f] = 0;
  for (let t = 0; t < r.teams; t++) {
    const a = A.teams[t], b = B.teams[t];
    let same = a.length === b.length, sameNoLevel = same, sameSpecies = same;
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      r.mons++;
      const x = a[i], y = b[i];
      if (!x || !y) { same = sameNoLevel = sameSpecies = false; r.monsOtherChange++; continue; }
      const diff = FIELDS.filter(f => x[F[f]] !== y[F[f]]);
      diff.forEach(f => r.fieldDiffers[f]++);
      if (!diff.length) r.monsIdentical++;
      else if (diff.length === 1 && diff[0] === 'level') r.monsLevelOnly++;
      else r.monsOtherChange++;
      if (diff.length) same = false;
      if (diff.some(f => f !== 'level')) sameNoLevel = false;
      if (x[F.sid] !== y[F.sid]) sameSpecies = false;
    }
    r.teamsIdentical += same; r.teamsIdenticalExceptLevel += sameNoLevel; r.teamsSameSpecies += sameSpecies;
  }
  for (const k of ['teamsIdentical', 'teamsIdenticalExceptLevel', 'teamsSameSpecies']) r[k + 'Pct'] = pct(r[k], r.teams);
  for (const k of ['monsIdentical', 'monsLevelOnly', 'monsOtherChange']) r[k + 'Pct'] = pct(r[k], r.mons);
  return r;
}
out.paired.s1 = paired(P1, C1);
out.paired.s2 = paired(P2, C2);

// ---- static: which species' sets.json entries changed ----------------------------------------------------------
const setsOf = ps => JSON.parse(fs.readFileSync(path.join(ps, 'data/random-battles', `gen${GEN}`, 'sets.json'), 'utf8'));
const SA = setsOf(PINNED), SB = setsOf(CURRENT);
const norm = e => JSON.stringify((e?.sets || []).map(s => ({
  role: s.role, movepool: (s.movepool || []).map(toID).sort(), abilities: (s.abilities || []).map(toID).sort(),
  tera: (s.teraTypes || []).map(toID).sort(),
})));
const levelChanged = new Set(), setChanged = new Set(), addedOrRemoved = new Set();
for (const sp of new Set([...Object.keys(SA), ...Object.keys(SB)])) {
  if (!(sp in SA) || !(sp in SB)) { addedOrRemoved.add(sp); continue; }
  if (SA[sp].level !== SB[sp].level) levelChanged.add(sp);
  if (norm(SA[sp]) !== norm(SB[sp])) setChanged.add(sp);
}
const share = (mons, set) => pct(mons.filter(m => set.has(m[F.sid])).length, mons.length);
out.static = {
  speciesLevelChanged: [...levelChanged].sort(), speciesSetChanged: [...setChanged].sort(),
  speciesAddedOrRemoved: [...addedOrRemoved].sort(),
  currentMonsInLevelChangedSpeciesPct: share(C1.mons.concat(C2.mons), levelChanged),
  currentMonsInSetChangedSpeciesPct: share(C1.mons.concat(C2.mons), setChanged),
  currentMonsInAddedSpeciesPct: share(C1.mons.concat(C2.mons), addedOrRemoved),
  currentMonsInAnyChangedSpeciesPct: share(C1.mons.concat(C2.mons), new Set([...levelChanged, ...setChanged, ...addedOrRemoved])),
};

// ---- distributional ----------------------------------------------------------------------------------------------
const setKey = m => [m[F.moves], m[F.item], m[F.ability], m[F.tera], m[F.evs], m[F.ivs], m[F.role]].join('|');
const MARGINALS = {
  moves: m => m[F.moves] ? m[F.moves].split(',') : [],  // a uniformly drawn move slot
  item: m => [m[F.item]], ability: m => [m[F.ability]], tera: m => [m[F.tera]], level: m => [String(m[F.level])],
  set: m => [setKey(m)],
};
function bySpecies(mons) {
  const sp = new Map();
  for (const m of mons) {
    let e = sp.get(m[F.sid]);
    if (!e) sp.set(m[F.sid], e = {n: 0, c: Object.fromEntries(Object.keys(MARGINALS).map(k => [k, new Map()]))});
    e.n++;
    for (const [k, f] of Object.entries(MARGINALS)) for (const v of f(m)) e.c[k].set(v, (e.c[k].get(v) || 0) + 1);
  }
  return sp;
}
function tv(ca, cb) {  // total-variation distance of two count maps
  let na = 0, nb = 0;
  for (const v of ca.values()) na += v;
  for (const v of cb.values()) nb += v;
  if (!na || !nb) return na || nb ? 1 : 0;
  let s = 0;
  for (const k of new Set([...ca.keys(), ...cb.keys()])) s += Math.abs((ca.get(k) || 0) / na - (cb.get(k) || 0) / nb);
  return s / 2;
}
function dist(A, B) {
  const a = bySpecies(A.mons), b = bySpecies(B.mons), NA = A.mons.length, NB = B.mons.length;
  const r = {speciesTV: 0, speciesOnlyInA: [], speciesOnlyInB: [], weightedTV: {}, monsInSpeciesWithTVgt0Pct: {},
    top: {}};
  const all = new Set([...a.keys(), ...b.keys()]);
  let s = 0;
  for (const sp of all) s += Math.abs((a.get(sp)?.n || 0) / NA - (b.get(sp)?.n || 0) / NB);
  r.speciesTV = +(s / 2).toFixed(5);
  r.speciesOnlyInA = [...all].filter(sp => !b.has(sp)).sort();
  r.speciesOnlyInB = [...all].filter(sp => !a.has(sp)).sort();
  const both = [...all].filter(sp => a.has(sp) && b.has(sp));
  const w = new Map(both.map(sp => [sp, (a.get(sp).n / NA + b.get(sp).n / NB) / 2]));
  const W = [...w.values()].reduce((x, y) => x + y, 0);
  for (const k of Object.keys(MARGINALS)) {
    let acc = 0, changedMons = 0;
    const per = [];
    for (const sp of both) {
      const d = tv(a.get(sp).c[k], b.get(sp).c[k]);
      acc += w.get(sp) * d;
      if (d > 0) changedMons += b.get(sp).n;
      per.push([sp, +d.toFixed(4), +(100 * w.get(sp)).toFixed(3)]);
    }
    r.weightedTV[k] = +(acc / W).toFixed(5);
    r.monsInSpeciesWithTVgt0Pct[k] = pct(changedMons, NB);
    r.top[k] = per.sort((x, y) => y[1] * y[2] - x[1] * x[2]).slice(0, 8);  // largest weight x TV
  }
  return r;
}
out.dist['pinned_s1 vs current_s1 (paired seeds)'] = dist(P1, C1);
out.dist['pinned_s1 vs current_s2 (independent)'] = dist(P1, C2);
out.dist['pinned_s1 vs pinned_s2 (noise floor)'] = dist(P1, P2);

// ---- support: probability-0 combinations under the pinned generator ---------------------------------------------
function staticOK(m, sets, withTera) {  // some pinned set of this role allows these moves / ability (/ tera)?
  const e = sets[m[F.sid]];
  if (!e) return false;
  const moves = m[F.moves] ? m[F.moves].split(',') : [];
  return e.sets.some(s => (!m[F.role] || s.role === m[F.role]) &&
    moves.every(mv => (s.movepool || []).some(x => toID(x) === mv)) &&
    (!s.abilities || !m[F.ability] || s.abilities.some(x => toID(x) === m[F.ability])) &&
    (!withTera || !s.teraTypes || !m[F.tera] || s.teraTypes.some(x => toID(x) === m[F.tera])));
}
const failShare = (mons, sets, withTera) => pct(mons.filter(m => !staticOK(m, sets, withTera)).length, mons.length);
const comboKey = m => [m[F.sid], m[F.moves], m[F.item], m[F.ability]].join('|');
const unseenShare = (mons, ref, key) => {
  const seen = new Set(ref.map(key));
  return pct(mons.filter(m => !seen.has(key(m))).length, mons.length);
};
out.support = {
  staticImpossibleUnderPinned: {
    'current s1+s2 (moves, ability)': failShare(C1.mons.concat(C2.mons), SA, false),
    'current s1+s2 (moves, ability, tera)': failShare(C1.mons.concat(C2.mons), SA, true),
    'pinned s1+s2 self-check (must be 0)': failShare(P1.mons.concat(P2.mons), SA, true),
    'current s1+s2 vs current sets.json self-check (must be 0)': failShare(C1.mons.concat(C2.mons), SB, true),
  },
  unseenInPinnedSample: {
    'current s2 (species, moves, item, ability) unseen in pinned s1': unseenShare(C2.mons, P1.mons, comboKey),
    'pinned s2 unseen in pinned s1 (false-positive floor)': unseenShare(P2.mons, P1.mons, comboKey),
    'current s2 unseen in pinned s1+s2': unseenShare(C2.mons, P1.mons.concat(P2.mons), comboKey),
  },
};

// ---- our set library (the inference prior / v2 posterior) --------------------------------------------------------
const spread = (s, dflt) => s.split(',').every(x => +x === dflt) ? '' : s;
// library set = [moves, ability, item, role, evs, ivs, (tera,) n, n_lead]; libraries built before the tera field
// (gen4's of 2026-10-28) have 8 entries
let libHasTera = true;
const libKey = m => [m[F.moves], m[F.ability], m[F.item], m[F.role], spread(m[F.evs], 85), spread(m[F.ivs], 31),
  ...(libHasTera ? [m[F.tera]] : [])].join('|');
if (LIBRARY && fs.existsSync(LIBRARY)) {
  const lib = JSON.parse(zlib.gunzipSync(fs.readFileSync(LIBRARY)).toString());
  const keys = new Set(), combos = new Set();
  libHasTera = Object.values(lib.species).every(e => e.sets.every(s => s.length === 9));
  for (const [sp, e] of Object.entries(lib.species)) {
    for (const s of e.sets) {
      keys.add(`${sp}|${s.slice(0, libHasTera ? 7 : 6).join('|')}`);
      combos.add([sp, s[0], s[2], s[1]].join('|'));
    }
  }
  const miss = (mons, f, set) => pct(mons.filter(m => !set.has(f(m))).length, mons.length);
  const staleLevel = Object.keys(lib.species).filter(sp => SB[sp] && SB[sp].level !== lib.species[sp].level);
  const libVsPinned = Object.keys(lib.species).filter(sp => SA[sp] && SA[sp].level !== lib.species[sp].level);
  out.library = {
    file: LIBRARY, teams: lib.teams, species: Object.keys(lib.species).length, hasTeraField: libHasTera,
    'current mons: full set absent from library %': miss(C1.mons.concat(C2.mons), m => `${m[F.sid]}|${libKey(m)}`, keys),
    'pinned mons: full set absent from library % (floor)': miss(P1.mons.concat(P2.mons), m => `${m[F.sid]}|${libKey(m)}`, keys),
    'current mons: (moves, item, ability) absent from library %': miss(C1.mons.concat(C2.mons), comboKey, combos),
    'pinned mons: (moves, item, ability) absent from library % (floor)': miss(P1.mons.concat(P2.mons), comboKey, combos),
    'current mons: species absent from library %': pct(C1.mons.concat(C2.mons).filter(m => !lib.species[m[F.sid]]).length, 2 * C1.mons.length),
    speciesLevelDiffersFromCurrentSets: staleLevel.map(sp => `${sp} ${lib.species[sp].level}->${SB[sp].level}`),
    speciesLevelDiffersFromPinnedSets: libVsPinned.map(sp => `${sp} ${lib.species[sp].level} vs ${SA[sp].level}`),
    'current mons whose level differs from the library level %': pct(C1.mons.concat(C2.mons).filter(m => lib.species[m[F.sid]] && lib.species[m[F.sid]].level !== m[F.level]).length, 2 * C1.mons.length),
  };
}

// ---- vocabulary ----------------------------------------------------------------------------------------------------
const vocab = JSON.parse(fs.readFileSync(VOCAB, 'utf8'));
const V = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [k, new Set(v)]));
const viaFallback = (kind, key) => {  // porygon.encoder.idx: hidden power variants, trailing digits
  if (V[kind].has(key)) return true;
  if (kind === 'moves' && key.startsWith('hiddenpower')) return V.moves.has('hiddenpower');
  return V[kind].has(key.replace(/[0-9]+$/, ''));
};
function entities(mons) {
  const e = {species: new Set(), speciesShown: new Set(), moves: new Set(), items: new Set(), abilities: new Set()};
  for (const m of mons) {
    e.species.add(m[F.sid]); e.speciesShown.add(m[F.sp]);
    if (m[F.moves]) m[F.moves].split(',').forEach(x => e.moves.add(x));
    if (m[F.item]) e.items.add(m[F.item]);
    if (m[F.ability]) e.abilities.add(m[F.ability]);
  }
  return e;
}
const peDex = JSON.parse(fs.readFileSync(path.join(PE, 'pokedex', `gen${GEN}pokedex.json`), 'utf8'));
const peMoves = JSON.parse(fs.readFileSync(path.join(PE, 'moves', `gen${GEN}moves.json`), 'utf8'));
const cur = entities(C1.mons.concat(C2.mons)), pin = entities(P1.mons.concat(P2.mons));
for (const kind of ['species', 'speciesShown', 'moves', 'items', 'abilities']) {
  const vk = kind === 'speciesShown' ? 'species' : kind;
  const list = [...cur[kind]].sort();
  const missingRaw = list.filter(x => !V[vk].has(x));
  const missingFallback = list.filter(x => {
    if (viaFallback(vk, x)) return false;
    if (vk === 'species') return !V.species.has(toID(dex.species.get(x).baseSpecies));  // encoder: base species
    return true;
  });
  out.vocab[kind] = {
    current: list.length, pinned: pin[kind].size,
    newInCurrent: list.filter(x => !pin[kind].has(x)), goneInCurrent: [...pin[kind]].filter(x => !cur[kind].has(x)).sort(),
    missingFromVocabRaw: missingRaw, missingFromVocabAfterFallback: missingFallback,
    pinnedMissingFromVocabRaw: [...pin[kind]].filter(x => !V[vk].has(x)).sort(),
  };
}
out.vocab.pokeEnv = {
  speciesShownMissing: [...cur.speciesShown].filter(x => !peDex[x]).sort(),
  speciesIdMissing: [...cur.species].filter(x => !peDex[x]).sort(),
  movesMissing: [...cur.moves].filter(x => !peMoves[x]).sort(),
};

fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
// ---- summary -------------------------------------------------------------------------------------------------------
const L = [];
L.push(`== ${FORMAT}: ${JSON.stringify(out.teams)}`);
const p = out.paired.s1;
L.push(`paired s1 (same seeds): teams identical ${p.teamsIdenticalPct}%, identical except level ${p.teamsIdenticalExceptLevelPct}%, ` +
  `same species ${p.teamsSameSpeciesPct}%; mons identical ${p.monsIdenticalPct}%, level-only ${p.monsLevelOnlyPct}%, ` +
  `other change ${p.monsOtherChangePct}%; field differs ${JSON.stringify(p.fieldDiffers)}`);
L.push(`paired s2: mons identical ${out.paired.s2.monsIdenticalPct}%, level-only ${out.paired.s2.monsLevelOnlyPct}%, other ${out.paired.s2.monsOtherChangePct}%`);
L.push(`static: ${JSON.stringify(out.static)}`);
for (const [k, d] of Object.entries(out.dist)) {
  L.push(`dist ${k}: speciesTV ${d.speciesTV}; weighted per-species TV ${JSON.stringify(d.weightedTV)}; ` +
    `mons in species with TV>0 % ${JSON.stringify(d.monsInSpeciesWithTVgt0Pct)}; only-in-A ${d.speciesOnlyInA.length}, only-in-B ${d.speciesOnlyInB.length}`);
}
L.push(`support: ${JSON.stringify(out.support)}`);
L.push(`library: ${JSON.stringify(out.library)}`);
for (const [k, v] of Object.entries(out.vocab)) L.push(`vocab ${k}: ${JSON.stringify(v)}`);
fs.writeFileSync(OUT.replace(/\.json$/, '.txt'), L.join('\n') + '\n');
console.log(L.join('\n'));
