// For every move / ability / item / species in a pairs file (teams.js): what Showdown's implementation of it does,
// as the list of its special keys (event handlers onX, callbacks, conditions, switch / multi-hit / field effects).
// coverage.py compares that against poke-engine's source.   node showdown_effects.js pairs.jsonl GEN > effects.json
const fs = require('fs');
const {Dex, toID} = require('../../../fastenv/js/ps');
const dex = Dex.forGen(parseInt(process.argv[3] || '9', 10));
const MOVE_KEYS = ['condition', 'sideCondition', 'slotCondition', 'pseudoWeather', 'weather', 'terrain', 'selfSwitch',
  'forceSwitch', 'multihit', 'selfdestruct', 'volatileStatus', 'ohko', 'isFutureMove', 'self', 'secondary',
  'secondaries', 'drain', 'recoil', 'heal', 'boosts', 'status', 'mindBlownRecoil', 'struggleRecoil', 'hasCrashDamage',
  'thawsTarget', 'willCrit', 'critRatio', 'breaksProtect', 'stallingMove', 'ignoreDefensive', 'ignoreAbility',
  'overrideOffensiveStat', 'overrideDefensiveStat', 'overrideOffensivePokemon', 'priority', 'damage', 'selfBoost'];
function special(obj, extra = []) {
  const out = [];
  for (const k of Object.keys(obj)) {
    if (typeof obj[k] === 'function') out.push(k);
    else if (extra.includes(k) && obj[k] !== undefined && obj[k] !== null && obj[k] !== false && obj[k] !== 0) out.push(k);
  }
  return out.sort();
}
const ids = {moves: new Set(), abilities: new Set(), items: new Set(), species: new Set()};
for (const line of fs.readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean)) {
  for (const team of JSON.parse(line)) for (const s of team) {
    s.moves.forEach(m => ids.moves.add(toID(m)));
    ids.abilities.add(toID(s.ability)); ids.items.add(toID(s.item)); ids.species.add(s.speciesId);
  }
}
const out = {moves: {}, abilities: {}, items: {}, species: {}};
for (const id of ids.moves) { const m = dex.moves.get(id); out.moves[id] = {name: m.name, basePower: m.basePower,
  category: m.category, keys: special(m, MOVE_KEYS)}; }
for (const id of ids.abilities) { const a = dex.abilities.get(id); out.abilities[id] = {name: a.name, keys: special(a)}; }
for (const id of ids.items) { const it = dex.items.get(id); out.items[id] = {name: it.name, keys: special(it,
  ['fling', 'naturalGift', 'isBerry', 'boosts', 'megaStone'])}; }
for (const id of ids.species) { const sp = dex.species.get(id); out.species[id] = {name: sp.name,
  forme: sp.forme || '', baseSpecies: sp.baseSpecies}; }
process.stdout.write(JSON.stringify(out));
