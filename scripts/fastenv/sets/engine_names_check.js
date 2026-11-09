// Are the species / moves / items / abilities of generated teams known to poke-engine? Its Rust enums map an unknown
// name to NONE / UNKNOWNITEM silently (define_enum_with_from_str ... default = NONE), so a new entity would be
// simulated as nothing rather than crash.
//   node engine_names_check.js POKE_ENGINE_SRC FORMAT TEAMS_DIR OUT.json
// POKE_ENGINE_SRC: the poke-engine checkout (src/pokemon.rs, src/choices.rs, src/genx/{items,abilities}.rs).
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const [SRC, FORMAT, DIR, OUT] = process.argv.slice(2);
function variants(file, name) {
  const text = fs.readFileSync(path.join(SRC, file), 'utf8');
  const m = text.match(new RegExp(`\\b${name}\\s*\\{([^}]*)\\}`));
  if (!m) throw new Error(`${name} not found in ${file}`);
  return new Set(m[1].split(',').map(s => s.replace(/\/\/.*$/gm, '').trim()).filter(Boolean));
}
const E = {species: variants('src/pokemon.rs', 'PokemonName'), moves: variants('src/choices.rs', 'Choices'),
  items: variants('src/genx/items.rs', 'Items'), abilities: variants('src/genx/abilities.rs', 'Abilities')};
const out = {format: FORMAT, engineSizes: Object.fromEntries(Object.entries(E).map(([k, v]) => [k, v.size]))};
for (const v of ['pinned', 'current']) {
  const mons = zlib.gunzipSync(fs.readFileSync(path.join(DIR, `${FORMAT}_${v}_s1.jsonl.gz`))).toString().trim()
    .split('\n').flatMap(l => JSON.parse(l));
  const seen = {species: new Map(), speciesShown: new Map(), moves: new Map(), items: new Map(), abilities: new Map()};
  const add = (k, x) => x && seen[k].set(x, (seen[k].get(x) || 0) + 1);
  for (const [sid, sp, , item, ability, , moves] of mons) {
    add('species', sid); add('speciesShown', sp); add('items', item); add('abilities', ability);
    (moves ? moves.split(',') : []).forEach(x => add('moves', x));
  }
  out[v] = {};
  for (const [k, m] of Object.entries(seen)) {
    const ek = k === 'speciesShown' ? 'species' : k;
    const missing = [...m.entries()].filter(([x]) => !E[ek].has(x.toUpperCase())).sort((a, b) => b[1] - a[1]);
    const n = [...m.values()].reduce((a, b) => a + b, 0);
    out[v][k] = {distinct: m.size, missing: missing.map(([x, c]) => `${x}:${c}`),
      missingOccurrencePct: +(100 * missing.reduce((a, [, c]) => a + c, 0) / n).toFixed(3)};
  }
}
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
