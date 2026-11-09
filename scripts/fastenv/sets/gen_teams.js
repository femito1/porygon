// Generate N random-battle teams with one Showdown build and fixed seeds, as gzipped JSON lines (one team a line).
//   node gen_teams.js PS_DIR FORMAT SEEDSET N OUT.jsonl.gz
// Team i of seed set k uses the seed `sodium,<k:08x><i:08x>5eed5eedfa57fa57` (ChaCha20, like the live servers), so
// two builds given the same (k, i) are a paired comparison and two seed sets k are independent samples.
// Each mon is [speciesId, species (display forme), level, item, ability, tera, moves (ids sorted, comma-joined),
//              evs "hp,atk,def,spa,spd,spe", ivs, role, gender, shiny]; item/ability/tera are ids ('' when absent).
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const [PS, FORMAT, SEEDSET, N, OUT] = process.argv.slice(2);
if (!OUT) {
  console.error('usage: node gen_teams.js PS_DIR FORMAT SEEDSET N OUT.jsonl.gz');
  process.exit(2);
}
const {Teams, toID} = require(path.join(path.resolve(PS), 'dist/sim'));
const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const hex8 = x => (x >>> 0).toString(16).padStart(8, '0');
const seedOf = (k, i) => `sodium,${hex8(k)}${hex8(i)}5eed5eedfa57fa57`;

const t0 = Date.now();
const lines = [];
for (let i = 0; i < +N; i++) {
  const team = Teams.generate(FORMAT, {seed: seedOf(+SEEDSET, i)});
  lines.push(JSON.stringify(team.map(s => [
    s.speciesId || toID(s.species), toID(s.species), s.level, toID(s.item || ''), toID(s.ability || ''),
    toID(s.teraType || ''), s.moves.map(toID).sort().join(','), STATS.map(k => s.evs[k]).join(','),
    STATS.map(k => s.ivs[k]).join(','), s.role || '', s.gender || '', s.shiny ? 1 : 0,
  ])));
}
fs.writeFileSync(OUT, zlib.gzipSync(lines.join('\n') + '\n'));
console.error(`${FORMAT} seedset ${SEEDSET}: ${N} teams in ${((Date.now() - t0) / 1000).toFixed(1)} s ` +
  `(${PS}) -> ${OUT}`);
