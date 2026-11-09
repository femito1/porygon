// Would re-running scripts/build_data.js on another Showdown build change vocab.json? Its ids are positional (sorted
// lists, network id = position + 2), so ANY added/removed entity shifts every later id and breaks trained embeddings.
//   node vocab_diff.js PINNED_PS CURRENT_PS VOCAB.json OUT.json
'use strict';
const fs = require('fs');
const path = require('path');

const child = require('child_process');

if (process.argv[2] === '--dump') {  // one build per process: two dist/sim trees never share a process
  const {Dex} = require(path.join(path.resolve(process.argv[3]), 'dist/sim'));
  const ids = table => [...new Set(table.all().map(e => e.id))].sort();  // exactly build_data.js's lists
  process.stdout.write(JSON.stringify({species: ids(Dex.species), moves: ids(Dex.moves), items: ids(Dex.items),
    abilities: ids(Dex.abilities)}));
  process.exit(0);
}
const [PIN, CUR, VOCAB, OUT] = process.argv.slice(2);
const vocabOf = ps => JSON.parse(child.execFileSync(process.execPath, [__filename, '--dump', ps],
  {maxBuffer: 1 << 26}).toString());
const file = JSON.parse(fs.readFileSync(VOCAB, 'utf8'));
const pin = vocabOf(PIN), cur = vocabOf(CUR);
const out = {};
for (const k of Object.keys(file)) {
  const F = new Set(file[k]), P = new Set(pin[k]), C = new Set(cur[k]);
  let shifted = 0;
  for (let i = 0; i < file[k].length; i++) if (cur[k].indexOf(file[k][i]) !== i) shifted++;
  out[k] = {file: file[k].length, pinnedBuild: pin[k].length, currentBuild: cur[k].length,
    fileEqualsPinnedBuild: JSON.stringify(file[k]) === JSON.stringify(pin[k]),
    addedInCurrent: [...C].filter(x => !F.has(x)), removedInCurrent: [...F].filter(x => !C.has(x)),
    pinnedBuildNotInFile: [...P].filter(x => !F.has(x)), idsShiftedIfRegenerated: shifted};
}
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
