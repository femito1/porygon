// Library of gen4 random-battle sets, sampled from Showdown's own team generator (data/random-battles/gen4/teams.ts),
// the prior `porygon.inference` samples opponent sets from.
// Usage: node scripts/build_set_library.js [TEAMS=400000] [WORKERS=4]  (after `node build` in pokemon-showdown/)
//
// Output src/porygon/data/gen4randombattle_library.json.gz:
//   {teams, species: {id: {level, sets: [[moves, ability, item, role, evs, ivs, n, n_lead], ...]}}}
// moves sorted and comma-joined, evs / ivs "hp,atk,def,spa,spd,spe" or "" for the defaults (85s / 31s),
// n the times the generator made this set, n_lead how many of those were the team's lead (slot 0, which the
// generator builds with isLead: Focus Sash).
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const {Worker, isMainThread, parentPort, workerData} = require('worker_threads');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const spread = (o, d) => STATS.every(s => o[s] === d) ? '' : STATS.map(s => o[s]).join(',');

function count(n) {
  const {Teams, toID} = require('../pokemon-showdown/dist/sim');
  const out = {};
  for (let t = 0; t < n; t++) {
    Teams.generate('gen4randombattle').forEach((set, i) => {
      const key = [[...set.moves].map(toID).sort().join(','), toID(set.ability), toID(set.item), set.role,
        spread(set.evs, 85), spread(set.ivs, 31)].join('|');
      const sp = (out[set.speciesId] ??= {level: set.level, sets: {}});
      const c = (sp.sets[key] ??= [0, 0]);
      c[0]++;
      if (i === 0) c[1]++;
    });
  }
  return out;
}

if (!isMainThread) {
  parentPort.postMessage(count(workerData));
} else {
  const total = Number(process.argv[2] || 400000), workers = Number(process.argv[3] || 4);
  const jobs = [...Array(workers)].map((_, i) => new Promise((resolve, reject) => {
    const w = new Worker(__filename, {workerData: Math.floor(total / workers) + (i < total % workers)});
    w.on('message', resolve);
    w.on('error', reject);
  }));
  Promise.all(jobs).then(parts => {
    const merged = {};
    for (const part of parts) {
      for (const [id, sp] of Object.entries(part)) {
        const m = (merged[id] ??= {level: sp.level, sets: {}});
        for (const [key, [n, lead]] of Object.entries(sp.sets)) {
          const c = (m.sets[key] ??= [0, 0]);
          c[0] += n;
          c[1] += lead;
        }
      }
    }
    const species = {};
    for (const id of Object.keys(merged).sort()) {
      const sets = Object.entries(merged[id].sets).sort((a, b) => b[1][0] - a[1][0]);
      species[id] = {level: merged[id].level, sets: sets.map(([key, [n, lead]]) => [...key.split('|'), n, lead])};
    }
    const file = path.join(__dirname, '..', 'src', 'porygon', 'data', 'gen4randombattle_library.json.gz');
    fs.writeFileSync(file, zlib.gzipSync(JSON.stringify({teams: total, species})));
    const nsets = Object.values(species).reduce((a, s) => a + s.sets.length, 0);
    console.log(`${total} teams: ${Object.keys(species).length} species, ${nsets} distinct sets -> ${file}`);
  });
}
