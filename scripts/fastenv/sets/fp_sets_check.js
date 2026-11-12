// Which month's sets does Foul Play's cached random-battle set file describe? (data file only: Foul Play caches
// https://pkmn.github.io/randbats/data/full/FORMAT.json in fp/data/pkmn_sets_cache/ on first use, never refreshes it)
//   node fp_sets_check.js FORMAT FP_CACHE.json PINNED_PS CURRENT_PS TEAMS_DIR OUT.json
// Compares the cache's per-species level with each build's sets.json and measures how many generated mons (pinned s1 /
// current s1 from gen_teams.js) have their exact set — "level,item,ability,moves(sorted)[,tera]" — in the cache.
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const [FORMAT, CACHE, PINNED, CURRENT, DIR, OUT] = process.argv.slice(2);
const GEN = FORMAT.match(/^gen(\d)/)[1];
const fp = JSON.parse(fs.readFileSync(CACHE, 'utf8'));
const sets = ps => JSON.parse(fs.readFileSync(path.join(ps, 'data/random-battles', `gen${GEN}`, 'sets.json'), 'utf8'));
const SA = sets(PINNED), SB = sets(CURRENT);
const fpLevel = {}, fpKeys = new Set(), fpNoLevel = new Set();
for (const [sp, s] of Object.entries(fp)) {
  const lv = {};
  for (const k of Object.keys(s)) {
    const parts = k.split(',');
    lv[parts[0]] = (lv[parts[0]] || 0) + s[k];
    fpKeys.add(`${sp}|${k}`);
    fpNoLevel.add(`${sp}|${parts.slice(1).join(',')}`);
  }
  fpLevel[sp] = +Object.entries(lv).sort((a, b) => b[1] - a[1])[0][0];
}
const out = {format: FORMAT, cache: CACHE, cacheMtime: fs.statSync(CACHE).mtime, species: Object.keys(fp).length,
  levelAgreesWithPinned: 0, levelAgreesWithCurrent: 0, levelDiffers: [], notInSets: [], coverage: {}};
for (const [sp, lv] of Object.entries(fpLevel)) {
  if (!SA[sp] && !SB[sp]) { out.notInSets.push(sp); continue; }
  const a = SA[sp]?.level, b = SB[sp]?.level;
  out.levelAgreesWithPinned += lv === a;
  out.levelAgreesWithCurrent += lv === b;
  if (a !== b || lv !== a) out.levelDiffers.push(`${sp}: cache ${lv}, pinned ${a}, current ${b}`);
}
for (const v of ['pinned', 'current']) {
  const mons = zlib.gunzipSync(fs.readFileSync(path.join(DIR, `${FORMAT}_${v}_s1.jsonl.gz`))).toString().trim()
    .split('\n').flatMap(l => JSON.parse(l));
  let full = 0, noLevel = 0, species = 0;
  for (const m of mons) {
    const [sid, , level, item, ability, tera, moves] = m;
    const rest = [item, ability, ...(moves ? moves.split(',') : []), ...(tera ? [tera] : [])].join(',');
    species += sid in fp;
    full += fpKeys.has(`${sid}|${level},${rest}`);
    noLevel += fpNoLevel.has(`${sid}|${rest}`);
  }
  const p = x => +(100 * x / mons.length).toFixed(3);
  out.coverage[v] = {mons: mons.length, speciesInCachePct: p(species), exactSetInCachePct: p(full),
    setIgnoringLevelInCachePct: p(noLevel)};
}
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out));
