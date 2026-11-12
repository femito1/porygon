// Team pairs from Showdown's random-battle generator, one JSON line [team, team] each, in scripts/gen_teams.js's
// format (that script requires the repo's pokemon-showdown submodule, empty in a worktree; this one finds the
// dist through fastenv/js/ps.js). A seed makes the pairs reproducible.
//   node teams.js PAIRS FORMAT [SEED] > pairs.jsonl
const {Teams} = require('../../../fastenv/js/ps');

const n = parseInt(process.argv[2] || '1000', 10);
const format = process.argv[3] || 'gen9randombattle';
const seed = parseInt(process.argv[4] || '1', 10);
const keep = ['speciesId', 'level', 'moves', 'ability', 'item', 'evs', 'ivs', 'role', 'teraType'];
const slim = team => team.map(s => Object.fromEntries(keep.map(k => [k, s[k]])));
for (let i = 0; i < n; i++) {
  const gen = j => Teams.getGenerator(format, [seed, i, j, 0x5eed]).getTeam();
  process.stdout.write(JSON.stringify([slim(gen(0)), slim(gen(1))]) + '\n');
}
