// Team pairs from Showdown's own random-battle generator, one JSON line [team, team] each, for engine-only
// self-play (`porygon.selfplay`). Each team is Showdown's PokemonSet list, the lead first.
// Usage: node scripts/gen_teams.js PAIRS [FORMAT=gen4randombattle] > teams.jsonl   (after `node build` in pokemon-showdown/)
const {Teams} = require('../pokemon-showdown/dist/sim');

const n = parseInt(process.argv[2] || '1000', 10);
const format = process.argv[3] || 'gen4randombattle';
const keep = ['speciesId', 'level', 'moves', 'ability', 'item', 'evs', 'ivs', 'role', 'teraType'];
const slim = team => team.map(s => Object.fromEntries(keep.map(k => [k, s[k]])));
for (let i = 0; i < n; i++) {
  process.stdout.write(JSON.stringify([slim(Teams.generate(format)), slim(Teams.generate(format))]) + '\n');
}
