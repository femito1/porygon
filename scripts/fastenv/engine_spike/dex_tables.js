// Showdown dex tables shared by both simulators' greedy policy (so the policy is the same function in both):
// moves {id: [basePower, type, category]} and types {attacking: {defending: multiplier}} (all lowercase).
//   node dex_tables.js GEN > tables-genN.json        (also require()d by ps_games.js)
const {Dex} = require('../../../fastenv/js/ps');

function tables(gen) {
  const dex = Dex.forGen(gen);
  const moves = {};
  for (const m of dex.moves.all()) moves[m.id] = [m.basePower, m.type.toLowerCase(), m.category];
  const all = dex.types.all().filter(t => !t.isNonstandard);
  for (const t of all) {  // typed hidden power (gen4 sets): Dex resolves 'hiddenpowerfire' to id hiddenpower, type Fire
    const m = dex.moves.get('hiddenpower' + t.id);
    if (m.exists && m.type === t.name) moves['hiddenpower' + t.id] = [m.basePower, t.id, m.category];
  }
  const types = {};
  for (const a of all) {
    const row = types[a.id] = {};
    for (const d of all) {
      const code = d.damageTaken[a.name];
      row[d.id] = code === 1 ? 2 : code === 2 ? 0.5 : code === 3 ? 0 : 1;
    }
  }
  return {moves, types};
}

function effectiveness(t, moveType, defTypes) {
  let m = 1;
  const row = t.types[moveType];
  if (!row) return 1;
  for (const d of defTypes) if (d in row) m *= row[d];
  return m;
}

module.exports = {tables, effectiveness};
if (require.main === module) process.stdout.write(JSON.stringify(tables(parseInt(process.argv[2] || '9', 10))));
