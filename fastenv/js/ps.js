// Locates Pokémon Showdown's compiled simulator (dist/sim). PORYGON_PS=<pokemon-showdown checkout> overrides;
// otherwise the repo's submodule, then the main checkout two levels up (git worktrees have an empty submodule).
const fs = require('fs');
const path = require('path');

function findPS() {
  const cands = [process.env.PORYGON_PS, path.join(__dirname, '../../pokemon-showdown'),
    path.join(__dirname, '../../../../../pokemon-showdown')].filter(Boolean);
  for (const c of cands) if (fs.existsSync(path.join(c, 'dist/sim/index.js'))) return path.resolve(c);
  throw new Error(`no built pokemon-showdown found (set PORYGON_PS); tried ${cands.join(', ')}`);
}
const PS_DIR = findPS();
module.exports = {PS_DIR, ...require(path.join(PS_DIR, 'dist/sim'))};
