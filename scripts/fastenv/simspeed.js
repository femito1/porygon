// Raw Showdown simulator speed, in-process, random legal play: node scripts/fastenv/simspeed.js FORMAT GAMES
const PS = require('path').join(require('../../fastenv/js/ps').PS_DIR, 'dist/sim');
const {Battle, Teams, PRNG} = require(PS);
const format = process.argv[2] || 'gen9randombattle';
const N = +(process.argv[3] || 200);
// warm
let tTeam = 0, tSim = 0, turns = 0, decisions = 0, logBytes = 0;
function randChoice(b, side) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) {
    const opts = side.pokemon.map((p,i)=>[p,i]).filter(([p,i])=>!p.fainted && !p.isActive).map(([p,i])=>i+1);
    return opts.length ? 'switch ' + opts[Math.floor(Math.random()*opts.length)] : 'pass';
  }
  if (req.teamPreview) return 'default';
  const a = req.active[0];
  const opts = [];
  a.moves.forEach((m,i)=>{ if(!m.disabled) opts.push('move '+(i+1)); });
  if (!a.trapped && !a.maybeTrapped) side.pokemon.forEach((p,i)=>{ if(!p.fainted && !p.isActive) opts.push('switch '+(i+1)); });
  if (a.canTerastallize && Math.random()<0.1) return opts.filter(o=>o.startsWith('move')).map(o=>o+' terastallize')[0] || opts[0];
  return opts.length ? opts[Math.floor(Math.random()*opts.length)] : 'default';
}
for (let g = 0; g < N + 20; g++) {
  const t0 = process.hrtime.bigint();
  const p1 = Teams.generate(format), p2 = Teams.generate(format);
  const t1 = process.hrtime.bigint();
  const b = new Battle({formatid: format, seed: undefined});
  b.setPlayer('p1', {name: 'a', team: Teams.pack(p1)});
  b.setPlayer('p2', {name: 'b', team: Teams.pack(p2)});
  let guard = 0;
  while (!b.ended && guard++ < 1000) {
    for (const side of b.sides) {
      const c = randChoice(b, side);
      if (c && !side.isChoiceDone()) { if (!b.choose(side.id, c)) b.choose(side.id, 'default'); if (g>=20) decisions++; }
    }
  }
  const t2 = process.hrtime.bigint();
  if (g >= 20) { tTeam += Number(t1-t0); tSim += Number(t2-t1); turns += b.turn; logBytes += b.log.join('\n').length; }
}
console.log(format, `games ${N}: team gen ${(tTeam/N/1e6).toFixed(2)} ms/game, sim ${(tSim/N/1e6).toFixed(2)} ms/game, turns/game ${(turns/N).toFixed(1)}, decisions/game ${(decisions/N).toFixed(1)}, sim us/decision ${(tSim/decisions/1e3).toFixed(1)}, log KB/game ${(logBytes/N/1024).toFixed(1)}`);
console.log(`=> ${(decisions/((tSim+tTeam)/1e9)).toFixed(0)} decisions/s/core (sim+teams)`);
