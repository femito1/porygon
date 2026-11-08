// Showdown side of the engine-fidelity spike: whole battles in Showdown's simulator (in-process, fastenv/js/battle.js)
// on the given team pairs, both players driven by the same simple policy as engine_games.py:
//   random  uniform over the legal choices (every usable move, the same with terastallize when allowed, every switch)
//   greedy  the usable move with the highest basePower x type effectiveness against the foe's current types
//           (Showdown's dex, dex_tables.js; ties at random), else uniform random
// Each pair is played REPS times (side one = pair[0]); one JSON line per game:
//   {pair, rep, winner 1/2/0 tie/-2 too long/-1 error, turns, decisions, first_faint (turn of the first faint)}
// The last stderr line is a timing JSON (total seconds, decisions, us per decision, share inside the simulator).
//   node ps_games.js --teams pairs.jsonl --format gen9randombattle --policy random --reps 4 > out.jsonl
const fs = require('fs');
const {SimBattle} = require('../../../fastenv/js/battle');
const {tables, effectiveness} = require('./dex_tables');

const args = {teams: null, format: 'gen9randombattle', policy: 'random', reps: 1, seed: 1, start: 0, count: 1e9,
  maxTurns: 1000, maxDecisions: 2000};
for (let i = 2; i < process.argv.length; i += 2) {
  const k = process.argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  args[k] = /^-?\d+$/.test(process.argv[i + 1]) ? parseInt(process.argv[i + 1], 10) : process.argv[i + 1];
}
const gen = parseInt(args.format.slice(3), 10);
const T = tables(gen);

let rngState = (args.seed * 2654435761) >>> 0 || 1;
function rand() {  // xorshift32, deterministic per run
  rngState ^= rngState << 13; rngState >>>= 0;
  rngState ^= rngState >>> 17;
  rngState ^= rngState << 5; rngState >>>= 0;
  return rngState / 4294967296;
}
const pick = a => a[Math.floor(rand() * a.length)];

function legalChoices(req) {
  const out = [];
  if (req.forceSwitch) {
    req.side.pokemon.forEach((p, i) => { if (!p.active && !p.condition.endsWith(' fnt')) out.push(`switch ${i + 1}`); });
    return out;
  }
  const a = req.active[0];
  a.moves.forEach((m, i) => {
    if (m.disabled) return;
    out.push(`move ${i + 1}`);
    if (a.canTerastallize) out.push(`move ${i + 1} terastallize`);
  });
  if (!a.trapped) {
    req.side.pokemon.forEach((p, i) => { if (!p.active && !p.condition.endsWith(' fnt')) out.push(`switch ${i + 1}`); });
  }
  return out;
}

function moveKey(mon, id) { return id === 'hiddenpower' ? 'hiddenpower' + mon.hpType.toLowerCase() : id; }

function greedy(battle, n, req, choices) {
  if (!req.forceSwitch && req.active) {
    const me = battle.sides[n].active[0], foe = battle.sides[1 - n].active[0];
    const defTypes = foe.getTypes().map(t => t.toLowerCase());
    let best = 0, bestIdx = [];
    req.active[0].moves.forEach((m, i) => {
      if (m.disabled || !choices.includes(`move ${i + 1}`)) return;
      const d = T.moves[moveKey(me, m.id)];
      if (!d || d[2] === 'Status' || !d[0]) return;
      const s = d[0] * effectiveness(T, d[1], defTypes);
      if (s > best) { best = s; bestIdx = [i]; } else if (s === best && s > 0) bestIdx.push(i);
    });
    if (best > 0) return `move ${pick(bestIdx) + 1}`;
  }
  return pick(choices);
}

const lines = fs.readFileSync(args.teams, 'utf8').split('\n').filter(Boolean);
const end = Math.min(lines.length, args.start + args.count);
let simNs = 0n, totalDecisions = 0, sideChoices = 0;
const t0 = process.hrtime.bigint();
for (let pi = args.start; pi < end; pi++) {
  // the slim sets (teams.js) carry speciesId only; Showdown's packer wants species (the nickname defaults to it)
  const pair = JSON.parse(lines[pi]).map(t => t.map(s => ({...s, species: s.speciesId, name: ''})));
  for (let rep = 0; rep < args.reps; rep++) {
    const res = {pair: pi, rep, winner: -1, turns: 0, decisions: 0, first_faint: -1};
    try {
      const a = process.hrtime.bigint();
      const b = new SimBattle(args.format, [pi + 1, rep + 1, args.seed, args.policy === 'greedy' ? 2 : 1], pair,
        {keepRequestJSON: false});
      b.start();
      simNs += process.hrtime.bigint() - a;
      while (!b.ended) {
        if (res.decisions >= args.maxDecisions || b.turn > args.maxTurns) { res.winner = -2; break; }
        const pend = b.pendingSides();
        if (!pend.length) throw new Error('no pending side');
        const turnBefore = b.turn;
        const fainted = b.battle.sides.reduce((s, sd) => s + sd.pokemon.filter(p => p.fainted).length, 0);
        for (const s of pend) {
          const req = b.request(s);
          let choices = legalChoices(req);
          for (;;) {
            const c = !choices.length ? 'default' : args.policy === 'greedy' ? greedy(b.battle, s, req, choices) : pick(choices);
            const x = process.hrtime.bigint();
            const ok = b.choose(s, c);
            simNs += process.hrtime.bigint() - x;
            sideChoices++;
            if (ok || c === 'default') break;
            choices = choices.filter(y => y !== c);  // e.g. a trap the request did not show
          }
        }
        res.decisions++;
        if (res.first_faint < 0 &&
            b.battle.sides.reduce((s, sd) => s + sd.pokemon.filter(p => p.fainted).length, 0) > fainted) {
          res.first_faint = turnBefore;
        }
      }
      res.turns = b.turn;
      if (b.ended) res.winner = b.winner === 'p1' ? 1 : b.winner === 'p2' ? 2 : 0;
    } catch (e) {
      res.winner = -1; res.error = String(e && e.message || e).slice(0, 200);
    }
    totalDecisions += res.decisions;
    process.stdout.write(JSON.stringify(res) + '\n');
  }
}
const secs = Number(process.hrtime.bigint() - t0) / 1e9;
process.stderr.write(JSON.stringify({sim: 'showdown', format: args.format, policy: args.policy, seconds: secs,
  decisions: totalDecisions, side_choices: sideChoices, us_per_decision: 1e6 * secs / totalDecisions,
  sim_us_per_decision: Number(simNs) / 1e3 / totalDecisions}) + '\n');
