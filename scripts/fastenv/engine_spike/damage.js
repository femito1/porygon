// Showdown side of the damage-fidelity check: for each team pair, the two leads at the start of turn 1, every
// damaging move of each lead against the other, Showdown's damage for all 16 rolls without a crit
// (battle.actions.getDamage after the move's ModifyType / ModifyMove events, as useMove runs them).
// Lead entry effects (weather, terrain, Intimidate / Download boosts, Protosynthesis...) are cleared first so both
// simulators see the same neutral field (engine self-play starts with no entry effects).
// With an empty queue Showdown treats the target as having moved already (Payback, Analytic): damage.py has the
// engine attacker move second to match.
// One JSON line per (pair, attacking side, move): {pair, side, move, rolls: [16 damages, max roll first] | null}
//   node damage.js pairs.jsonl gen9randombattle [COUNT] > sd_damage.jsonl
const fs = require('fs');
const {SimBattle} = require('../../../fastenv/js/battle');

const [file, format, count] = [process.argv[2], process.argv[3] || 'gen9randombattle', parseInt(process.argv[4] || '1e9', 10)];
const lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).slice(0, count);
lines.forEach((line, pi) => {
  const pair = JSON.parse(line).map(t => t.map(s => ({...s, species: s.speciesId, name: ''})));
  const b = new SimBattle(format, [pi + 1, 9, 9, 9], pair, {keepRequestJSON: false});
  b.start();
  const battle = b.battle;
  battle.field.clearWeather(); battle.field.clearTerrain();
  const act = [battle.sides[0].active[0], battle.sides[1].active[0]];
  for (const p of act) {
    p.clearBoosts();
    for (const v of Object.keys(p.volatiles)) p.removeVolatile(v);
  }
  for (const side of [0, 1]) {
    const src = act[side], tgt = act[1 - side];
    for (const slot of src.moveSlots) {
      const id = slot.id === 'hiddenpower' ? 'hiddenpower' + src.hpType.toLowerCase() : slot.id;
      const out = {pair: pi, side: side + 1, move: id, category: battle.dex.moves.get(id).category, rolls: null};
      if (out.category !== 'Status') {
        const rolls = [];
        for (let k = 0; k < 16; k++) {
          let move = battle.dex.getActiveMove(id);
          move.willCrit = false;
          battle.activeMove = move; battle.activePokemon = src; battle.activeTarget = tgt;
          let d;
          try {
            battle.singleEvent('ModifyType', move, null, src, tgt, move, move);
            battle.singleEvent('ModifyMove', move, null, src, tgt, move, move);
            move = battle.runEvent('ModifyType', src, tgt, move, move);
            move = battle.runEvent('ModifyMove', src, tgt, move, move);
            move.willCrit = false;
            move.hit = 1;  // first hit (Triple Axel's basePowerCallback reads it)
            battle.randomizer = bd => Math.trunc(Math.trunc(bd * (100 - k)) / 100);
            // absorb / immunity abilities (Water Absorb, Levitate-like onTryHit...) act in TryHit, before getDamage
            d = battle.runEvent('TryHit', tgt, src, move) ? battle.actions.getDamage(src, tgt, move, true) : 0;
          } catch (e) { d = 'error'; }
          for (const v of Object.keys(src.volatiles)) src.removeVolatile(v);
          rolls.push(typeof d === 'number' ? d : d === 'error' ? -1 : 0);
        }
        out.rolls = rolls;
      }
      process.stdout.write(JSON.stringify(out) + '\n');
    }
  }
});
