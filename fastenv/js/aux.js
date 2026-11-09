// Auxiliary training targets known only to the environment: the opponent's true team (hidden sets) and the
// opponent's actual choice. Ids are porygon's vocab ids (src/porygon/data/vocab.json, same rule as
// porygon.encoder.idx: 0 = none/padding, 1 = unknown, i + 2 = vocab entry i); tera types are 1 + the index in
// TYPES (encoder order: Bug Dark Dragon Electric Fairy Fighting Fire Flying Ghost Grass Ground Ice Normal Poison
// Psychic Rock Steel Water), 0 = none.
'use strict';
const fs = require('fs');
const path = require('path');
const {toID} = require('./ps');

const VOCAB = JSON.parse(fs.readFileSync(path.join(__dirname, '../../src/porygon/data/vocab.json'), 'utf8'));
const INDEX = {};
for (const [k, list] of Object.entries(VOCAB)) {
  INDEX[k] = new Map(list.map((e, i) => [e, i + 2]));
}
const TYPES = ['bug', 'dark', 'dragon', 'electric', 'fairy', 'fighting', 'fire', 'flying', 'ghost', 'grass',
  'ground', 'ice', 'normal', 'poison', 'psychic', 'rock', 'steel', 'water'];

function vocabId(kind, key) {
  if (key === null || key === undefined) return 1;
  if (key === '') return 0;
  const t = INDEX[kind];
  if (t.has(key)) return t.get(key);
  if (kind === 'moves' && key.startsWith('hiddenpower')) return t.get('hiddenpower');
  return t.get(key.replace(/[0-9]+$/, '')) ?? 1;
}

/** One team's true sets: per mon {name, ids: Int16Array(8) [species, item, ability, move x4, tera]}. */
function teamIds(team, gen) {
  return team.map(set => {
    const ids = new Int16Array(8);
    ids[0] = vocabId('species', toID(set.species));
    ids[1] = vocabId('items', toID(set.item));
    ids[2] = vocabId('abilities', toID(set.ability));
    for (let k = 0; k < 4; k++) ids[3 + k] = set.moves[k] ? vocabId('moves', toID(set.moves[k])) : 0;
    ids[7] = gen >= 9 && set.teraType ? 1 + TYPES.indexOf(toID(set.teraType)) : 0;
    return {name: set.name || set.species, ids};
  });
}

/** Opponent's true team in the observer's opponent-token order: the tokens the tracker has (revealed mons, in
 *  poke-env opponent_team order) first, then the unrevealed mons in their team order. revealed[k] = 1 for tokens.
 *  Illusion: a token is matched by the name it was revealed under. */
function writeOpponentTeam(tracker, trueTeam, out, off, revealed, roff) {
  const idents = tracker.opponentIdents ? tracker.opponentIdents() : [];
  const used = new Uint8Array(trueTeam.length);
  let k = 0;
  for (const ident of idents) {
    if (k >= 6) break;
    const name = ident.slice(ident.indexOf(': ') + 2);
    let j = trueTeam.findIndex((m, i) => !used[i] && m.name === name);
    if (j < 0) j = trueTeam.findIndex((m, i) => !used[i]);
    used[j] = 1;
    out.set(trueTeam[j].ids, off + 8 * k);
    revealed[roff + k] = 1;
    k++;
  }
  for (let j = 0; j < trueTeam.length && k < 6; j++) {
    if (used[j]) continue;
    out.set(trueTeam[j].ids, off + 8 * k);
    revealed[roff + k] = 0;
    k++;
  }
  for (; k < 6; k++) { out.fill(0, off + 8 * k, off + 8 * k + 8); revealed[roff + k] = 0; }
}

/** The opponent's choice as a label for the observer: [action index in the opponent's own action space,
 *  move vocab id (0 if a switch), switched-in species vocab id (0 if a move), terastallized 0/1]. */
function choiceLabel(oppTracker, action, choice, simBattle, oppSide) {
  const req = simBattle.request(oppSide);
  const out = [action, 0, 0, 0];
  if (!req || !choice) return out;
  const m = /^(move|switch) (\d+)( terastallize)?/.exec(choice);
  if (!m) return out;
  const k = +m[2] - 1;
  if (m[1] === 'move' && req.active && req.active[0] && req.active[0].moves[k]) {
    out[1] = vocabId('moves', req.active[0].moves[k].id);
    out[3] = m[3] ? 1 : 0;
  } else if (m[1] === 'switch' && req.side && req.side.pokemon[k]) {
    out[2] = vocabId('species', toID(req.side.pokemon[k].details.split(',')[0]));
  }
  return out;
}

module.exports = {vocabId, teamIds, writeOpponentTeam, choiceLabel, TYPES};
