// Static diff of the random-battle set files of two Showdown checkouts (data/random-battles/GEN/sets.json: the
// per-species level table and each role's movepool / abilities / tera types).
//   node static_diff.js OLD_PS NEW_PS GEN OUT.json      (GEN = gen9 | gen4)
// Prints a summary; OUT.json holds every per-species change. Sets are matched by role (by position when a species
// repeats a role). Move / ability / tera names are compared as ids.
'use strict';
const fs = require('fs');
const path = require('path');

const [OLD, NEW, GEN, OUT] = process.argv.slice(2);
if (!OUT) {
  console.error('usage: node static_diff.js OLD_PS NEW_PS GEN OUT.json');
  process.exit(2);
}
const toID = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
const load = ps => JSON.parse(fs.readFileSync(path.join(ps, 'data/random-battles', GEN, 'sets.json'), 'utf8'));
const a = load(OLD), b = load(NEW);

function keyed(sets) {  // role -> set, "role#2" for a repeated role
  const out = {}, seen = {};
  for (const s of sets || []) {
    const r = s.role || '?';
    seen[r] = (seen[r] || 0) + 1;
    out[seen[r] > 1 ? `${r}#${seen[r]}` : r] = s;
  }
  return out;
}
const diffList = (x, y) => {
  const X = new Set((x || []).map(toID)), Y = new Set((y || []).map(toID));
  return {added: [...Y].filter(e => !X.has(e)).sort(), removed: [...X].filter(e => !Y.has(e)).sort()};
};
const nonEmpty = d => d.added.length + d.removed.length > 0;
const FIELDS = ['movepool', 'abilities', 'teraTypes'];

const out = {gen: GEN, old: OLD, new: NEW, added: [], removed: [], changed: {}, keys: {}};
for (const f of [a, b]) for (const e of Object.values(f)) {
  for (const k of Object.keys(e)) out.keys[`species.${k}`] = 1;
  for (const s of e.sets || []) for (const k of Object.keys(s)) out.keys[`set.${k}`] = 1;
}
out.keys = Object.keys(out.keys).sort();
out.added = Object.keys(b).filter(s => !(s in a)).sort();
out.removed = Object.keys(a).filter(s => !(s in b)).sort();
for (const sp of Object.keys(b).filter(s => s in a).sort()) {
  const ea = a[sp], eb = b[sp], ch = {};
  if (ea.level !== eb.level) ch.level = [ea.level, eb.level];
  for (const k of new Set([...Object.keys(ea), ...Object.keys(eb)])) {  // other species-level fields
    if (k !== 'level' && k !== 'sets' && JSON.stringify(ea[k]) !== JSON.stringify(eb[k])) ch[k] = [ea[k], eb[k]];
  }
  const ka = keyed(ea.sets), kb = keyed(eb.sets);
  const rolesAdded = Object.keys(kb).filter(r => !(r in ka)), rolesRemoved = Object.keys(ka).filter(r => !(r in kb));
  if (rolesAdded.length) ch.rolesAdded = rolesAdded;
  if (rolesRemoved.length) ch.rolesRemoved = rolesRemoved;
  for (const r of Object.keys(kb).filter(r => r in ka)) {
    const sd = {};
    for (const f of FIELDS) {
      const d = diffList(ka[r][f], kb[r][f]);
      if (nonEmpty(d)) sd[f] = d;
    }
    for (const k of new Set([...Object.keys(ka[r]), ...Object.keys(kb[r])])) {  // any other set field
      if (!FIELDS.includes(k) && k !== 'role' && JSON.stringify(ka[r][k]) !== JSON.stringify(kb[r][k])) {
        sd[k] = [ka[r][k], kb[r][k]];
      }
    }
    if (Object.keys(sd).length) (ch.sets ??= {})[r] = sd;
  }
  if (JSON.stringify(Object.keys(ka)) !== JSON.stringify(Object.keys(kb)) && !rolesAdded.length &&
      !rolesRemoved.length) ch.roleOrder = [Object.keys(ka), Object.keys(kb)];
  if (Object.keys(ch).length) out.changed[sp] = ch;
}

const ch = Object.entries(out.changed);
const count = pred => ch.filter(([, c]) => pred(c)).length;
const setField = f => count(c => c.sets && Object.values(c.sets).some(s => s[f]));
out.summary = {
  speciesOld: Object.keys(a).length, speciesNew: Object.keys(b).length,
  added: out.added.length, removed: out.removed.length, changed: ch.length,
  levelChanged: count(c => c.level), levelOnly: count(c => c.level && Object.keys(c).length === 1),
  rolesAddedOrRemoved: count(c => c.rolesAdded || c.rolesRemoved),
  movepoolChanged: setField('movepool'), abilitiesChanged: setField('abilities'), teraChanged: setField('teraTypes'),
  levelDeltas: ch.filter(([, c]) => c.level).map(([s, c]) => `${s} ${c.level[0]}->${c.level[1]}`),
};
fs.writeFileSync(OUT, JSON.stringify(out, null, 1));
console.log(`${GEN} sets.json: ${JSON.stringify(out.summary)}`);
for (const [sp, c] of ch.filter(([, c]) => Object.keys(c).some(k => k !== 'level'))) {
  console.log(`  ${sp}: ${JSON.stringify(c)}`);
}
