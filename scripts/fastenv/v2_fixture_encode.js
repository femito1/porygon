// encoder_v2 on netv2 parity fixtures (porygon.v2.parity): reads one JSON job per line on stdin
//   {format, p1, p2, chunks_p1, chunks_p2, dec: [[k1, k2], ...]}   (chunks = poke-env split websocket messages)
// replays p1's first k1 and p2's first k2 chunks into SideTrackers (usernames from the fixture) and writes one JSON
// line per decision: {o2: {field: [b64 uint16 idx, b64 values]}, a2: {...}} (aux vs p2's tracker).
// Used by scripts/fastenv/parity_v2.py --fixture.
'use strict';
const path = require('path');
const readline = require('readline');
const JS = path.join(__dirname, '../../fastenv/js');
const {SideTracker} = require(path.join(JS, 'tracker.js'));
const E2 = require(path.join(JS, 'encoder_v2.js'));

function sparseTyped(a) {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== 0) n++;
  const ix = new Uint16Array(n), vs = new a.constructor(n);
  for (let i = 0, k = 0; i < a.length; i++) if (a[i] !== 0) { ix[k] = i; vs[k++] = a[i]; }
  return [Buffer.from(ix.buffer).toString('base64'), Buffer.from(vs.buffer, vs.byteOffset, vs.byteLength).toString('base64')];
}
const text = chunk => chunk.map(l => l.join('|')).join('\n');

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
let ns = 0n, n = 0;
rl.on('line', line => {
  if (!line) return;
  const job = JSON.parse(line);
  const gen = parseInt(job.format[3], 10);
  const t1 = new SideTracker({format: job.format, side: 0, username: job.p1});
  const t2 = new SideTracker({format: job.format, side: 1, username: job.p2});
  const o2 = E2.allocate(E2.fields(gen), 1), a2 = E2.allocate(E2.auxFields(gen), 1);
  let i1 = 0, i2 = 0;
  for (const [k1, k2] of job.dec) {
    while (i1 < k1) t1.onLog(text(job.chunks_p1[i1++]));
    while (i2 < k2) t2.onLog(text(job.chunks_p2[i2++]));
    const t0 = process.hrtime.bigint();
    E2.encodeFields(t1, o2, 0);
    ns += process.hrtime.bigint() - t0; n++;
    E2.encodeAux(t1, t2, null, a2, 0);
    const r = {o2: {}, a2: {}};
    for (const k in o2) r.o2[k] = sparseTyped(o2[k]);
    for (const k in a2) r.a2[k] = sparseTyped(a2[k]);
    process.stdout.write(JSON.stringify(r) + '\n');
  }
});
rl.on('close', () => console.error(`v2_fixture_encode: ${n} decisions, encode ${(Number(ns) / 1e3 / Math.max(n, 1)).toFixed(1)} us/call`));
