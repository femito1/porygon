// fastenv host: B in-process Showdown battles in one Node process, stepped in batches by Python
// (porygon.fastenv). See .orchestra/CONTRACTS.md "Environment API" for the row semantics.
//
//   node host.js --format gen9randombattle --battles 64 --seed 1 --encoder v1 --fd-in 3 --fd-out 4
//
// Wire protocol (little-endian; fd-in / fd-out are pipes, stderr is for logs):
//   Python -> host: u32 cmd, u32 n, then n x (u32 slot, i32 action).  cmd 1 = STEP, 2 = CLOSE.
//                   STEP answers every decision row of the previous batch (any order).
//   host -> Python: first a hello: u32 MAGIC_HELLO, u32 len, JSON {gen, n_actions, format, encoder, columns:
//     [[name, dtype, shape], ...]}; then one batch per STEP (and one at start-up): header u32 x 4 (MAGIC, n_rows,
//     n_columns, n_actions), then every column in hello order, n_rows x prod(shape) elements each.
//     Columns: reward f32, slot u32, turn u16, flags u8, win i8, mask u8[A], opp_prev i16[4],
//     opp_team i16[6,8], opp_revealed u8[6], the encoder's observation ('obs' f32[OBS_DIM] for v1; 'obs/<field>'
//     for multi-field encoders) and its aux targets ('aux/<field>').
//   slot = battle * 2 + side (side 0 = p1). flags: 1 needs_action, 2 done (terminal row), 4 first decision of a
//   battle, 8 the previous action of this slot was rejected by Showdown and replaced by 'default'.
// --duplicate 1: every slot plays duplicate pairs (game 2k+1 replays game 2k's seed with the teams swapped), so
//   side 0 plays both teams of each pair: for head-to-head evaluation (porygon.fastenv_eval).
'use strict';
const fs = require('fs');
const path = require('path');
const {SimBattle, Teams} = require('./battle');
const {PRNG} = require('./ps');
const aux = require('./aux');

const MAGIC = 0x50594746, MAGIC_HELLO = 0x4c454846;  // 'FGYP', 'FHEL'

function parseArgs(argv) {
  const a = {format: 'gen9randombattle', battles: 64, seed: 1, encoder: 'v1', fdIn: 0, fdOut: 1,
    shaping: '0.05,0.05,0', victory: 1, maxTurns: 1000, duplicate: 0};
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '').replace(/-(.)/g, (_, c) => c.toUpperCase());
    a[k] = argv[i + 1];
  }
  a.battles = +a.battles; a.seed = +a.seed; a.fdIn = +a.fdIn; a.fdOut = +a.fdOut;
  a.victory = +a.victory; a.maxTurns = +a.maxTurns; a.duplicate = +a.duplicate;
  const [f, h, s] = String(a.shaping).split(',').map(Number);
  a.shapingW = {fainted: f, hp: h, status: s, victory: a.victory};
  return a;
}

const args = parseArgs(process.argv);
console.log = (...x) => process.stderr.write(x.join(' ') + '\n');  // never corrupt the binary stream
const E = require(args.encoder === 'stub' ? './encoder_stub' : `./encoder_${args.encoder}`);
const {SideTracker} = require(args.encoder === 'stub' ? './encoder_stub' : './tracker');
const GEN = +args.format.match(/^gen(\d)/)[1];
const D = E.OBS_DIM, A = E.nActions(GEN);
const prng = new PRNG([args.seed & 0xffff, (args.seed >>> 16) & 0xffff, 0x5eed, 0xfa57]);

// --- battles ---------------------------------------------------------------------------------------
class Slot {  // one battle position: the current battle, its trackers and per-side bookkeeping
  constructor(i) {
    this.i = i;
    this.games = 0;
    this.newBattle();
  }

  newBattle() {
    const seed = () => [0, 0, 0, 0].map(() => prng.random(0x10000));
    let teams, battleSeed;
    if (args.duplicate && this.games % 2 === 1) {  // second game of a duplicate pair: same seed, teams swapped
      teams = [this.pairTeams[1], this.pairTeams[0]];
      battleSeed = this.pairSeed;
    } else {
      teams = [Teams.generate(args.format, {seed: seed()}), Teams.generate(args.format, {seed: seed()})];
      battleSeed = seed();
      // keep pristine copies: the simulator may annotate the set objects it is given
      this.pairTeams = teams.map(t => JSON.parse(JSON.stringify(t)));
      this.pairSeed = battleSeed;
    }
    this.trackers = [new SideTracker({format: args.format, side: 0}), new SideTracker({format: args.format, side: 1})];
    this.value = [0, 0];       // last state value per side (reward = difference, as poke-env's helper)
    this.lastAction = [-1, -1];
    this.choice = [null, null];
    this.rejected = [false, false];
    this.first = [true, true];
    this.crashed = false;
    this.oppPrev = [[-1, 0, 0, 0], [-1, 0, 0, 0]];
    this.auxTeams = [aux.teamIds(teams[0], GEN), aux.teamIds(teams[1], GEN)];
    this.battle = new SimBattle(args.format, battleSeed, teams.map(t => JSON.parse(JSON.stringify(t))), {
      keepRequestJSON: false,
      onLog: (n, text) => this.trackers[n].onLog(text),
      onRequest: (n, req) => this.trackers[n].onRequest(req),
      onError: (n, text) => this.trackers[n].onError(text),
    });
    this.battle.start();
    this.games++;
  }
}

const slots = Array.from({length: args.battles}, (_, i) => new Slot(i));
const stats = {decisions: 0, rejected: 0, games: 0, turns: 0, truncated: 0, crashed: 0};

// --- output batch ----------------------------------------------------------------------------------
// Columns: env columns, the encoder's observation fields, the encoder's aux fields (if any). Each column is
// [name, dtype, width (elements per row), shape (per row)]. Sent once at start-up (hello); every batch then has
// the columns in this order, each n_rows * width elements, ordered by element size (so all are aligned).
const DT = {f32: Float32Array, i32: Int32Array, u32: Uint32Array, i16: Int16Array, u16: Uint16Array,
  u8: Uint8Array, i8: Int8Array};
const OBS_FIELDS = E.fields ? E.fields(GEN) : [['obs', 'f32', [E.OBS_DIM]]];
const AUX_FIELDS = E.auxFields ? E.auxFields(GEN) : [];
const prod = shape => shape.reduce((x, y) => x * y, 1);
const COLUMNS = [
  ['reward', 'f32', [1]], ['slot', 'u32', [1]], ['turn', 'u16', [1]], ['flags', 'u8', [1]], ['win', 'i8', [1]],
  ['mask', 'u8', [A]], ['opp_prev', 'i16', [4]], ['opp_team', 'i16', [6, 8]], ['opp_revealed', 'u8', [6]],
  ...OBS_FIELDS.map(([k, d, sh]) => [OBS_FIELDS.length === 1 ? k : `obs/${k}`, d, sh]),
  ...AUX_FIELDS.map(([k, d, sh]) => [`aux/${k}`, d, sh]),
].map(([name, dtype, shape]) => ({name, dtype, shape, width: prod(shape), T: DT[dtype]}))
  .sort((a, b) => b.T.BYTES_PER_ELEMENT - a.T.BYTES_PER_ELEMENT);
const COL = Object.fromEntries(COLUMNS.map(c => [c.name, c]));
const obsCols = OBS_FIELDS.map(([k]) => COL[OBS_FIELDS.length === 1 ? k : `obs/${k}`]);
const auxCols = AUX_FIELDS.map(([k]) => COL[`aux/${k}`]);

let cap = 0, buf, V = {};  // V[name]: typed view of capacity cap * width
function ensure(need) {
  if (need <= cap) return;
  const old = V, oldCap = cap;
  cap = Math.max(need, cap * 2, 64);
  buf = Buffer.alloc(16 + COLUMNS.reduce((t, c) => t + cap * c.width * c.T.BYTES_PER_ELEMENT, 0));
  let o = 16;
  V = {};
  for (const c of COLUMNS) {
    V[c.name] = new c.T(buf.buffer, buf.byteOffset + o, cap * c.width);
    o += cap * c.width * c.T.BYTES_PER_ELEMENT;
    if (oldCap) V[c.name].set(old[c.name].subarray(0, n * c.width));
  }
}
const obsOut = () => Object.fromEntries(OBS_FIELDS.map(([k], i) => [k, V[obsCols[i].name]]));
const auxOut = () => Object.fromEntries(AUX_FIELDS.map(([k], i) => [k, V[auxCols[i].name]]));

let n = 0;
function row(slot, side, needsAction, done) {
  ensure(n + 1);
  const t = slot.trackers[side];
  const v = E.stateValue(t, args.shapingW);
  V.reward[n] = v - slot.value[side];
  slot.value[side] = v;
  V.slot[n] = slot.i * 2 + side;
  let f = (needsAction ? 1 : 0) | (done ? 2 : 0);
  if (slot.first[side] && needsAction) { f |= 4; slot.first[side] = false; }
  if (slot.rejected[side]) { f |= 8; slot.rejected[side] = false; }
  V.flags[n] = f;
  V.turn[n] = Math.min(slot.battle.turn, 65535);
  const w = slot.battle.winner;
  V.win[n] = !done || slot.crashed ? 0 : w === (side ? 'p2' : 'p1') ? 1 : w === (side ? 'p1' : 'p2') ? -1 : 0;
  const op = slot.oppPrev[side];
  for (let k = 0; k < 4; k++) V.opp_prev[n * 4 + k] = op[k];
  slot.oppPrev[side] = [-1, 0, 0, 0];
  aux.writeOpponentTeam(t, slot.auxTeams[1 - side], V.opp_team, n * 48, V.opp_revealed, n * 6);
  const A_ = A;
  if (needsAction) {
    if (E.fields) E.encodeFields(t, obsOut(), n);
    else E.encode(t, V[obsCols[0].name], n * obsCols[0].width);
    E.actionMask(t, V.mask, n * A_);
    if (E.encodeAux) E.encodeAux(t, slot.trackers[1 - side], slot.battle.teams[1 - side], auxOut(), n);
  } else {
    for (const c of [...obsCols, ...auxCols]) V[c.name].fill(0, n * c.width, (n + 1) * c.width);
    V.mask.fill(0, n * A_, (n + 1) * A_);
  }
  n++;
}

function emitSlot(slot) {
  const b = slot.battle;
  if (!b.ended && !slot.crashed && b.turn >= args.maxTurns) {  // endless battle: call it a tie
    b.battle.tie();
    stats.truncated++;
  }
  if (b.ended || slot.crashed) {
    stats.games++;
    stats.turns += b.turn;
    row(slot, 0, false, true);
    row(slot, 1, false, true);
    slot.newBattle();
  }
  for (const side of [0, 1]) if (slot.trackers[side].decisionPending) row(slot, side, true, false);
}

function flush() {
  ensure(1);
  const h = new Uint32Array(buf.buffer, buf.byteOffset, 4);
  h[0] = MAGIC; h[1] = n; h[2] = COLUMNS.length; h[3] = A;
  // columns were laid out for `cap` rows: compact them to n rows in place (same order, smaller strides)
  let o = 16;
  for (const c of COLUMNS) {
    const arr = V[c.name], bytes = n * c.width * c.T.BYTES_PER_ELEMENT;
    buf.copy(buf, o, arr.byteOffset - buf.byteOffset, arr.byteOffset - buf.byteOffset + bytes);
    o += bytes;
  }
  writeAll(buf, o);
  n = 0;
}

function hello() {
  const json = Buffer.from(JSON.stringify({gen: GEN, n_actions: A, format: args.format, encoder: args.encoder,
    columns: COLUMNS.map(c => [c.name, c.dtype, c.shape])}));
  const h = Buffer.alloc(8);
  h.writeUInt32LE(MAGIC_HELLO, 0); h.writeUInt32LE(json.length, 4);
  writeAll(h, 8); writeAll(json, json.length);
}

function writeAll(b, len) {
  let off = 0;
  while (off < len) {
    try {
      off += fs.writeSync(args.fdOut, b, off, len - off);
    } catch (e) {
      if (e.code === 'EPIPE') finish();  // Python closed its end: shut down
      if (e.code !== 'EAGAIN') throw e;
    }
  }
}

const hdr = Buffer.alloc(8);
let inBuf = Buffer.alloc(8 * 1024);
function readExact(b, len) {
  let off = 0;
  while (off < len) {
    let k;
    try { k = fs.readSync(args.fdIn, b, off, len - off, null); } catch (e) { if (e.code === 'EAGAIN') continue; throw e; }
    if (k === 0) return false;
    off += k;
  }
  return true;
}

// --- main loop -------------------------------------------------------------------------------------
hello();
for (const s of slots) emitSlot(s);
flush();
const touched = new Set();
for (;;) {
  if (!readExact(hdr, 8)) break;
  const cmd = hdr.readUInt32LE(0), m = hdr.readUInt32LE(4);
  if (cmd === 2) break;
  if (inBuf.length < 8 * m) inBuf = Buffer.alloc(8 * m * 2);
  if (!readExact(inBuf, 8 * m)) break;
  touched.clear();
  for (let k = 0; k < m; k++) {
    const s = inBuf.readUInt32LE(8 * k), action = inBuf.readInt32LE(8 * k + 4);
    const slot = slots[s >> 1], side = s & 1;
    slot.choice[side] = E.actionToChoice(slot.trackers[side], action);
    slot.lastAction[side] = action;
    touched.add(slot);
  }
  for (const slot of touched) {
    const b = slot.battle;
    // both sides' choices of this turn are known: record each side's view of the opponent's choice (aux label)
    for (const side of [0, 1]) {
      if (slot.choice[side] === null) continue;
      const opp = 1 - side;
      if (slot.choice[opp] !== null) {
        slot.oppPrev[side] = aux.choiceLabel(slot.trackers[opp], slot.lastAction[opp], slot.choice[opp], b, opp);
      }
    }
    for (const side of [0, 1]) {
      const c = slot.choice[side];
      if (c === null) continue;
      slot.trackers[side].decisionPending = false;
      stats.decisions++;
      let ok;
      try {
        ok = b.choose(side, c);
      } catch (e) {  // a simulator exception ("Infinite loop", crash): the battle is void, scored as a tie
        stats.crashed++;
        if (stats.crashed <= 5) process.stderr.write(`fastenv: battle crashed: ${e.stack.split('\n').slice(0, 3).join(' | ')}\n`);
        slot.crashed = true;
        break;
      }
      if (!ok) {
        stats.rejected++;
        slot.rejected[side] = true;
        // [Unavailable choice] (hidden information, e.g. an unrevealed trapping ability) comes with an updated
        // request: the tracker wants a new decision, which goes back to the policy. Otherwise play 'default'.
        if (!slot.trackers[side].decisionPending && !b.ended && b.pendingSides().includes(side)) {
          b.choose(side, 'default');
        }
      }
    }
    slot.choice[0] = slot.choice[1] = null;
    emitSlot(slot);
  }
  flush();
}
finish();

function finish() {
  if (process.env.PORYGON_FASTENV_VERBOSE) process.stderr.write(`host done: ${JSON.stringify(stats)}\n`);
  process.exit(0);
}
