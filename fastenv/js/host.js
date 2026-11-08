// fastenv host: B in-process Showdown battles in one Node process, stepped in batches by Python
// (porygon.fastenv). See .orchestra/CONTRACTS.md "Environment API" for the row semantics.
//
//   node host.js --format gen9randombattle --battles 64 --seed 1 --encoder v1 --fd-in 3 --fd-out 4
//
// Wire protocol (little-endian; fd-in / fd-out are pipes, stderr is for logs):
//   Python -> host: u32 cmd, u32 n, then n x (u32 slot, i32 action).  cmd 1 = STEP, 2 = CLOSE.
//                   STEP answers every decision row of the previous batch (any order).
//   host -> Python: one batch per STEP (and one at start-up):
//     header u32 x 4: MAGIC, n_rows, obs_dim, n_actions
//     columns, in this order: obs f32[n*obs_dim], reward f32[n], slot u32[n], opp_team i16[n*6*8],
//     opp_prev i16[n*4], turn u16[n], mask u8[n*n_actions], flags u8[n], win i8[n], opp_revealed u8[n*6]
//   slot = battle * 2 + side (side 0 = p1). flags: 1 needs_action, 2 done (terminal row), 4 first decision of a
//   battle, 8 the previous action of this slot was rejected by Showdown and replaced by 'default'.
'use strict';
const fs = require('fs');
const path = require('path');
const {SimBattle, Teams} = require('./battle');
const {PRNG} = require('./ps');
const aux = require('./aux');

const MAGIC = 0x50594746;  // 'FGYP'

function parseArgs(argv) {
  const a = {format: 'gen9randombattle', battles: 64, seed: 1, encoder: 'v1', fdIn: 0, fdOut: 1,
    shaping: '0.05,0.05,0', victory: 1, maxTurns: 1000};
  for (let i = 2; i < argv.length; i += 2) {
    const k = argv[i].replace(/^--/, '').replace(/-(.)/g, (_, c) => c.toUpperCase());
    a[k] = argv[i + 1];
  }
  a.battles = +a.battles; a.seed = +a.seed; a.fdIn = +a.fdIn; a.fdOut = +a.fdOut;
  a.victory = +a.victory; a.maxTurns = +a.maxTurns;
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
    const teams = [Teams.generate(args.format, {seed: seed()}), Teams.generate(args.format, {seed: seed()})];
    this.trackers = [new SideTracker({format: args.format, side: 0}), new SideTracker({format: args.format, side: 1})];
    this.value = [0, 0];       // last state value per side (reward = difference, as poke-env's helper)
    this.lastAction = [-1, -1];
    this.choice = [null, null];
    this.rejected = [false, false];
    this.first = [true, true];
    this.crashed = false;
    this.oppPrev = [[-1, 0, 0, 0], [-1, 0, 0, 0]];
    this.auxTeams = [aux.teamIds(teams[0], GEN), aux.teamIds(teams[1], GEN)];
    this.battle = new SimBattle(args.format, seed(), teams, {
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
let cap = 0, buf, obs, reward, slotA, oppTeam, oppPrev, turnA, mask, flags, win, revealed;
function ensure(n) {
  if (n <= cap) return;
  cap = Math.max(n, cap * 2, 64);
  const sizes = [4 * cap * D, 4 * cap, 4 * cap, 2 * cap * 48, 2 * cap * 4, 2 * cap, cap * A, cap, cap, cap * 6];
  buf = Buffer.alloc(16 + sizes.reduce((x, y) => x + y, 0));
  let o = 16;
  const view = (T, len) => { const v = new T(buf.buffer, buf.byteOffset + o, len); o += len * T.BYTES_PER_ELEMENT; return v; };
  obs = view(Float32Array, cap * D); reward = view(Float32Array, cap); slotA = view(Uint32Array, cap);
  oppTeam = view(Int16Array, cap * 48); oppPrev = view(Int16Array, cap * 4); turnA = view(Uint16Array, cap);
  mask = view(Uint8Array, cap * A); flags = view(Uint8Array, cap); win = view(Int8Array, cap);
  revealed = view(Uint8Array, cap * 6);
}

let n = 0;
function row(slot, side, needsAction, done) {
  ensure(n + 1);
  const t = slot.trackers[side];
  const v = E.stateValue(t, args.shapingW);
  reward[n] = v - slot.value[side];
  slot.value[side] = v;
  slotA[n] = slot.i * 2 + side;
  let f = (needsAction ? 1 : 0) | (done ? 2 : 0);
  if (slot.first[side] && needsAction) { f |= 4; slot.first[side] = false; }
  if (slot.rejected[side]) { f |= 8; slot.rejected[side] = false; }
  flags[n] = f;
  turnA[n] = Math.min(slot.battle.turn, 65535);
  const w = slot.battle.winner;
  win[n] = !done || slot.crashed ? 0 : w === (side ? 'p2' : 'p1') ? 1 : w === (side ? 'p1' : 'p2') ? -1 : 0;
  const op = slot.oppPrev[side];
  for (let k = 0; k < 4; k++) oppPrev[n * 4 + k] = op[k];
  slot.oppPrev[side] = [-1, 0, 0, 0];
  aux.writeOpponentTeam(t, slot.auxTeams[1 - side], oppTeam, n * 48, revealed, n * 6);
  if (needsAction) {
    E.encode(t, obs, n * D);
    E.actionMask(t, mask, n * A);
  } else {
    obs.fill(0, n * D, (n + 1) * D);
    mask.fill(0, n * A, (n + 1) * A);
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
  const h = new Uint32Array(buf.buffer, buf.byteOffset, 4);
  h[0] = MAGIC; h[1] = n; h[2] = D; h[3] = A;
  // columns were laid out for `cap` rows: compact them to n rows in place (same order, smaller strides)
  const parts = [[obs, D], [reward, 1], [slotA, 1], [oppTeam, 48], [oppPrev, 4], [turnA, 1], [mask, A], [flags, 1],
    [win, 1], [revealed, 6]];
  let o = 16;
  for (const [arr, w] of parts) {
    const bytes = n * w * arr.BYTES_PER_ELEMENT;
    buf.copy(buf, o, arr.byteOffset - buf.byteOffset, arr.byteOffset - buf.byteOffset + bytes);
    o += bytes;
  }
  writeAll(buf, o);
  n = 0;
}

function writeAll(b, len) {
  let off = 0;
  while (off < len) {
    try { off += fs.writeSync(args.fdOut, b, off, len - off); } catch (e) { if (e.code !== 'EAGAIN') throw e; }
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
process.stderr.write(`host done: ${JSON.stringify(stats)}\n`);
