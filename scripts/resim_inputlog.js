#!/usr/bin/env node
/**
 * Re-simulate Showdown random battles from their input logs: exact first-person views.
 *
 *   node scripts/resim_inputlog.js --builds DIR [--fallback SHA] [--git-dir PS.git] [--no-streams] IN.jsonl[.gz] [OUT]
 *
 * Input: JSONL (optionally gzipped, or `-` for stdin), one battle per line: {id, inputlog, log?} (`log`: the stored
 * replay log, used to validate the re-simulation). The input log of a random battle holds the Showdown commit
 * (`>version SHA`), the battle seed and each player's team seed (`>start`, `>player`) and every accepted choice, so
 * replaying it on the same simulator code reproduces the battle exactly: both true teams, and the messages and
 * `|request|`s each player's client received.
 *
 * Simulator code: DIR/<sha>/dist (built by scripts/build_showdown_versions.sh). A battle whose commit is not built
 * runs on --fallback (a built sha) if given, else on the newest built commit older than its own (commit times in
 * DIR/index.json), and is still validated: `exact` says whether the regenerated log matches the stored one.
 *
 * Output: JSONL, one object per battle:
 *   {id, version (the battle's commit), sim (the commit used), ok, error?, exact (null without a stored log),
 *    diff? (first mismatching line pair), winner ("p1"|"p2"|"tie"|null), end ("normal"|"forfeit"|"timer"),
 *    players: {p1, p2}, ratings: {p1, p2}, teams: {p1: [sets], p2: [sets]},
 *    streams: {p1: [chunk], p2: [chunk]}   (what each client received, in order: battle updates with private
 *                                           lines resolved for that player, and `|request|` JSON chunks),
 *    choices: {p1: [{chunk, choice}], p2: [...]}   (each request chunk index and the choice that answered it)}
 * With --no-streams the streams are left out (validation / team extraction only).
 */
"use strict";

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const readline = require("readline");
const child_process = require("child_process");

function parseArgs(argv) {
  const a = { builds: null, fallback: null, gitDir: null, streams: true, files: [] };
  for (let i = 0; i < argv.length; i++) {
    const x = argv[i];
    if (x === "--builds") a.builds = argv[++i];
    else if (x === "--fallback") a.fallback = argv[++i];
    else if (x === "--git-dir") a.gitDir = argv[++i];
    else if (x === "--no-streams") a.streams = false;
    else a.files.push(x);
  }
  if (!a.builds || a.files.length < 1) {
    console.error("usage: resim_inputlog.js --builds DIR [--fallback SHA] [--git-dir PS.git] [--no-streams] IN [OUT]");
    process.exit(2);
  }
  return a;
}

// ------------------------------------------------------------------ simulator versions

const SIMS = new Map(); // sha -> {BattleStream, extract}
const MAX_LOADED = 4; // each loaded simulator costs ~100 MB

function loadSim(builds, sha) {
  if (SIMS.has(sha)) {
    const s = SIMS.get(sha);
    SIMS.delete(sha);
    SIMS.set(sha, s); // LRU order
    return s;
  }
  const dist = path.resolve(builds, sha, "dist");
  const { BattleStream } = require(path.join(dist, "sim", "battle-stream"));
  let extract = null;
  try {
    extract = require(path.join(dist, "sim", "battle")).extractChannelMessages || null;
  } catch (e) { /* older layout */ }
  const sim = { BattleStream, extract: extract || extractChannelMessages };
  SIMS.set(sha, sim);
  if (SIMS.size > MAX_LOADED) {
    const old = SIMS.keys().next().value;
    SIMS.delete(old);
    const prefix = path.resolve(builds, old) + path.sep;
    for (const k of Object.keys(require.cache)) if (k.startsWith(prefix)) delete require.cache[k];
  }
  return sim;
}

// Showdown's own channel split (sim/battle.ts): `|split|pN` is followed by the secret line (that player and the
// omniscient channel) and the shared line (everyone else). Used when a build does not export it.
const SPLIT = /^\|split\|p([1234])\n(.*)\n(.*)|.+/gm;
function extractChannelMessages(message, channels) {
  const out = {};
  for (const c of channels) out[c] = [];
  for (const [line, player, secret, shared] of message.matchAll(SPLIT)) {
    const p = player ? parseInt(player) : 0;
    for (const c of channels) {
      let l = line;
      if (p) {
        l = c === -1 || c === p ? secret : shared;
        if (!l) continue;
      }
      out[c].push(l);
    }
  }
  return out;
}

let INDEX = null; // sha -> commit time (s), from DIR/index.json
const COMMIT_TIME = new Map();

function commitTime(args, sha) {
  if (INDEX && sha in INDEX) return INDEX[sha];
  if (COMMIT_TIME.has(sha)) return COMMIT_TIME.get(sha);
  let t = null;
  if (args.gitDir) {
    try {
      t = parseInt(child_process.execFileSync("git", ["--git-dir", args.gitDir, "log", "-1", "--format=%ct", sha],
        { stdio: ["ignore", "pipe", "ignore"] }).toString().trim()) || null;
    } catch (e) { t = null; }
  }
  COMMIT_TIME.set(sha, t);
  return t;
}

function builtShas(builds) {
  return fs.readdirSync(builds).filter(d => fs.existsSync(path.join(builds, d, "dist", "sim", "battle-stream.js")));
}

function pickSim(args, version) {
  const built = builtShas(args.builds);
  if (version && built.includes(version)) return version;
  if (args.fallback) return args.fallback;
  const t = version ? commitTime(args, version) : null;
  let best = null, bestT = -Infinity;
  for (const sha of built) {
    const st = commitTime(args, sha);
    if (st === null) continue;
    if ((t === null || st <= t) && st > bestT) { best = sha; bestT = st; }
  }
  if (best === null && built.length) best = built[built.length - 1];
  return best;
}

// ------------------------------------------------------------------ one battle

// Room/chat lines the stored replay has and the simulator never makes.
const ROOM_LINES = new Set(["j", "J", "l", "L", "n", "N", "c", "c:", "chat", "raw", "html", "uhtml", "uhtmlchange",
  "inactive", "inactiveoff", "t:", "badge", "player", "init", "title", "join", "leave", "name", "unlink", "hidelines",
  "tempnotify", "tempnotifyoff", "notify", "userstats", "error", "bigerror", "debug", "message", "-message", "rated",
  "timestamp", "tournament", "seed", "expire", "deinit", "noinit", "uhtmlchange", "controlshtml", "fieldhtml",
  "selectorhtml", "sentchoice", "askreg", "zbroken", "-hint"]);

function battleLines(text) {
  return text.split("\n").filter(l => {
    if (!l.startsWith("|")) return false;
    const kind = l.split("|")[1];
    return !ROOM_LINES.has(kind);
  });
}

function resim(args, rec) {
  const lines = rec.inputlog.split("\n");
  const versionLine = lines.find(l => l.startsWith(">version "));
  const version = versionLine ? versionLine.slice(9).trim().split(" ")[0] : null;
  const out = { id: rec.id, version, sim: null, ok: false, exact: null };
  const sha = pickSim(args, version);
  if (!sha) throw new Error("no simulator build");
  out.sim = sha;
  const { BattleStream, extract } = loadSim(args.builds, sha);

  const stream = new BattleStream({ noCatch: true, keepAlive: true });
  const pushed = [];
  stream.push = chunk => { pushed.push(chunk); };
  stream.pushEnd = () => {};
  stream.pushError = (e) => { throw e; };

  const streams = { p1: [], p2: [] };
  const choices = { p1: [], p2: [] };
  const omniscient = [];
  const spectator = [];
  const players = {}, ratings = {};
  let forfeit = null; // side that forcelose'd
  const drain = () => {
    for (const chunk of pushed.splice(0)) {
      const nl = chunk.indexOf("\n");
      const type = nl < 0 ? chunk : chunk.slice(0, nl);
      const data = nl < 0 ? "" : chunk.slice(nl + 1);
      if (type === "update") {
        const ch = extract(data, [-1, 0, 1, 2]);
        omniscient.push(...ch[-1]);
        spectator.push(...ch[0]);
        if (ch[1].length) streams.p1.push(ch[1].join("\n"));
        if (ch[2].length) streams.p2.push(ch[2].join("\n"));
      } else if (type === "sideupdate") {
        const nl2 = data.indexOf("\n");
        const side = data.slice(0, nl2), body = data.slice(nl2 + 1);
        if (!streams[side]) continue;
        streams[side].push(body);
        if (body.startsWith("|request|")) {
          const req = JSON.parse(body.slice(9) || "null");
          if (req && !req.wait && !req.teamPreview) choices[side].push({ chunk: streams[side].length - 1, choice: null });
        }
      }
    }
  };
  for (const line of lines) {
    if (!line.startsWith(">")) continue;
    const sp = line.indexOf(" ");
    const type = sp < 0 ? line.slice(1) : line.slice(1, sp);
    const msg = sp < 0 ? "" : line.slice(sp + 1);
    if (type === "version" || type === "version-origin" || type === "chat" || type === "chat-inputlogonly") continue;
    if (type === "player") {
      const [slot, opts] = [msg.slice(0, 2), JSON.parse(msg.slice(3))];
      players[slot] = opts.name;
      ratings[slot] = opts.rating || null;
    }
    if ((type === "p1" || type === "p2") && msg !== "undo") {
      const pending = choices[type][choices[type].length - 1];
      if (pending) pending.choice = msg;
    } else if ((type === "p1" || type === "p2") && msg === "undo") {
      const pending = choices[type][choices[type].length - 1];
      if (pending) pending.choice = null;
    }
    if (type === "forcelose") forfeit = msg.trim();
    stream._write(line);
    drain();
  }
  const battle = stream.battle;
  drain();
  const sides = battle ? battle.sides : [];
  out.players = players;
  out.ratings = ratings;
  out.teams = {};
  for (const s of sides) out.teams[s.id] = (s.team || []).map(cleanSet);
  const winner = battle && battle.ended ? (battle.winner ? (sides.find(s => s.name === battle.winner) || {}).id || null : "tie") : null;
  out.winner = winner;
  out.turns = battle ? battle.turn : null;
  let end = "normal";
  if (forfeit) end = "forfeit";
  if (rec.log && /\|-?message\|[^\n]*(lost due to inactivity|lost because of their timer|disconnected and timed out)/.test(rec.log)) end = "timer";
  out.end = end;
  if (rec.log) {
    const mine = battleLines(spectator.join("\n"));
    const mineOmni = battleLines(omniscient.join("\n"));
    const theirs = battleLines(rec.log);
    const cmp = (a, b) => {
      const n = Math.min(a.length, b.length);
      for (let i = 0; i < n; i++) if (a[i] !== b[i]) return { line: i, sim: a[i], log: b[i] };
      return a.length === b.length ? null : { line: n, sim: a[n] || null, log: b[n] || null };
    };
    // Public replays show exact HP for both sides (omniscient lines); older ones the spectator view.
    const dOmni = cmp(mineOmni, theirs), dSpec = cmp(mine, theirs);
    out.exact = !dOmni || !dSpec;
    out.view = !dOmni ? "omniscient" : !dSpec ? "spectator" : null;
    if (!out.exact) out.diff = (dOmni && dSpec && dSpec.line > dOmni.line) ? dSpec : dOmni;
  }
  if (args.streams) {
    out.streams = streams;
    out.choices = choices;
  }
  out.ok = true;
  return out;
}

function cleanSet(s) {
  const o = {};
  for (const k of ["name", "species", "gender", "shiny", "level", "moves", "ability", "evs", "ivs", "item", "nature",
    "teraType", "role", "happiness", "hpType", "gigantamax", "dynamaxLevel"]) if (s[k] !== undefined) o[k] = s[k];
  return o;
}

// ------------------------------------------------------------------ main

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const idx = path.join(args.builds, "index.json");
  if (fs.existsSync(idx)) INDEX = JSON.parse(fs.readFileSync(idx, "utf8"));
  const [inPath, outPath] = args.files;
  let input = inPath === "-" ? process.stdin : fs.createReadStream(inPath);
  if (inPath.endsWith(".gz")) input = input.pipe(zlib.createGunzip());
  let output = outPath ? fs.createWriteStream(outPath) : process.stdout;
  if (outPath && outPath.endsWith(".gz")) {
    const gz = zlib.createGzip();
    gz.pipe(output);
    output = gz;
  }
  const rl = readline.createInterface({ input, crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line.trim()) continue;
    let rec, res;
    try {
      rec = JSON.parse(line);
      res = resim(args, rec);
    } catch (e) {
      res = { id: rec ? rec.id : null, ok: false, error: `${e.name}: ${String(e.message).slice(0, 300)}` };
    }
    if (!output.write(JSON.stringify(res) + "\n")) await new Promise(r => output.once("drain", r));
  }
  output.end();
}

main().catch(e => { console.error(e); process.exit(1); });
