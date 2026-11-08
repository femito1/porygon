// One in-process Showdown battle with per-player message streams, exactly as a Showdown server delivers them to
// each player (each `update` log split by |split|pN: the secret line goes to pN, the shared line to the other
// player; then each player's request), with no server, no websocket and (optionally) no request JSON.
//
//   const b = new SimBattle(format, seed, [team1, team2], {onLog(side, text) {}, onRequest(side, req, json) {}});
//   b.start(); while (!b.ended) for (const s of b.pendingSides()) b.choose(s, 'move 1');
//
// onLog(side, text): one update's log for that player, lines joined by \n, no ">battle-..." room header and no
// server-only lines (|init|, |title|, |j|): what follows the header in the server message.
// onRequest(side, req, json): the player's new request as an object (Showdown's side.activeRequest) and, unless
// keepRequestJSON is false, the exact JSON text the server would send after "|request|".
// onError(side, text): "|error|[Invalid choice] ..." messages.
// Message order is the server's: an update's log first, then both players' requests (Battle.sendUpdates).
const {Battle, Teams} = require('./ps');

class SimBattle {
  constructor(format, seed, teams, {onLog = null, onRequest = null, onError = null, keepRequestJSON = true} = {}) {
    this.format = format;
    this.teams = teams;  // [PokemonSet[], PokemonSet[]]: the true teams (aux targets)
    this.onLog = onLog;
    this.onRequest = onRequest;
    this.onError = onError;
    this.keepRequestJSON = keepRequestJSON;
    this.battle = new Battle({formatid: format, seed, send: (t, d) => this._send(t, d)});
  }

  start() {
    const self = this, keep = this.keepRequestJSON;
    for (const [i, id] of ['p1', 'p2'].entries()) {
      this.battle.setPlayer(id, {name: id, team: Teams.pack(this.teams[i])});
      // requests are only emitted from sendUpdates, so hooking each side right after it exists catches all
      this.battle.sides[i].emitRequest = function (update = this.activeRequest, updatedRequest = false) {
        if (updatedRequest) update.update = true;
        this.activeRequest = update;
        if (self.onRequest) self.onRequest(this.n, update, keep ? JSON.stringify(update) : null);
      };
    }
    this.battle.sendUpdates();
  }

  get ended() { return this.battle.ended; }
  get winner() { return this.battle.winner; }  // '' for a tie, else the winner's name 'p1' / 'p2'
  get turn() { return this.battle.turn; }

  /** Sides (0/1) that owe a choice now. */
  pendingSides() {
    const out = [];
    for (const side of this.battle.sides) {
      const r = side.activeRequest;
      if (r && !r.wait && !side.isChoiceDone()) out.push(side.n);
    }
    return out;
  }

  request(n) { return this.battle.sides[n].activeRequest; }

  /** Returns false (the choice stays open) if Showdown rejects it. */
  choose(n, choice) {
    const ok = this.battle.choose(n ? 'p2' : 'p1', choice);
    this.battle.sendUpdates();  // what BattleStream does after every command: flush the log, then requests
    return ok;
  }

  _send(type, data) {
    if (type === 'update') {
      if (!this.onLog) return;
      const lines = Array.isArray(data) ? data : data.split('\n');
      const out = [[], []];
      for (let i = 0; i < lines.length; i++) {
        const l = lines[i];
        if (l.startsWith('|split|p')) {
          const p = l.charCodeAt(8) - 49;  // 'p1' -> 0
          const secret = lines[i + 1], shared = lines[i + 2];
          i += 2;
          if (secret) out[p].push(secret);
          if (shared) out[1 - p].push(shared);
        } else {
          out[0].push(l); out[1].push(l);
        }
      }
      for (const n of [0, 1]) if (out[n].length) this.onLog(n, out[n].join('\n'));
    } else if (type === 'sideupdate') {  // only errors arrive here (requests go through emitRequest above)
      const nl = data.indexOf('\n');
      if (this.onError) this.onError(data.charCodeAt(1) - 49, data.slice(nl + 1));
    }
  }
}

module.exports = {SimBattle, Teams};
