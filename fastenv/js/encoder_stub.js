// Stand-in tracker + encoder with the encoder_v1 interface, for benchmarking the host and IPC before the real
// one exists: decisions come straight from the request, the observation is zeros (plus a few request numbers),
// actions map to the request's own move/switch order. Not for training.
'use strict';
class SideTracker {
  constructor({format, side}) { this.side = side; this.req = null; this.decisionPending = false; this.turns = 0; }
  onLog(text) { this.turns++; }
  onRequest(req) { this.req = req; this.decisionPending = !req.wait && !req.teamPreview; }
  onError(text) {}
  opponentIdents() { return []; }
}
const OBS_DIM = 4079;
function nActions(gen) { return gen >= 9 ? 26 : 10; }
function encode(t, out, off) {
  out.fill(0, off, off + OBS_DIM);
  const r = t.req;
  if (r && r.side) r.side.pokemon.forEach((p, i) => { out[off + 95 + 332 * i] = parseInt(p.condition) || 0; });
}
function actionMask(t, out, off) {
  const r = t.req, A = 26;
  out.fill(0, off, off + A);
  if (!r) return;
  const trapped = r.active && r.active[0] && (r.active[0].trapped);
  if (!trapped || r.forceSwitch) r.side.pokemon.forEach((p, i) => {
    if (!p.active && !p.condition.endsWith(' fnt')) out[off + i] = 1;
  });
  if (r.active && !r.forceSwitch) {
    r.active[0].moves.forEach((m, i) => {
      if (!m.disabled) { out[off + 6 + i] = 1; if (r.active[0].canTerastallize) out[off + 22 + i] = 1; }
    });
  }
}
function actionToChoice(t, a) {
  if (a < 6) return `switch ${a + 1}`;
  if (a >= 22) return `move ${a - 21} terastallize`;
  return `move ${((a - 6) % 4) + 1}`;
}
function stateValue(t, w) { return 0; }
module.exports = {SideTracker, OBS_DIM, nActions, encode, actionMask, actionToChoice, stateValue};
