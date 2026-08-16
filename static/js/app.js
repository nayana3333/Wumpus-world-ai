// Front end for the Wumpus World server.
// All rules and inference live in Python; this file only draws state and
// forwards actions. Nothing here knows where the pits are.

'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const el = {
  grid: $('#grid'), board: $('#board'), trail: $('#trail'),
  token: $('#agentToken'), arrowFx: $('#arrowFx'),
  status: $('#statusLine'), log: $('#log'), riskTable: $('#riskTable'),
  proofBody: $('#proofBody'),
  overlay: $('#overlay'), ovEmoji: $('#ovEmoji'), ovTitle: $('#ovTitle'),
  ovText: $('#ovText'), ovScore: $('#ovScore'),
  hudScore: $('#hudScore'), hudSteps: $('#hudSteps'), hudArrow: $('#hudArrow'),
  hudGold: $('#hudGold'), hudWumpus: $('#hudWumpus'),
  agentList: $('#agentList'), advice: $('#adviceCard'),
  adviceAction: $('#adviceAction'), adviceReason: $('#adviceReason'),
  confBar: $('#confBar'), confPct: $('#confPct'),
  inpSize: $('#inpSize'), inpPits: $('#inpPits'), inpSeed: $('#inpSeed'),
  inpSpeed: $('#inpSpeed'), speedLabel: $('#speedLabel'),
  tglVision: $('#tglVision'), tglNumbers: $('#tglNumbers'),
  btnAuto: $('#btnAuto'), btnStep: $('#btnStep'), btnHint: $('#btnHint'), btnSolve: $('#btnSolve'),
  toast: $('#toast'),
  benchChart: $('#benchChart'), benchTable: $('#benchTable'), benchNote: $('#benchNote'),
  curve: $('#curve'), trainSummary: $('#trainSummary'),
};

const store = {
  state: null,
  knowledge: null,
  agent: 'probabilistic',
  agents: [],
  difficulties: null,
  autoTimer: null,
  playing: false,
  cellEls: [],
  loggedDerivations: 0,
  busy: false,
  solving: false,
};

async function api(path, body) {
  const opts = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

function toast(message, ms = 2200) {
  el.toast.textContent = message;
  el.toast.hidden = false;
  setTimeout(() => el.toast.classList.add('show'), 20);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.toast.classList.remove('show');
    setTimeout(() => { el.toast.hidden = true; }, 250);
  }, ms);
}

const key = (r, c) => `${r},${c}`;

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
}

// --- sound ------------------------------------------------------------------

const sound = {
  on: localStorage.getItem('ww:sound') !== 'off',
  ctx: null,
  play(freq, dur = 0.09, type = 'sine', gain = 0.05) {
    if (!this.on) return;
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return;
      this.ctx = new Ctx();
    }
    const osc = this.ctx.createOscillator();
    const amp = this.ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
    amp.gain.setValueAtTime(gain, this.ctx.currentTime);
    amp.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + dur);
    osc.connect(amp).connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + dur);
  },
  move() { this.play(320, 0.05, 'triangle', 0.03); },
  bump() { this.play(110, 0.13, 'square', 0.04); },
  gold() { [660, 880, 1180].forEach((f, i) => setTimeout(() => this.play(f, 0.12, 'sine', 0.05), i * 80)); },
  shoot() { this.play(880, 0.05, 'sawtooth', 0.04); setTimeout(() => this.play(240, 0.18, 'sawtooth', 0.03), 60); },
  scream() { [520, 400, 300, 200].forEach((f, i) => setTimeout(() => this.play(f, 0.2, 'sawtooth', 0.05), i * 90)); },
  death() { [220, 165, 110].forEach((f, i) => setTimeout(() => this.play(f, 0.28, 'square', 0.05), i * 140)); },
  win() { [523, 659, 784, 1047].forEach((f, i) => setTimeout(() => this.play(f, 0.2, 'sine', 0.06), i * 120)); },
};

// --- board ------------------------------------------------------------------

// Derive the cell size from the space actually available: a 12x12 cave at the
// stylesheet default is wider than the middle column, and a 4x4 one leaves most
// of it empty.
const MAX_CELL = 92;

function fitCells(size) {
  // Clamp against the viewport as well as the parent. If the board is already
  // overflowing, the parent can measure wider than the screen, and sizing off
  // that alone keeps it overflowing.
  const parentWidth = el.board.parentElement?.clientWidth || 0;
  const viewport = document.documentElement.clientWidth - 40;
  const available = Math.min(parentWidth, viewport > 0 ? viewport : parentWidth) - 8;
  const gap = parseFloat(getComputedStyle(el.grid).gap) || 6;
  const ideal = Math.floor((available - 28 - gap * (size - 1)) / size);
  // A hidden container measures zero; don't bake that in, fall back to the
  // stylesheet's breakpoint value.
  if (!Number.isFinite(ideal) || ideal < 24) {
    el.board.style.removeProperty('--cell');
    return;
  }
  el.board.style.setProperty('--cell', `${Math.min(MAX_CELL, ideal)}px`);
}

function buildGrid(size) {
  fitCells(size);
  el.grid.style.setProperty('--n', size);
  el.grid.innerHTML = '';
  store.cellEls = [];

  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.r = r;
      cell.dataset.c = c;
      cell.innerHTML =
        `<span class="coord">${r},${c}</span>` +
        `<span class="marks"></span>` +
        `<span class="glyph"></span>` +
        `<span class="risk"><span class="rp"></span><span class="rw"></span></span>`;
      cell.addEventListener('click', () => onCellClick(r, c));
      el.grid.appendChild(cell);
      row.push(cell);
    }
    store.cellEls.push(row);
  }
}

function glyphFor(tile, state) {
  if (tile.pit) return { char: '\u{1F573}️', cls: 'pit' };
  if (tile.wumpus) return { char: state.wumpus_alive ? '\u{1F479}' : '\u{1F480}', cls: 'wump' };
  if (tile.gold) return { char: '\u{1F4B0}', cls: 'gold' };
  return { char: '', cls: '' };
}

function render(payload, opts = {}) {
  const prev = store.state;
  const state = payload.state;
  const kn = payload.knowledge;
  store.state = state;
  store.knowledge = kn;

  if (!store.cellEls.length || store.cellEls.length !== state.size) buildGrid(state.size);

  const [ar, ac] = state.agent;
  const adjacent = new Set([key(ar - 1, ac), key(ar + 1, ac), key(ar, ac - 1), key(ar, ac + 1)]);
  // Marking every square as a possible Wumpus location says nothing. Only ring
  // them once the percepts have actually narrowed the field down.
  const wumpusNarrowed = !kn.wumpus_dead
    && kn.wumpus_candidates.length > 0
    && kn.wumpus_candidates.length <= 4;

  for (let r = 0; r < state.size; r++) {
    for (let c = 0; c < state.size; c++) {
      const cell = store.cellEls[r][c];
      const tile = state.tiles[r][c];
      const info = kn.cells[key(r, c)] || {};

      cell.classList.toggle('visited', tile.visited);
      cell.classList.toggle('fog', !tile.seen);
      cell.classList.toggle('start', tile.start);
      cell.classList.toggle('reachable', !state.game_over && adjacent.has(key(r, c)));
      cell.classList.toggle('wumpus-suspect', wumpusNarrowed && !tile.visited && (info.wumpus || 0) > 0);
      cell.dataset.label = tile.visited ? 'safe' : (info.label || 'unknown');

      const g = tile.seen ? glyphFor(tile, state) : { char: '', cls: '' };
      const glyph = cell.querySelector('.glyph');
      if (glyph.textContent !== g.char) glyph.textContent = g.char;
      glyph.className = `glyph ${g.cls}`;

      // What we smelled the last time we stood here.
      const memory = (kn.percept_memory || {})[key(r, c)];
      cell.querySelector('.marks').textContent = memory
        ? `${memory.breeze ? '\u{1F4A8}' : ''}${memory.stench ? '\u{1F7E2}' : ''}`
        : '';

      const rp = cell.querySelector('.rp');
      const rw = cell.querySelector('.rw');
      if (tile.visited || info.danger === undefined) {
        rp.textContent = '';
        rw.textContent = '';
      } else {
        rp.textContent = info.pit > 0 ? `P ${Math.round(info.pit * 100)}%` : '';
        rw.textContent = info.wumpus > 0 ? `W ${Math.round(info.wumpus * 100)}%` : '';
      }
    }
  }

  positionToken(ar, ac, state);
  drawTrail(state);
  renderHud(state);
  renderPercepts(state);
  renderLog(payload, opts);
  renderRiskTable(state, kn);

  el.status.textContent = state.status;
  if (state.arrow_flight && state.arrow_flight.length) flyArrow(state);

  playSounds(prev, state);
  updateControls(state);

  if (state.game_over) {
    showOverlay(state);
    recordOutcome(state);
  } else {
    el.overlay.hidden = true;
  }
}

// The token, trail and arrow sit inside .board, which is the cells' offsetParent,
// so offsetLeft/offsetTop already measure from the right origin. Using those
// instead of getBoundingClientRect avoids reading a rect mid-transition.
function cellCentre(r, c) {
  const cell = store.cellEls[r]?.[c];
  if (!cell) return null;
  return [cell.offsetLeft + cell.offsetWidth / 2, cell.offsetTop + cell.offsetHeight / 2];
}

function positionToken(r, c, state) {
  const cell = store.cellEls[r]?.[c];
  if (!cell) return;
  el.token.style.width = `${cell.offsetWidth}px`;
  el.token.style.height = `${cell.offsetHeight}px`;
  el.token.style.transform = `translate(${cell.offsetLeft}px, ${cell.offsetTop}px)`;

  const face = el.token.querySelector('.token-face');
  if (state.outcome === 'pit' || state.outcome === 'wumpus') face.textContent = 'x';
  else face.textContent = state.has_gold ? '@$' : '@';
  el.token.classList.toggle('dead', state.outcome === 'pit' || state.outcome === 'wumpus');
}

function drawTrail(state) {
  const w = el.board.clientWidth;
  const h = el.board.clientHeight;
  el.trail.setAttribute('viewBox', `0 0 ${w} ${h}`);
  el.trail.setAttribute('width', w);
  el.trail.setAttribute('height', h);

  const points = state.path
    .map(([r, c]) => cellCentre(r, c))
    .filter(Boolean)
    .map(([x, y]) => `${x},${y}`);

  el.trail.innerHTML = points.length > 1 ? `<polyline points="${points.join(' ')}"></polyline>` : '';
}

function renderHud(state) {
  el.hudScore.textContent = state.score;
  el.hudScore.classList.toggle('neg', state.score < 0);
  el.hudSteps.textContent = state.steps;
  el.hudArrow.textContent = state.has_arrow ? '1' : '0';
  el.hudGold.textContent = state.has_gold ? 'held' : '-';
  el.hudWumpus.textContent = state.wumpus_alive ? 'alive' : 'dead';
}

function renderPercepts(state) {
  $$('#percepts li').forEach((li) => li.classList.toggle('on', !!state.percept[li.dataset.p]));
}

function renderLog(payload, opts) {
  const derivations = payload.knowledge.derivations || [];

  if (opts.reset) {
    el.log.innerHTML = '';
    store.loggedDerivations = 0;
  }

  if (opts.decision) {
    const d = opts.decision;
    const li = document.createElement('li');
    li.className = 'act';
    const verb = `${d.action.toUpperCase()}${d.direction ? ' ' + d.direction.toUpperCase() : ''}`;
    li.innerHTML = `<span class="rule">act</span><span><b>${verb}</b> - ${escapeHtml(d.reason)}</span>`;
    el.log.appendChild(li);
  }

  for (let i = store.loggedDerivations; i < derivations.length; i++) {
    const d = derivations[i];
    const li = document.createElement('li');
    li.innerHTML = `<span class="rule">${d.rule}</span><span>${escapeHtml(d.text)}</span>`;
    el.log.appendChild(li);
  }
  store.loggedDerivations = derivations.length;

  const state = payload.state;
  if (state.game_over && !el.log.querySelector('.terminal')) {
    const li = document.createElement('li');
    li.className = `terminal ${state.won ? 'win' : (state.outcome === 'escaped' ? '' : 'dead')}`;
    li.innerHTML = `<span class="rule">end</span><span>${escapeHtml(state.status)} Final score ${state.score}.</span>`;
    el.log.appendChild(li);
  }

  if (!el.log.children.length) {
    el.log.innerHTML = '<li class="log-empty">Move, and derived facts appear here.</li>';
  }
  el.log.scrollTop = el.log.scrollHeight;
}

function renderRiskTable(state, kn) {
  const rows = [];
  for (let r = 0; r < state.size; r++) {
    for (let c = 0; c < state.size; c++) {
      const info = kn.cells[key(r, c)];
      if (!info || info.visited) continue;
      rows.push({ r, c, ...info });
    }
  }
  rows.sort((a, b) => a.danger - b.danger || a.r - b.r || a.c - b.c);

  const cls = (v) => (v <= 0 ? 'v-dim' : v >= 1 ? 'v-bad' : v < 0.25 ? 'v-safe' : 'v-warn');
  const pct = (v) => (v <= 0 ? '.' : `${Math.round(v * 100)}%`);

  el.riskTable.innerHTML =
    '<thead><tr><th>cell</th><th>P(pit)</th><th>P(wump)</th><th>danger</th><th>verdict</th></tr></thead><tbody>' +
    (rows.length
      ? rows.map((x) => `<tr data-r="${x.r}" data-c="${x.c}">
          <td>(${x.r},${x.c})</td>
          <td class="${cls(x.pit)}">${pct(x.pit)}</td>
          <td class="${cls(x.wumpus)}">${pct(x.wumpus)}</td>
          <td class="${cls(x.danger)}">${pct(x.danger)}</td>
          <td class="${cls(x.danger)}">${x.label}</td>
        </tr>`).join('')
      : '<tr><td colspan="5" class="v-dim">Whole cave explored.</td></tr>') +
    '</tbody>';

  $$('#riskTable tbody tr[data-r]').forEach((tr) => {
    tr.addEventListener('click', () => showProof(Number(tr.dataset.r), Number(tr.dataset.c)));
  });
}

function updateControls(state) {
  const over = state.game_over;
  // Solve renders a frame at a time, and each render lands here. Without the
  // `solving` guard every frame would re-enable the buttons mid-animation, so
  // the controls would look live while they are not.
  const locked = over || store.solving;
  $$('.dbtn').forEach((b) => { b.disabled = locked; });
  $$('[data-shoot]').forEach((b) => { b.disabled = locked || !state.has_arrow; });
  $('#btnClimb').disabled = locked;
  el.btnStep.disabled = locked;
  el.btnHint.disabled = locked;
  el.btnSolve.disabled = locked;
  el.btnAuto.disabled = locked;
  if (over && store.playing) stopAuto();
}

function playSounds(prev, state) {
  if (!prev) return;
  if (state.percept.bump) {
    sound.bump();
    el.token.classList.add('bumped');
    setTimeout(() => el.token.classList.remove('bumped'), 300);
  } else if (String(prev.agent) !== String(state.agent)) {
    sound.move();
  }
  if (!prev.has_gold && state.has_gold) sound.gold();
  if (prev.has_arrow && !state.has_arrow) sound.shoot();
  if (prev.wumpus_alive && !state.wumpus_alive) sound.scream();
  if (!prev.game_over && state.game_over) {
    if (state.won) sound.win();
    else if (state.outcome !== 'escaped') sound.death();
  }
}

function flyArrow(state) {
  const flight = state.arrow_flight;
  if (!flight.length) return;
  const last = flight[flight.length - 1];
  const start = cellCentre(state.agent[0], state.agent[1]);
  const end = cellCentre(last[0], last[1]);
  if (!start || !end) return;
  const angle = Math.atan2(end[1] - start[1], end[0] - start[0]) * 180 / Math.PI;

  el.arrowFx.textContent = '>';
  el.arrowFx.style.transition = 'none';
  el.arrowFx.style.transform = `translate(${start[0] - 10}px, ${start[1] - 10}px) rotate(${angle}deg)`;
  el.arrowFx.classList.add('fire');
  setTimeout(() => {
    el.arrowFx.style.transition = 'transform .3s linear';
    el.arrowFx.style.transform = `translate(${end[0] - 10}px, ${end[1] - 10}px) rotate(${angle}deg)`;
    setTimeout(() => el.arrowFx.classList.remove('fire'), 340);
  }, 20);
}

function showOverlay(state) {
  const won = state.won;
  const escaped = state.outcome === 'escaped';
  el.ovEmoji.textContent = won ? '$' : escaped ? '-' : 'x';
  el.ovTitle.textContent = won
    ? 'Out with the gold'
    : escaped ? 'Out empty-handed' : (state.outcome === 'pit' ? 'Fell into a pit' : 'Eaten by the Wumpus');
  el.ovText.textContent = state.status;
  el.ovScore.textContent = state.score;
  el.overlay.hidden = false;
}

// --- actions ----------------------------------------------------------------

function onCellClick(r, c) {
  const state = store.state;
  if (!state) return;
  const [ar, ac] = state.agent;
  const dr = r - ar;
  const dc = c - ac;
  const dir = dr === -1 && dc === 0 ? 'up'
    : dr === 1 && dc === 0 ? 'down'
    : dr === 0 && dc === -1 ? 'left'
    : dr === 0 && dc === 1 ? 'right' : null;

  // Adjacent means "walk there", anything else means "explain this square".
  if (dir && !state.game_over) act('move', dir);
  else showProof(r, c);
}

async function act(action, direction) {
  if (store.busy || !store.state || store.state.game_over) return;
  store.busy = true;
  try {
    el.advice.hidden = true;
    render(await api('/api/action', { action, direction }));
  } catch (err) {
    toast('Lost the connection to the server.');
  } finally {
    store.busy = false;
  }
}

// --- agent ------------------------------------------------------------------

function renderAgents() {
  el.agentList.innerHTML = store.agents.map((a) => `
    <button class="agent-opt ${a.name === store.agent ? 'is-active' : ''}" data-agent="${a.name}">
      <b>${escapeHtml(a.label)}</b>
      <small>${escapeHtml(a.blurb)}</small>
    </button>`).join('');

  $$('.agent-opt', el.agentList).forEach((btn) => {
    btn.addEventListener('click', () => {
      store.agent = btn.dataset.agent;
      renderAgents();
      const chosen = store.agents.find((a) => a.name === store.agent);
      toast(`Agent: ${chosen ? chosen.label : store.agent}`);
    });
  });
}

async function aiStep() {
  if (store.busy || !store.state || store.state.game_over) return false;
  store.busy = true;
  try {
    const payload = await api('/api/ai/step', { agent: store.agent });
    render(payload, { decision: payload.decision });
    if (payload.decision) showAdvice(payload.decision);
    return !payload.state.game_over;
  } catch (err) {
    toast('AI step failed.');
    return false;
  } finally {
    store.busy = false;
  }
}

async function hint() {
  if (!store.state || store.state.game_over) return;
  try {
    const payload = await api('/api/hint', { agent: store.agent });
    render(payload);
    if (payload.decision) {
      showAdvice(payload.decision);
      highlightHint(payload.decision);
    }
  } catch (err) {
    toast('Could not reach the agent.');
  }
}

function showAdvice(decision) {
  el.advice.hidden = false;
  el.adviceAction.textContent =
    `${decision.action.toUpperCase()}${decision.direction ? ' ' + decision.direction.toUpperCase() : ''}`;
  el.adviceReason.textContent = decision.reason;
  const pct = Math.round((decision.confidence ?? 1) * 100);
  el.confBar.style.width = `${pct}%`;
  el.confPct.textContent = `${pct}%`;
}

function highlightHint(decision) {
  $$('.cell.hinted').forEach((c) => c.classList.remove('hinted'));
  if (decision.action !== 'move' || !store.state) return;
  const delta = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] }[decision.direction];
  if (!delta) return;
  const cell = store.cellEls[store.state.agent[0] + delta[0]]?.[store.state.agent[1] + delta[1]];
  if (cell) {
    cell.classList.add('hinted');
    setTimeout(() => cell.classList.remove('hinted'), 3000);
  }
}

// --- proof panel ------------------------------------------------------------

async function showProof(r, c) {
  $$('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === 'proof'));
  $$('.tab-panel').forEach((p) => p.classList.toggle('is-active', p.dataset.panel === 'proof'));
  $$('.cell.probed').forEach((x) => x.classList.remove('probed'));
  store.cellEls[r]?.[c]?.classList.add('probed');

  el.proofBody.innerHTML = '<p class="proof-hint">Running DPLL...</p>';
  try {
    renderProof(await api('/api/proof', { r, c }), r, c);
  } catch (err) {
    el.proofBody.innerHTML = '<p class="proof-hint">The prover could not be reached.</p>';
  }
}

function renderProof(data, r, c) {
  if (!data.available) {
    el.proofBody.innerHTML = `<p class="proof-hint">${escapeHtml(data.reason)}</p>`;
    return;
  }

  const verdictClass = data.verdict.includes('safe') ? 'safe'
    : data.verdict.includes('deadly') ? 'deadly' : 'unknown';

  let html = `<div class="proof-verdict ${verdictClass}">(${r},${c}) is ${escapeHtml(data.verdict)}</div>`;

  if (!data.entailed.length) {
    html += '<p class="proof-hint">Nothing about this square is entailed yet. '
      + 'Logic gives no answer, which is exactly when the probability side takes over &mdash; see the Risk tab.</p>';
  }

  data.entailed.forEach((claim) => {
    html += `<div class="proof-claim">KB entails <b>${escapeHtml(claim.literal)}</b>: ${escapeHtml(claim.claim)}</div>`;
    if (!claim.steps.length) {
      html += '<p class="proof-hint">DPLL confirms it, but unit resolution alone gives no short chain to show.</p>';
      return;
    }
    html += '<ol class="proof-steps">';
    claim.steps.forEach((step) => {
      html += `<li>
        <div class="res">${escapeHtml(step.from)}</div>
        <div class="res">&rarr; resolve on ${escapeHtml(step.on)} &rArr; ${escapeHtml(step.gives)}</div>
        ${step.why ? `<div class="why">${escapeHtml(step.why)}</div>` : ''}
      </li>`;
    });
    html += '</ol>';
  });

  html += `<p class="proof-meta">Knowledge base: ${data.clauses} CNF clauses. `
    + 'Entailment is decided by asking DPLL whether the KB plus the negation of the claim is unsatisfiable. '
    + 'An empty clause at the end of a chain is the contradiction.</p>';

  el.proofBody.innerHTML = html;
}

// --- auto play --------------------------------------------------------------

function speedMs() { return 1260 - Number(el.inpSpeed.value); }

function startAuto() {
  if (store.playing) return;
  store.playing = true;
  el.btnAuto.textContent = 'Pause';
  el.btnAuto.classList.add('is-playing');
  const tick = async () => {
    if (!store.playing) return;
    const more = await aiStep();
    if (more && store.playing) store.autoTimer = setTimeout(tick, speedMs());
    else stopAuto();
  };
  tick();
}

function stopAuto() {
  store.playing = false;
  clearTimeout(store.autoTimer);
  el.btnAuto.textContent = 'Auto play';
  el.btnAuto.classList.remove('is-playing');
}

async function solve() {
  if (store.busy || store.solving || !store.state || store.state.game_over) return;
  stopAuto();
  store.busy = true;
  store.solving = true;
  updateControls(store.state);
  try {
    const { frames } = await api('/api/ai/solve', { agent: store.agent, max_steps: 200 });
    for (const frame of frames) {
      render(frame, { decision: frame.decision });
      if (frame.decision) showAdvice(frame.decision);
      await new Promise((done) => setTimeout(done, Math.min(160, speedMs())));
    }
    toast(`Finished in ${frames.length} actions.`);
  } catch (err) {
    toast('Solve failed.');
  } finally {
    store.busy = false;
    store.solving = false;
    if (store.state) updateControls(store.state);
  }
}

// --- new game ---------------------------------------------------------------

async function newGame(opts = {}) {
  stopAuto();
  el.advice.hidden = true;
  const size = Number(el.inpSize.value) || 4;
  const pits = Number(el.inpPits.value);
  const seedRaw = opts.seed !== undefined ? opts.seed : el.inpSeed.value.trim();
  const seed = seedRaw === '' || seedRaw === null ? null : Number(seedRaw);

  try {
    const payload = await api('/api/new', {
      size,
      pits: Number.isFinite(pits) ? pits : 3,
      seed: Number.isFinite(seed) ? seed : null,
      agent: store.agent,
    });
    store.cellEls = [];
    el.inpSeed.value = payload.state.seed ?? '';
    el.proofBody.innerHTML = '<p class="proof-hint">Click any square to see what can be proved about it.</p>';
    render(payload, { reset: true });
    countedThisGame = false;
    renderRecord();
  } catch (err) {
    toast('Could not start a new cave.');
  }
}

// --- local record -----------------------------------------------------------

let countedThisGame = false;

function readRecord() {
  try { return JSON.parse(localStorage.getItem('ww:record')) || {}; }
  catch { return {}; }
}

function recordOutcome(state) {
  if (countedThisGame) return;
  countedThisGame = true;
  const data = readRecord();
  data.games = (data.games || 0) + 1;
  if (state.won) {
    data.wins = (data.wins || 0) + 1;
    data.streak = (data.streak || 0) + 1;
  } else {
    data.streak = 0;
  }
  data.bestStreak = Math.max(data.bestStreak || 0, data.streak || 0);
  data.best = data.best === undefined ? state.score : Math.max(data.best, state.score);
  localStorage.setItem('ww:record', JSON.stringify(data));
  renderRecord();
}

function renderRecord() {
  const d = readRecord();
  $('#stGames').textContent = d.games || 0;
  $('#stWins').textContent = d.wins || 0;
  $('#stBest').textContent = d.best === undefined ? '-' : d.best;
  $('#stStreak').textContent = d.streak || 0;
}

// --- benchmark --------------------------------------------------------------

async function runBenchmark() {
  const btn = $('#btnRunBench');
  btn.disabled = true;
  btn.textContent = 'Running...';
  el.benchChart.innerHTML = '<div class="bench-empty">Playing caves...</div>';
  el.benchTable.innerHTML = '';

  try {
    const data = await api('/api/benchmark', {
      agents: store.agents.map((a) => a.name),
      episodes: Number($('#bmEpisodes').value) || 150,
      size: Number($('#bmSize').value) || 4,
      pits: Number($('#bmPits').value) || 3,
    });
    renderBenchmark(data);
  } catch (err) {
    el.benchChart.innerHTML = '<div class="bench-empty">The benchmark failed.</div>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run';
  }
}

function renderBenchmark(data) {
  const rows = [...data.rows].sort((a, b) => b.avg_score - a.avg_score);
  const scores = rows.map((r) => r.avg_score);
  const lo = Math.min(0, ...scores);
  const hi = Math.max(1, ...scores);
  const span = hi - lo || 1;

  // Width is written inline so it is right the moment the bar exists; the
  // grow-in is a CSS animation, which keeps working when timers are throttled.
  el.benchChart.innerHTML = rows.map((r, i) => `
    <div class="bench-row rank-${i}">
      <div class="bl">${escapeHtml(r.label)}<small>${r.win_rate}% won, ${r.death_rate}% died${r.stuck_rate ? `, ${r.stuck_rate}% stuck` : ''}</small></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(4, ((r.avg_score - lo) / span) * 100).toFixed(1)}%">${r.avg_score}</div></div>
    </div>`).join('');

  const num = (v) => `<td class="${v > 0 ? 'pos' : v < 0 ? 'neg' : ''}">${v}</td>`;
  el.benchTable.innerHTML =
    '<thead><tr><th>agent</th><th>win %</th><th>death %</th><th title="never decided to leave">stuck %</th>'
    + '<th>avg score</th><th>median</th><th>best</th><th>worst</th><th>avg moves</th><th>seen</th></tr></thead><tbody>'
    + rows.map((r) => `<tr>
      <td>${escapeHtml(r.label)}</td>
      <td>${r.win_rate}</td>
      <td>${r.death_rate}</td>
      <td>${r.stuck_rate}</td>
      ${num(r.avg_score)}
      ${num(r.median_score)}
      ${num(r.best_score)}
      ${num(r.worst_score)}
      <td>${r.avg_steps}</td>
      <td>${r.avg_explored}</td>
    </tr>`).join('') + '</tbody>';

  const cfg = data.config;
  const capNote = data.capped_from
    ? ` (trimmed from ${data.capped_from} to keep the request quick)`
    : '';
  el.benchNote.textContent =
    `${cfg.episodes} caves per agent${capNote}, ${cfg.size}x${cfg.size}, ${cfg.pits} pits, same seeds for every agent.`;
}

// --- Q-learning training ----------------------------------------------------

async function runTraining() {
  const btn = $('#btnRunTrain');
  btn.disabled = true;
  btn.textContent = 'Training...';
  el.trainSummary.textContent = 'Running episodes. This takes a few seconds.';

  try {
    const data = await api('/api/rl/train', {
      episodes: Number($('#trEpisodes').value) || 12000,
      size: Number($('#trSize').value) || 4,
      pits: Number($('#trPits').value) || 3,
    });
    drawCurve(data.curve);
    const last = data.curve[data.curve.length - 1];
    const first = data.curve[0];
    const capNote = data.capped_from
      ? ` (trimmed from ${data.capped_from} to keep the request quick)`
      : '';
    el.trainSummary.innerHTML =
      `Trained <b>${data.episodes}</b> episodes${capNote} on ${data.config.size}x${data.config.size} with ${data.config.pits} pits. `
      + `Average score went from <b>${first.avg_score}</b> to <b>${last.avg_score}</b>, `
      + `visiting <b>${data.states_seen}</b> distinct states. `
      + `The agent bundled with the app was trained for <b>${data.shipped_episodes}</b> episodes; `
      + 'pick "Q-Learning Agent" in the sidebar to watch it play.';
  } catch (err) {
    el.trainSummary.textContent = 'Training failed.';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Train';
  }
}

function drawCurve(curve) {
  if (!curve || !curve.length) return;
  const W = 700;
  const H = 220;
  const pad = { l: 52, r: 34, t: 12, b: 26 };
  const scores = curve.map((p) => p.avg_score);
  const lo = Math.min(...scores, 0);
  const hi = Math.max(...scores, 0);
  const span = (hi - lo) || 1;
  const maxEp = curve[curve.length - 1].episode;

  const x = (ep) => pad.l + (ep / maxEp) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - lo) / span) * (H - pad.t - pad.b);
  const yEps = (e) => pad.t + (1 - e) * (H - pad.t - pad.b);

  const line = curve.map((p) => `${x(p.episode).toFixed(1)},${y(p.avg_score).toFixed(1)}`).join(' ');
  const eps = curve.map((p) => `${x(p.episode).toFixed(1)},${yEps(p.epsilon).toFixed(1)}`).join(' ');

  el.curve.setAttribute('viewBox', `0 0 ${W} ${H}`);
  el.curve.innerHTML = `
    <line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H - pad.b}"></line>
    <line class="axis" x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}"></line>
    <line class="zero" x1="${pad.l}" y1="${y(0).toFixed(1)}" x2="${W - pad.r}" y2="${y(0).toFixed(1)}"></line>
    <text x="4" y="${(y(hi) + 4).toFixed(1)}">${Math.round(hi)}</text>
    <text x="4" y="${(y(lo) + 4).toFixed(1)}">${Math.round(lo)}</text>
    <text x="4" y="${(y(0) + 4).toFixed(1)}">0</text>
    <text x="${pad.l}" y="${H - 8}">0</text>
    <text x="${W - pad.r - 40}" y="${H - 8}">${maxEp}</text>
    <text x="${W - pad.r - 46}" y="${H - 8 - 12}" style="font-size:9px">episodes</text>
    <polyline class="eps" points="${eps}"></polyline>
    <polyline class="line" points="${line}"></polyline>`;
}

// --- wiring -----------------------------------------------------------------

function wire() {
  $('#btnNew').addEventListener('click', () => newGame());
  $('#btnPlayAgain').addEventListener('click', () => { el.inpSeed.value = ''; newGame({ seed: '' }); });
  $('#btnRetrySeed').addEventListener('click', () => newGame({ seed: store.state?.seed ?? '' }));
  $('#btnReveal').addEventListener('click', async () => {
    render(await api('/api/reveal', {}));
    toast('Map revealed. The game is still running.');
  });

  $$('[data-dir]').forEach((b) => b.addEventListener('click', () => act('move', b.dataset.dir)));
  $$('[data-shoot]').forEach((b) => b.addEventListener('click', () => act('shoot', b.dataset.shoot)));
  $('#btnGrab').addEventListener('click', () => act('grab'));
  $('#btnClimb').addEventListener('click', () => act('climb'));

  el.btnStep.addEventListener('click', () => { stopAuto(); aiStep(); });
  el.btnHint.addEventListener('click', hint);
  el.btnSolve.addEventListener('click', solve);
  el.btnAuto.addEventListener('click', () => (store.playing ? stopAuto() : startAuto()));

  el.inpSpeed.addEventListener('input', () => {
    const ms = speedMs();
    el.speedLabel.textContent = ms < 200 ? 'fastest' : ms < 450 ? 'fast' : ms < 800 ? 'normal' : 'slow';
  });

  el.tglVision.addEventListener('change', () => el.board.classList.toggle('vision', el.tglVision.checked));
  el.tglNumbers.addEventListener('change', () => el.board.classList.toggle('no-numbers', !el.tglNumbers.checked));

  $$('#difficulty .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $$('#difficulty .chip').forEach((c) => c.classList.remove('is-active'));
      chip.classList.add('is-active');
      const d = store.difficulties?.[chip.dataset.diff];
      if (d) {
        el.inpSize.value = d.size;
        el.inpPits.value = d.pits;
      }
      el.inpSeed.value = '';
      newGame({ seed: '' });
    });
  });

  $('#btnCopySeed').addEventListener('click', async () => {
    const seed = store.state?.seed;
    if (seed === null || seed === undefined) return toast('This cave has no seed.');
    try {
      await navigator.clipboard.writeText(String(seed));
      toast(`Copied seed ${seed}.`);
    } catch {
      toast(`Seed: ${seed}`);
    }
  });

  $('#btnResetStats').addEventListener('click', () => {
    localStorage.removeItem('ww:record');
    renderRecord();
    toast('Record cleared.');
  });

  $('#btnSound').addEventListener('click', (e) => {
    sound.on = !sound.on;
    localStorage.setItem('ww:sound', sound.on ? 'on' : 'off');
    e.currentTarget.textContent = `Sound: ${sound.on ? 'on' : 'off'}`;
    e.currentTarget.setAttribute('aria-pressed', String(sound.on));
    if (sound.on) sound.move();
  });

  $$('.tab').forEach((tab) => tab.addEventListener('click', () => {
    $$('.tab').forEach((t) => t.classList.remove('is-active'));
    $$('.tab-panel').forEach((p) => p.classList.remove('is-active'));
    tab.classList.add('is-active');
    $(`[data-panel="${tab.dataset.tab}"]`).classList.add('is-active');
  }));

  $('#btnHelp').addEventListener('click', () => { $('#helpModal').hidden = false; });
  $('#btnBenchmark').addEventListener('click', () => { $('#benchModal').hidden = false; });
  $('#btnTrain').addEventListener('click', () => { $('#trainModal').hidden = false; });
  $('#btnRunBench').addEventListener('click', runBenchmark);
  $('#btnRunTrain').addEventListener('click', runTraining);
  $$('.modal').forEach((modal) => {
    modal.addEventListener('click', (e) => {
      if (e.target === modal || e.target.hasAttribute('data-close')) modal.hidden = true;
    });
  });

  document.addEventListener('keydown', (e) => {
    // e.target is not always an Element, so check before calling matches().
    if (e.target instanceof Element && e.target.matches('input, textarea, select')) return;
    if (e.key === 'Escape') {
      $$('.modal').forEach((m) => { m.hidden = true; });
      return;
    }
    if ($$('.modal').some((m) => !m.hidden)) return;

    const dirs = {
      ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
      w: 'up', s: 'down', a: 'left', d: 'right',
    };
    const dir = dirs[e.key] || dirs[e.key.toLowerCase()];
    if (dir) {
      e.preventDefault();
      act(e.shiftKey ? 'shoot' : 'move', dir);
      return;
    }
    switch (e.key.toLowerCase()) {
      case 'g': e.preventDefault(); act('grab'); break;
      case 'c': e.preventDefault(); act('climb'); break;
      case 'h': e.preventDefault(); hint(); break;
      case 'n': e.preventDefault(); newGame(); break;
      case ' ': e.preventDefault(); stopAuto(); aiStep(); break;
      case 'enter': e.preventDefault(); if (store.playing) stopAuto(); else startAuto(); break;
    }
  });

  // Cell size depends on the column width, so recompute whenever the board's
  // geometry can have changed.
  const reposition = () => {
    if (!store.state) return;
    fitCells(store.state.size);
    positionToken(store.state.agent[0], store.state.agent[1], store.state);
    drawTrail(store.state);
  };
  window.addEventListener('resize', reposition);
  if (window.ResizeObserver) new ResizeObserver(reposition).observe(el.board);
}

async function boot() {
  wire();
  el.board.classList.add('vision');
  $('#btnSound').textContent = `Sound: ${sound.on ? 'on' : 'off'}`;
  renderRecord();
  el.inpSpeed.dispatchEvent(new Event('input'));

  try {
    const meta = await api('/api/meta');
    store.agents = meta.agents;
    store.difficulties = meta.difficulties;
    renderAgents();
  } catch (err) {
    toast('Could not load the agent list.');
  }

  try {
    const payload = await api('/api/state');
    el.inpSize.value = payload.state.size;
    el.inpPits.value = payload.state.pits_count;
    el.inpSeed.value = payload.state.seed ?? '';
    render(payload, { reset: true });
  } catch (err) {
    toast('Could not reach the server.');
  }
}

boot();
