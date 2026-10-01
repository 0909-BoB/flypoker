const SUIT_SYMBOL = { c: "♣", d: "♦", h: "♥", s: "♠" };
const RED_SUITS = new Set(["d", "h"]);

// Flies pause before revealing a decision: longer when the spot is close
// (equity near 50%) or the call is a big share of their stack, plus a little
// randomness so the pace doesn't feel mechanical.
const THINK_BASE_MS = 2200;
function flyThinkMs(msg) {
  const uncertainty = 1 - Math.abs((msg.equity ?? 0.5) * 2 - 1);
  const stakes = Math.min(1, msg.commit_ratio ?? 0);
  return Math.round(THINK_BASE_MS + 1400 * uncertainty + 1200 * stakes + Math.random() * 800);
}
const RUNOUT_STREET_MS = 1200;
const HAND_RESULT_PAUSE_MS = 2600;

const ROLE_COLORS = {
  "in:equity": "#ff6fae",
  "in:pot_odds": "#5ec8ff",
  "in:spr": "#ffa94d",
  "in:commit": "#ff7a59",
  "in:depth": "#9be564",
  "in:rel_stack": "#4dabf7",
  "in:street": "#4dd9c0",
  "in:position": "#b892ff",
  "in:opp_aggression": "#d4e157",
  "out:action": "#ffe066",
  "relay": "#57628c",
};

const IDENTITIES = ["human", "fly1", "fly2", "fly3"];
// Neutral labels on purpose: each fly's personality is now sampled fresh
// (continuous, independent aggression/fold-tendency parameters -- see
// server.py's sample_personality_bias) every time a game starts, not one
// of a few fixed named archetypes, so a label like "aggressive" would be
// misleading more often than not.
const FLY_LABEL = { fly1: "果蠅一", fly2: "果蠅二", fly3: "果蠅三" };
const POS_LABELS = ["莊", "小盲", "大盲", "槍口"];
const POS_KEYS = ["btn", "sb", "bb", "utg"];
const STREET_LABEL = { preflop: "翻牌前", flop: "翻牌", turn: "轉牌", river: "河牌" };
const streetName = (s) => STREET_LABEL[s] || s;
const HAND_RANK_LABEL = {
  high_card: "高牌", pair: "一對", two_pair: "兩對", trips: "三條",
  straight: "順子", flush: "同花", full_house: "葫蘆", quads: "四條",
  straight_flush: "同花順",
};
let lastHandRank = undefined; // undefined so the very first render always "changes"

let ws = null;
let gameConfig = { startingStack: 2000, bigBlind: 20 }; // set by the setup screen before connect()
let currentSeats = null; // ["human","fly1","fly2","fly3"] in physical seat order, this hand
let lastStacks = { human: 2000, fly1: 2000, fly2: 2000, fly3: 2000 };
let lastBankroll = { human: 0, fly1: 0, fly2: 0, fly3: 0 };
let lastPot = 0;
let msgQueue = [];
let queueRunning = false;
let currentObs = null;
let focusedFly = null; // which fly's decision the circuit/brain panel is currently showing


const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const el = (id) => document.getElementById(id);
const seatEl = (identity) => document.getElementById(identity === "human" ? "seat-human" : "seat-" + identity);
const cardsElId = (identity) => (identity === "human" ? "human-cards" : "cards-" + identity);
const stackElId = (identity) => (identity === "human" ? "human-stack" : "stack-" + identity);

// "Ts" -> "10♠": the engine's compact card codes are never shown as-is.
function fmtCard(code) {
  const rank = code.slice(0, -1) === "T" ? "10" : code.slice(0, -1);
  const suit = code.slice(-1);
  return rank + (SUIT_SYMBOL[suit] || suit);
}

function cardNode(str, faceDown = false) {
  const div = document.createElement("div");
  if (faceDown) {
    div.className = "card back";
    return div;
  }
  const rank = str.slice(0, -1);
  const suit = str.slice(-1);
  div.className = "card" + (RED_SUITS.has(suit) ? " red" : "");
  div.textContent = fmtCard(str);
  return div;
}

function renderCards(containerId, cardStrs, faceDown = false) {
  const c = el(containerId);
  c.innerHTML = "";
  (cardStrs || []).forEach((s) => c.appendChild(cardNode(s, faceDown)));
}

function flipRevealCards(containerId, cardStrs) {
  const c = el(containerId);
  c.innerHTML = "";
  (cardStrs || []).forEach((s) => {
    const node = cardNode(s, false);
    node.classList.add("flip-in");
    c.appendChild(node);
  });
}

function log(text, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  const panel = el("log");
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
}

function showBadge(target, text, isFold = false) {
  if (!target) return;
  const badge = document.createElement("div");
  badge.className = "action-badge" + (isFold ? " fold" : "");
  badge.textContent = text;
  target.appendChild(badge);
  setTimeout(() => badge.remove(), 1700);
}

function bumpNumber(target, newValue, prevValue) {
  target.textContent = newValue;
  target.classList.remove("bump", "up", "down");
  void target.offsetWidth;
  target.classList.add("bump");
  if (newValue > prevValue) target.classList.add("up");
  else if (newValue < prevValue) target.classList.add("down");
  setTimeout(() => target.classList.remove("bump", "up", "down"), 350);
}

function setStack(identity, value) {
  const node = el(stackElId(identity));
  if (!node) return;
  const prev = lastStacks[identity];
  if (value !== prev) bumpNumber(node, value, prev);
  else node.textContent = value;
  lastStacks[identity] = value;
}

function updateBankroll(bankroll) {
  IDENTITIES.forEach((id) => {
    const node = el("bankroll-" + id);
    const prev = lastBankroll[id];
    const next = bankroll[id];
    if (next !== prev) bumpNumber(node, next, prev);
    else node.textContent = next;
    lastBankroll[id] = next;
  });
}

function applyStacks(stacksBySeat) {
  if (!stacksBySeat || !currentSeats) return;
  currentSeats.forEach((identity, seatIdx) => setStack(identity, stacksBySeat[seatIdx]));
}

function setPot(value) {
  const span = el("pot-amount");
  if (value !== lastPot) bumpNumber(span, value, lastPot);
  else span.textContent = value;
  lastPot = value;
}

// SAN: how composed each fly is (see sanity.py). Colour runs blue -> amber -> red as it falls.
function sanLabel(v) {
  return v >= 80 ? "冷靜" : v >= 55 ? "緊張" : v >= 30 ? "動搖" : "崩潰";
}
function sanColor(v) {
  return v >= 80 ? "#2563eb" : v >= 55 ? "#d99a00" : v >= 30 ? "#e8743b" : "#e0455a";
}
function updateSan(fly, value, reason) {
  const row = el("san-" + fly);
  if (!row || value == null) return;
  const fill = row.querySelector(".san-fill");
  fill.style.width = `${Math.max(2, value)}%`;
  fill.style.background = sanColor(value);
  row.querySelector(".san-val").textContent = Math.round(value);
  row.querySelector(".san-state").textContent = sanLabel(value);
  if (reason !== undefined) {
    el("san-reason").innerHTML = reason ? `${FLY_LABEL[fly]}:<b>${reason}</b>` : "";
  }
}
function updateAllSan(sanByFly) {
  if (sanByFly) Object.entries(sanByFly).forEach(([fly, v]) => updateSan(fly, v));
}

function setHandRank(rank) {
  const node = el("hand-rank");
  if (!node) return;
  if (!rank) {
    node.hidden = true;
    lastHandRank = rank;
    return;
  }
  node.textContent = HAND_RANK_LABEL[rank] || rank;
  node.hidden = false;
  if (rank !== lastHandRank) {
    node.classList.remove("bump");
    void node.offsetWidth;
    node.classList.add("bump");
    setTimeout(() => node.classList.remove("bump"), 350);
  }
  lastHandRank = rank;
}

function setCircuitFocus(identity) {
  focusedFly = identity;
  ["fly1", "fly2", "fly3"].forEach((id) => el("san-" + id).classList.toggle("focused", id === identity));
  el("circuit-focus").innerHTML = identity ? `目前顯示：<b>${FLY_LABEL[identity] || identity}</b> 的神經活動` : "目前顯示：-";
}

function setActionsEnabled(enabled, legalActions = []) {
  document.querySelectorAll("#actions button[data-action]").forEach((btn) => {
    const action = btn.dataset.action;
    btn.disabled = !enabled || (legalActions.length > 0 && !legalActions.includes(action));
  });
  el("open-raise-btn").disabled = !enabled || (legalActions.length > 0 && !legalActions.includes("RAISE_SMALL"));
  if (!enabled) closeRaisePanel();
}

function setStatus(text, cls = "") {
  const s = el("status");
  s.className = "status" + (cls ? " " + cls : "");
  s.innerHTML = text;
}

function setThinkingPill(identity) {
  ["fly1", "fly2", "fly3"].forEach((id) => {
    const pill = el("thinking-" + id);
    if (pill) pill.hidden = id !== identity;
  });
}

function startFlyThinking(identity) {
  document.querySelectorAll(".fly-seat.thinking").forEach((n) => n.classList.remove("thinking"));
  seatEl("human").classList.remove("active-turn");
  const target = identity ? seatEl(identity) : null;
  if (target) target.classList.add("thinking");
  setThinkingPill(identity);
  if (window.brain3d) window.brain3d.setThinking(true);
  document.querySelector(".brain-panel").classList.add("thinking");
  const label = identity ? FLY_LABEL[identity] || identity : "果蠅";
  setStatus(`<span class="spinner"></span>${label}思考中...`, "waiting");
}

function stopAllThinking() {
  document.querySelectorAll(".fly-seat.thinking").forEach((n) => n.classList.remove("thinking"));
  setThinkingPill(null);
  if (window.brain3d) window.brain3d.setThinking(false);
  document.querySelector(".brain-panel").classList.remove("thinking");
}

function updatePositionBadges(seats) {
  IDENTITIES.forEach((identity) => {
    const badge = el("pos-" + identity);
    if (!badge) return;
    const idx = seats.indexOf(identity);
    const label = POS_LABELS[idx] || "";
    badge.textContent = label;
    badge.className = "pos-badge" + (label ? " " + POS_KEYS[idx] : "");
  });
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    // The server waits for this before dealing anything (see
    // server.py's ws_endpoint) -- it re-validates server-side too, this
    // isn't the only guard against a bad starting_stack/big_blind.
    ws.send(JSON.stringify({
      type: "start_game",
      starting_stack: gameConfig.startingStack,
      big_blind: gameConfig.bigBlind,
    }));
    setStatus("已連線,等待發牌...");
  };
  ws.onclose = () => { setStatus("連線中斷"); setActionsEnabled(false); };
  ws.onerror = () => setStatus("連線錯誤");

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "circuit") {
      setupCircuit(msg.nodes, msg.edges);
      return;
    }
    msgQueue.push(msg);
    processQueue();
  };
}

// --- Live circuit view ---------------------------------------------------
// 3D brain (brain3d.js): real male-cns:v1.0 brain mesh with each circuit
// neuron drawn at its real position; spikes flash the matching dots.

function setupCircuit(nodes, edges) {
  window.__pendingCircuit = { nodes, edges, colors: ROLE_COLORS };
  if (window.brain3d) window.brain3d.setNodes(nodes, ROLE_COLORS, edges);
  el("circuit-note").textContent = `${nodes.length} 神經元`;
}

// Real spike events ([neuron, ms, count]) stretched over `durationMs`.
function animateCircuitSpikes(spikes, durationMs) {
  if (window.brain3d) window.brain3d.play(spikes, durationMs);
}

async function processQueue() {
  if (queueRunning) return;
  queueRunning = true;
  while (msgQueue.length) {
    const msg = msgQueue.shift();
    try {
      await handleMessage(msg);
    } catch (err) {
      // One bad message must not stop every later one (queueRunning would
      // stay true forever and the table would look frozen).
      console.error("failed to handle message", msg && msg.type, err);
      log(`  [畫面錯誤] ${msg && msg.type}: ${err.message}`, "lose");
    }
  }
  queueRunning = false;
}

async function handleMessage(msg) {
  {
    if (msg.type === "fly_decision") {
      // Show that fly's thinking pill *before* the pause, so it's visible
      // during the "thinking" delay and reads as "this is who's deciding
      // right now" -- showing it only after already rendering the
      // decision (the previous order) made the pill appear right as that
      // fly's result was revealed, which reads as "still thinking" about
      // a decision it just finished.
      const thinkMs = flyThinkMs(msg);
      startFlyThinking(msg.fly);
      // The brain "works" while the fly thinks: spread its spike animation
      // over the whole pause instead of playing it after the answer.
      updateSan(msg.fly, msg.san, msg.san_reason);
      setCircuitFocus(msg.fly);
      animateCircuitSpikes(msg.spikes, thinkMs);
      await sleep(thinkMs);
      if (window.brain3d) window.brain3d.reveal(); // output neurons flare as the answer lands
    } else if (msg.type === "runout_street") {
      animateCircuitSpikes(msg.spikes, RUNOUT_STREET_MS);
      await sleep(RUNOUT_STREET_MS);
    }
    renderMessage(msg);
    if (msg.type === "your_turn" || msg.type === "hand_result") {
      stopAllThinking();
    } else if (msg.type === "runout_street") {
      stopAllThinking();
      document.querySelector(".brain-panel").classList.add("thinking");
      setStatus(`全下攤牌,開出${streetName(msg.street)}...`, "waiting");
    }
    if (msg.type === "hand_result") {
      await sleep(HAND_RESULT_PAUSE_MS);
    }
  }
}

function renderMessage(msg) {
  switch (msg.type) {
    case "new_hand": {
      currentSeats = msg.seats;
      el("hand-number").textContent = msg.hand_number;
      renderCards("human-cards", msg.human_hole, false);
      ["fly1", "fly2", "fly3"].forEach((id) => renderCards(cardsElId(id), ["??", "??"], true));
      renderCards("board-cards", []);
      lastPot = msg.small_blind + msg.big_blind;
      el("pot-amount").textContent = lastPot;
      // Chips carry over between hands now, so this hand's opening stacks
      // come straight from the server (no reset to the buy-in amount).
      msg.seats.forEach((id, i) => {
        lastStacks[id] = msg.stacks[i];
        el(stackElId(id)).textContent = String(msg.stacks[i]);
      });
      if (msg.bankroll) updateBankroll(msg.bankroll);
      updateAllSan(msg.san);
      el("san-reason").innerHTML = "";
      el("street-label").textContent = streetName("preflop");
      el("fly-decision").textContent = "";
      setHandRank(null);
      setCircuitFocus(null);
      document.querySelectorAll(".seat").forEach((s) => s.classList.remove("active-turn", "thinking"));
      updatePositionBadges(msg.seats);
      log(`── 第 ${msg.hand_number} 手 (座位: ${msg.seats.map((s) => (s === "human" ? "你" : FLY_LABEL[s])).join(" / ")}) ──`, "hand-sep");
      setActionsEnabled(false);
      break;
    }
    case "your_turn": {
      const obs = msg.obs;
      currentObs = obs;
      el("street-label").textContent = streetName(obs.street);
      renderCards("board-cards", obs.board);
      setHandRank(obs.hand_rank);
      setPot(obs.pot);
      applyStacks(msg.stacks);
      setActionsEnabled(true, obs.legal_actions);
      el("check-call-btn").textContent = obs.to_call > 0 ? `跟注 ${obs.to_call}` : "過牌";
      document.querySelectorAll(".seat").forEach((s) => s.classList.remove("active-turn"));
      seatEl("human").classList.add("active-turn");
      setStatus(obs.to_call > 0 ? `輪到你:需跟注 ${obs.to_call}` : "輪到你:可過牌或下注", "your-turn");
      currentTurnId = msg.turn_id;
      startTurnTimer(msg.time_limit || 15);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "turn_ack", turn_id: msg.turn_id }));
      }
      break;
    }
    case "runout_street": {
      el("street-label").textContent = streetName(msg.street);
      renderCards("board-cards", msg.board);
      setHandRank(msg.hand_rank);
      applyStacks(msg.stacks);
      el("san-reason").innerHTML = ""; // no decision here, so no stress reason to show
      setCircuitFocus(msg.fly);
      el("fly-decision").innerHTML = `${streetName(msg.street)} · ${FLY_LABEL[msg.fly] || msg.fly} 的神經活動(全下後自動開牌,勝率約 ${(msg.equity * 100).toFixed(0)}%)`;
      // Not logged here: hand_result dumps the backend's complete,
      // correctly-ordered log for the whole hand (the only place the
      // human's own actions get logged too, since sendAction() doesn't
      // log them locally), so logging this live as well would both
      // duplicate it and, worse, make the log panel appear to jump
      // backwards in time once that full dump lands after this line.
      break;
    }
    case "fly_decision": {
      setActionsEnabled(false);
      document.querySelectorAll(".seat").forEach((s) => s.classList.remove("active-turn"));
      const target = seatEl(msg.fly);
      if (target) target.classList.add("active-turn");
      if (msg.board) {
        // A fly can be first to act on a new street (not just the human),
        // so the board/street/hand-rank need to sync here too -- otherwise
        // they'd lag one message behind until the human's own turn.
        el("street-label").textContent = streetName(msg.street);
        renderCards("board-cards", msg.board);
        setHandRank(msg.hand_rank);
      }
      if (msg.stacks) applyStacks(msg.stacks);
      updateSan(msg.fly, msg.san, msg.san_reason);
      setCircuitFocus(msg.fly);
      const label = FLY_LABEL[msg.fly] || msg.fly;
      el("fly-decision").innerHTML =
        `${streetName(msg.street)} · ${label} <b>${actionLabel(msg.action)}</b>` +
        `(勝率約 ${(msg.equity * 100).toFixed(0)}%,` +
        `${msg.depth_bb < 15 ? "短碼," : ""}<span style="white-space:nowrap">剩 ${msg.depth_bb} 個大盲</span>` +
        `${msg.commit_ratio >= 0.5 ? `,跟注佔籌碼 ${(msg.commit_ratio * 100).toFixed(0)}%` : ""})`;
      // See the runout_street case above for why this isn't also logged
      // here -- hand_result's full backend dump is the single source of
      // truth for the log panel now.
      showBadge(target, actionLabel(msg.action), msg.action === "FOLD");
      break;
    }
    case "hand_result": {
      stopTurnTimer();
      setActionsEnabled(false);
      msg.log.forEach((line) => formatLogLine(line).forEach((text) => log("  " + text)));
      if (msg.showdown && msg.showdown_holes) {
        for (const [identity, cards] of Object.entries(msg.showdown_holes)) {
          flipRevealCards(cardsElId(identity), cards);
        }
      }
      const cls = msg.human_payoff > 0 ? "win" : msg.human_payoff < 0 ? "lose" : "";
      log(`  你這手 ${msg.human_payoff > 0 ? "+" : ""}${msg.human_payoff}`, cls);
      if (msg.learning) {
        const l = msg.learning;
        log(`  果蠅已累積學習 ${l.hands} 手(更新 ${l.updates} 次${l.healthy ? "" : ",偏差過大已暫停"})`, "learn-note");
      }
      updateBankroll(msg.bankroll);
      updateAllSan(msg.san);
      if (msg.final_stacks) IDENTITIES.forEach((id) => setStack(id, msg.final_stacks[id]));
      document.querySelectorAll(".seat").forEach((s) => s.classList.remove("active-turn", "thinking"));
      if (msg.payoffs) {
        for (const [identity, delta] of Object.entries(msg.payoffs)) {
          if (delta === 0) continue;
          showBadge(seatEl(identity), `${delta > 0 ? "+" : ""}${delta}`);
        }
      }
      setStatus("這手結束,準備下一手...");
      break;
    }
    case "turn_timeout": {
      stopTurnTimer();
      setActionsEnabled(false);
      seatEl("human").classList.remove("active-turn");
      const what = msg.action === "FOLD" ? "蓋牌" : "過牌";
      showBadge(seatEl("human"), `時間到 · ${what}`, msg.action === "FOLD");
      setStatus(`時間到,自動${what}`, "waiting");
      break;
    }
    case "busted": {
      openRebuyDialog(msg);
      break;
    }
    case "fly_rebuy": {
      log(`  ${FLY_LABEL[msg.fly] || msg.fly} 籌碼輸光,補碼 ${msg.amount}`, "hand-sep");
      break;
    }
    case "left": {
      leaveTable();
      break;
    }
    default: {
      // A message type this script doesn't know means it's out of date
      // with the server -- say so instead of silently freezing.
      setActionsEnabled(false);
      setStatus(`頁面版本過舊,請重新整理 (未知訊息: ${msg.type})`, "error");
      break;
    }
    case "error": {
      if (!el("rebuy-dialog").hidden) {
        showRebuyError(msg.message);
        break;
      }
      stopAllThinking();
      setActionsEnabled(false);
      setStatus(`發生錯誤,請重新整理頁面: ${msg.message}`, "error");
      log(`  [錯誤] ${msg.message}`, "lose");
      break;
    }
  }
}

function actionLabel(name) {
  return {
    FOLD: "蓋牌", CHECK_CALL: "過牌/跟注", RAISE_SMALL: "加注",
    RAISE_BIG: "加注", ALL_IN: "全下",
  }[name] || name;
}

function identityLabel(identity) {
  return identity === "human" ? "你" : (FLY_LABEL[identity] || identity);
}

// The engine logs plain "seat0"/"seat 1" (it doesn't know about fly names
// or the human -- that mapping only exists in the UI layer), which reads
// fine as a technical log but not as something a player wants to read.
// Swap seat numbers for this hand's actual identities before displaying.
function translateSeatsInLogLine(line) {
  if (!currentSeats) return line;
  return line.replace(/seat\s?(\d)/g, (match, digit) => {
    const identity = currentSeats[Number(digit)];
    return identity ? identityLabel(identity) : match;
  });
}

let currentTurnId = null;
let turnTimer = null;

function stopTurnTimer() {
  if (turnTimer) clearInterval(turnTimer);
  turnTimer = null;
  el("turn-timer").hidden = true;
}

function startTurnTimer(limitSec) {
  stopTurnTimer();
  const box = el("turn-timer"), fill = el("turn-timer-fill"), text = el("turn-timer-text");
  const end = performance.now() + limitSec * 1000;
  box.hidden = false;
  box.classList.remove("urgent");
  const tick = () => {
    const left = Math.max(0, (end - performance.now()) / 1000);
    fill.style.width = `${(left / limitSec) * 100}%`;
    text.textContent = `${Math.ceil(left)} 秒`;
    box.classList.toggle("urgent", left <= 5);
    if (left <= 0) {
      clearInterval(turnTimer);
      turnTimer = null;
    }
  };
  tick();
  turnTimer = setInterval(tick, 100);
}

// The engine writes its hand history as compact developer strings
// ("[flop] seat1 raises to 250 (RAISE_BIG)", "showdown board=[Tc, Kc] ...").
// Turn each into plain Chinese for the log panel; one engine line can become
// several display lines.
const HAND_NAME_ZH = HAND_RANK_LABEL;
function seatName(digit) {
  const identity = currentSeats && currentSeats[Number(digit)];
  return identity ? identityLabel(identity) : `座位${digit}`;
}
function fmtCards(list) {
  return list.split(",").map((c) => c.trim()).filter(Boolean).map(fmtCard).join(" ");
}
function formatLogLine(line) {
  let m;
  if ((m = line.match(/^\[(\w+)\] seat\s?(\d) (.+)$/))) {
    const [, street, seat, rest] = m;
    const who = seatName(seat);
    const head = `${streetName(street)} · ${who} `;
    if (rest === "folds") return [head + "蓋牌"];
    if (rest === "checks") return [head + "過牌"];
    if ((m = rest.match(/^calls (\d+)$/))) return [`${head}跟注 ${m[1]}`];
    if ((m = rest.match(/^raises to (\d+)(?: \((\w+)\))?$/))) {
      return [`${head}加注到 ${m[1]}${m[2] === "ALL_IN" ? "(全下)" : ""}`];
    }
  }
  if ((m = line.match(/^seat\s?(\d) wins uncontested pot (\d+)/))) {
    return [`${seatName(m[1])} 贏得底池 ${m[2]}(其他人都蓋牌)`];
  }
  if ((m = line.match(/^showdown board=\[(.*?)\] (.*) pots=\[(.*)\] winners_by_pot=(.*)$/))) {
    const [, board, hands, pots, winners] = m;
    const out = [`攤牌 · 公共牌 ${fmtCards(board)}`];
    for (const h of hands.matchAll(/seat(\d)=\[(.*?)\]\((\w+)\)/g)) {
      out.push(`  ${seatName(h[1])}:${fmtCards(h[2])} — ${HAND_NAME_ZH[h[3]] || h[3]}`);
    }
    let winnerLists = [];
    try { winnerLists = JSON.parse(winners); } catch (e) { /* leave empty */ }
    const amounts = [...pots.matchAll(/\((\d+), \[/g)].map((x) => x[1]);
    amounts.forEach((amount, i) => {
      const names = (winnerLists[i] || []).map(seatName).join("、");
      const label = amounts.length > 1 ? (i === 0 ? "主池" : `邊池${i}`) : "底池";
      out.push(`  ${label} ${amount} → ${names} 贏得`);
    });
    return out;
  }
  return [translateSeatsInLogLine(line)];
}

function sendAction(action, amount) {
  stopTurnTimer();
  const badgeText = amount != null ? `加注到 ${amount}` : actionLabel(action);
  showBadge(seatEl("human"), badgeText, action === "FOLD");
  const payload = { type: "action", action, turn_id: currentTurnId };
  if (amount != null) payload.amount = amount;
  ws.send(JSON.stringify(payload));
  setActionsEnabled(false);
  seatEl("human").classList.remove("active-turn");
}

document.querySelectorAll("#actions button[data-action]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    sendAction(btn.dataset.action);
  });
});

// --- Raise slider panel: mirrors the bet-sizing UX of mainstream poker
// clients (quick pot-relative presets + a free-drag slider/number input) ---
const raisePanel = el("raise-panel");
const raiseSlider = el("raise-slider");
const raiseNumber = el("raise-number");
const raiseAmountLabel = el("raise-amount-label");

function closeRaisePanel() {
  raisePanel.hidden = true;
}

function clampRaise(value) {
  const min = Number(raiseSlider.min), max = Number(raiseSlider.max);
  return Math.min(max, Math.max(min, Math.round(value)));
}

// Sets slider, box and label to `value` (clamped to the legal range).
function setRaiseAmount(value) {
  const clamped = clampRaise(value);
  raiseSlider.value = clamped;
  raiseNumber.value = clamped;
  raiseAmountLabel.textContent = clamped;
  raiseNumber.classList.remove("out-of-range");
}

el("open-raise-btn").addEventListener("click", () => {
  if (el("open-raise-btn").disabled || !currentObs) return;
  const min = currentObs.min_raise;
  const max = currentObs.max_raise;
  raiseSlider.min = min;
  raiseSlider.max = max;
  raiseNumber.min = min;
  raiseNumber.max = max;
  el("raise-range").textContent = `可下 ${min} ~ ${max}`;
  setRaiseAmount(currentObs.raise_presets.half_pot ?? min);
  raisePanel.hidden = false;
});

el("raise-cancel").addEventListener("click", closeRaisePanel);

raiseSlider.addEventListener("input", () => setRaiseAmount(raiseSlider.value));
// Typing: follow the number with the slider and the label, but never rewrite
// the box itself mid-keystroke (clamping "1" up to the minimum on every key
// made it impossible to type a multi-digit amount). The box snaps into the
// legal range when you finish (Enter / click away).
raiseNumber.addEventListener("input", () => {
  if (raiseNumber.value === "") return;
  const typed = Number(raiseNumber.value);
  if (!Number.isFinite(typed)) return;
  const clamped = clampRaise(typed);
  raiseSlider.value = clamped;
  raiseAmountLabel.textContent = clamped;
  raiseNumber.classList.toggle("out-of-range", typed !== clamped);
});
raiseNumber.addEventListener("change", () => {
  setRaiseAmount(raiseNumber.value === "" ? raiseSlider.value : raiseNumber.value);
});
raiseNumber.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    setRaiseAmount(raiseNumber.value === "" ? raiseSlider.value : raiseNumber.value);
    el("raise-confirm").click();
  }
});

document.querySelectorAll("#raise-presets button").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (!currentObs) return;
    const preset = btn.dataset.preset;
    const value = preset === "max" ? currentObs.max_raise : currentObs.raise_presets[preset];
    setRaiseAmount(value);
  });
});

el("raise-confirm").addEventListener("click", () => {
  if (!currentObs) return;
  const amount = clampRaise(raiseNumber.value === "" ? raiseSlider.value : Number(raiseNumber.value));
  closeRaisePanel();
  if (amount >= currentObs.max_raise) {
    sendAction("ALL_IN");
  } else {
    sendAction("RAISE_SMALL", amount);
  }
});

setActionsEnabled(false);

// --- Setup screen: collects starting stack / big blind before the table
// ever opens a connection, instead of dealing immediately on page load ---
function showSetupError(message) {
  const box = el("setup-error");
  box.textContent = message;
  box.hidden = false;
}

function validateSetup(stack, bb) {
  if (!Number.isInteger(stack) || stack <= 0) return "買入籌碼必須是大於 0 的整數";
  if (!Number.isInteger(bb) || bb <= 0) return "大盲注必須是大於 0 的整數";
  if (bb > stack) return "大盲注不能大於買入籌碼";
  return null;
}

el("setup-start-btn").addEventListener("click", () => {
  const stack = Math.trunc(Number(el("setup-stack").value));
  const bb = Math.trunc(Number(el("setup-bb").value));
  const err = validateSetup(stack, bb);
  if (err) {
    showSetupError(err);
    return;
  }
  gameConfig = { startingStack: stack, bigBlind: bb };
  el("setup-screen").hidden = true;
  el("game-screen").hidden = false;
  connect();
});

// --- Rebuy dialog: shown when the human's stack hits 0. Rebuy resumes the
// game with a fresh stack; leaving closes the table and returns to setup. ---
function showRebuyError(message) {
  const box = el("rebuy-error");
  box.textContent = message;
  box.hidden = false;
}

function openRebuyDialog(msg) {
  stopAllThinking();
  setActionsEnabled(false);
  el("rebuy-amount").value = msg.suggested;
  el("rebuy-amount").min = msg.min;
  el("rebuy-amount").max = msg.max;
  el("rebuy-error").hidden = true;
  el("rebuy-dialog").hidden = false;
  setStatus("籌碼輸光了,補碼繼續或離桌", "waiting");
}

el("rebuy-confirm").addEventListener("click", () => {
  const amount = Math.trunc(Number(el("rebuy-amount").value));
  const min = Number(el("rebuy-amount").min), max = Number(el("rebuy-amount").max);
  if (!Number.isInteger(amount) || amount < min || amount > max) {
    showRebuyError(`補碼金額必須在 ${min} 到 ${max} 之間`);
    return;
  }
  ws.send(JSON.stringify({ type: "rebuy", amount }));
  el("rebuy-dialog").hidden = true;
  setStatus("補碼完成,準備下一手...");
});

el("rebuy-leave").addEventListener("click", () => {
  ws.send(JSON.stringify({ type: "leave" }));
});

function leaveTable() {
  stopTurnTimer();
  updateAllSan({ fly1: 100, fly2: 100, fly3: 100 });
  el("rebuy-dialog").hidden = true;
  if (ws) ws.close();
  msgQueue = [];
  el("game-screen").hidden = true;
  el("setup-screen").hidden = false;
  lastBankroll = { human: 0, fly1: 0, fly2: 0, fly3: 0 };
  lastStacks = { human: 0, fly1: 0, fly2: 0, fly3: 0 };
  log("── 你已離桌 ──", "hand-sep");
}
