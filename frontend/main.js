const SUIT_SYMBOL = { c: "♣", d: "♦", h: "♥", s: "♠" };
const RED_SUITS = new Set(["d", "h"]);

const ROLE_LABELS = {
  "in:equity": "equity",
  "in:pot_odds": "pot odds",
  "in:spr": "stack/pot",
  "in:street": "street",
  "in:position": "position",
  "in:opp_aggression": "opp 壓力",
  "out:action": "輸出(決策)",
  "relay": "relay pool",
};
const ROLE_ORDER = Object.keys(ROLE_LABELS);
const BAR_MAX = { "relay": 6000, "out:action": 60 };
const DEFAULT_IN_MAX = 350;
const MIN_THINK_MS = 550;

let ws = null;
let humanSeat = 0;
let lastStacks = { human: 2000, fly: 2000 };
let lastPot = 0;
let msgQueue = [];
let queueRunning = false;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const el = (id) => document.getElementById(id);
const humanSeatEl = () => document.querySelector(".human-seat");
const flySeatEl = () => document.querySelector(".fly-seat");

function cardNode(str, faceDown = false) {
  const div = document.createElement("div");
  if (faceDown) {
    div.className = "card back";
    return div;
  }
  const rank = str.slice(0, -1);
  const suit = str.slice(-1);
  div.className = "card" + (RED_SUITS.has(suit) ? " red" : "");
  div.textContent = rank + (SUIT_SYMBOL[suit] || suit);
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

function showBadge(seatEl, text, isFold = false) {
  if (!seatEl) return;
  const badge = document.createElement("div");
  badge.className = "action-badge" + (isFold ? " fold" : "");
  badge.textContent = text;
  seatEl.appendChild(badge);
  setTimeout(() => badge.remove(), 1700);
}

function bumpNumber(target, newValue, prevValue) {
  target.textContent = newValue;
  target.classList.remove("bump", "up", "down");
  void target.offsetWidth; // restart animation
  target.classList.add("bump");
  if (newValue > prevValue) target.classList.add("up");
  else if (newValue < prevValue) target.classList.add("down");
  setTimeout(() => target.classList.remove("bump", "up", "down"), 350);
}

function setStack(who, value) {
  const id = who === "human" ? "human-stack" : "fly-stack";
  const prev = lastStacks[who];
  if (value !== prev) bumpNumber(el(id), value, prev);
  else el(id).textContent = value;
  lastStacks[who] = value;
}

function setPot(value) {
  const span = el("pot-amount");
  if (value !== lastPot) bumpNumber(span, value, lastPot);
  else span.textContent = value;
  lastPot = value;
}

function updateBrainGrid(activity) {
  const grid = el("brain-grid");
  grid.innerHTML = "";
  ROLE_ORDER.forEach((role) => {
    const count = activity[role] ?? 0;
    const max = BAR_MAX[role] || DEFAULT_IN_MAX;
    const pct = Math.min(100, (count / max) * 100);
    const row = document.createElement("div");
    row.className = "brain-row" + (role.startsWith("out:") ? " out" : "");
    row.innerHTML = `
      <div class="label">${ROLE_LABELS[role] || role}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div class="count">${count}</div>
    `;
    grid.appendChild(row);
  });
}

function setActionsEnabled(enabled, legalActions = []) {
  document.querySelectorAll("#actions button").forEach((btn) => {
    const action = btn.dataset.action;
    btn.disabled = !enabled || (legalActions.length > 0 && !legalActions.includes(action));
  });
}

function setStatus(text, cls = "") {
  const s = el("status");
  s.className = "status" + (cls ? " " + cls : "");
  s.innerHTML = text;
}

function startFlyThinking() {
  flySeatEl().classList.add("thinking");
  humanSeatEl().classList.remove("active-turn");
  document.querySelector(".brain-panel").classList.add("thinking");
  setStatus('<span class="spinner"></span>果蠅思考中...', "waiting");
}

function stopFlyThinking() {
  flySeatEl().classList.remove("thinking");
  document.querySelector(".brain-panel").classList.remove("thinking");
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setStatus("已連線,等待發牌...");
  ws.onclose = () => { setStatus("連線中斷"); setActionsEnabled(false); };
  ws.onerror = () => setStatus("連線錯誤");

  ws.onmessage = (event) => {
    msgQueue.push(JSON.parse(event.data));
    processQueue();
  };
}

// Messages are rendered one at a time, in arrival order, with a fixed pause
// before each fly_decision. Earlier this used a setTimeout per message with
// a shared "when did thinking start" timestamp; when two fly_decision
// messages arrived close together (fly acts first on a new street, with no
// human action in between) their independently-scheduled timeouts could
// fire out of order and leave the UI re-armed into a "thinking" state with
// nothing left to clear it. A single serial queue makes that ordering bug
// structurally impossible.
async function processQueue() {
  if (queueRunning) return;
  queueRunning = true;
  while (msgQueue.length) {
    const msg = msgQueue.shift();
    if (msg.type === "fly_decision") {
      await sleep(MIN_THINK_MS);
    }
    renderMessage(msg);
    if (msg.type === "your_turn" || msg.type === "hand_result") {
      stopFlyThinking();
    } else {
      startFlyThinking();
    }
  }
  queueRunning = false;
}

function renderMessage(msg) {
  switch (msg.type) {
    case "new_hand": {
      humanSeat = msg.human_seat;
      el("hand-number").textContent = msg.hand_number;
      renderCards("human-cards", msg.human_hole, false);
      renderCards("fly-cards", ["??", "??"], true);
      renderCards("board-cards", []);
      lastPot = msg.small_blind + msg.big_blind;
      el("pot-amount").textContent = lastPot;
      lastStacks = { human: 2000, fly: 2000 };
      el("human-stack").textContent = "2000";
      el("fly-stack").textContent = "2000";
      el("street-label").textContent = "preflop";
      el("fly-decision").textContent = "";
      humanSeatEl().classList.remove("active-turn");
      flySeatEl().classList.remove("active-turn");
      log(`── 第 ${msg.hand_number} 手 (你的座位 ${msg.human_seat === 0 ? "按鈕/小盲" : "大盲"}) ──`, "hand-sep");
      setActionsEnabled(false);
      break;
    }
    case "your_turn": {
      const obs = msg.obs;
      el("street-label").textContent = obs.street;
      renderCards("board-cards", obs.board);
      setPot(obs.pot);
      setStack("human", obs.my_stack);
      setStack("fly", obs.opp_stack);
      setActionsEnabled(true, obs.legal_actions);
      humanSeatEl().classList.add("active-turn");
      flySeatEl().classList.remove("active-turn");
      setStatus(obs.to_call > 0 ? `輪到你:需跟注 ${obs.to_call}` : "輪到你:可過牌或下注", "your-turn");
      break;
    }
    case "fly_decision": {
      setActionsEnabled(false);
      updateBrainGrid(msg.activity);
      el("fly-decision").innerHTML =
        `[${msg.street}] 果蠅 <b>${actionLabel(msg.action)}</b>` +
        ` (勝率估計 ${(msg.equity * 100).toFixed(0)}%)`;
      log(`  果蠅[${msg.street}] ${actionLabel(msg.action)} (equity=${msg.equity.toFixed(2)})`);
      showBadge(flySeatEl(), actionLabel(msg.action), msg.action === "FOLD");
      break;
    }
    case "hand_result": {
      setActionsEnabled(false);
      msg.log.forEach((line) => log("  " + line));
      if (msg.showdown && msg.fly_hole) {
        flipRevealCards("fly-cards", msg.fly_hole);
      }
      const cls = msg.human_payoff > 0 ? "win" : msg.human_payoff < 0 ? "lose" : "";
      log(`  你這手 ${msg.human_payoff > 0 ? "+" : ""}${msg.human_payoff}`, cls);
      const humanBankrollEl = el("human-bankroll");
      const flyBankrollEl = el("fly-bankroll");
      bumpNumber(humanBankrollEl, msg.bankroll.human, parseFloat(humanBankrollEl.textContent));
      bumpNumber(flyBankrollEl, msg.bankroll.fly, parseFloat(flyBankrollEl.textContent));
      humanSeatEl().classList.remove("active-turn");
      flySeatEl().classList.remove("active-turn");
      showBadge(msg.human_payoff >= 0 ? flySeatEl() : humanSeatEl(),
        msg.human_payoff === 0 ? "平手" : (msg.human_payoff > 0 ? `你 +${msg.human_payoff}` : `果蠅 +${-msg.human_payoff}`));
      setStatus("這手結束,準備下一手...");
      break;
    }
  }
}

function actionLabel(name) {
  return {
    FOLD: "蓋牌", CHECK_CALL: "過牌/跟注", RAISE_SMALL: "小加注",
    RAISE_BIG: "大加注", ALL_IN: "全下",
  }[name] || name;
}

document.querySelectorAll("#actions button").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.disabled) return;
    showBadge(humanSeatEl(), actionLabel(btn.dataset.action), btn.dataset.action === "FOLD");
    ws.send(JSON.stringify({ type: "action", action: btn.dataset.action }));
    setActionsEnabled(false);
    humanSeatEl().classList.remove("active-turn");
    startFlyThinking();
  });
});

setActionsEnabled(false);
connect();
