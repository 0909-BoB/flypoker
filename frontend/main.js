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

let ws = null;
let humanSeat = 0;
let currentLegalActions = [];

const el = (id) => document.getElementById(id);

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

function log(text, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  const panel = el("log");
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
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

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => { el("status").textContent = "已連線,等待發牌..."; };
  ws.onclose = () => { el("status").textContent = "連線中斷"; setActionsEnabled(false); };
  ws.onerror = () => { el("status").textContent = "連線錯誤"; };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleMessage(msg);
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case "new_hand": {
      humanSeat = msg.human_seat;
      el("hand-number").textContent = msg.hand_number;
      renderCards("human-cards", msg.human_hole, false);
      renderCards("fly-cards", ["??", "??"], true);
      renderCards("board-cards", []);
      el("pot-amount").textContent = msg.small_blind + msg.big_blind;
      el("human-stack").textContent = "2000";
      el("fly-stack").textContent = "2000";
      el("street-label").textContent = "preflop";
      el("fly-decision").textContent = "";
      log(`── 第 ${msg.hand_number} 手 (你的座位 ${msg.human_seat === 0 ? "按鈕/小盲" : "大盲"}) ──`, "hand-sep");
      setActionsEnabled(false);
      break;
    }
    case "your_turn": {
      const obs = msg.obs;
      el("street-label").textContent = obs.street;
      renderCards("board-cards", obs.board);
      el("pot-amount").textContent = obs.pot;
      el("human-stack").textContent = obs.my_stack;
      el("fly-stack").textContent = obs.opp_stack;
      currentLegalActions = obs.legal_actions;
      setActionsEnabled(true, obs.legal_actions);
      el("status").textContent = obs.to_call > 0
        ? `輪到你:需跟注 ${obs.to_call}`
        : "輪到你:可過牌或下注";
      break;
    }
    case "fly_decision": {
      setActionsEnabled(false);
      updateBrainGrid(msg.activity);
      el("fly-decision").innerHTML =
        `[${msg.street}] 果蠅 <b>${actionLabel(msg.action)}</b>` +
        ` (勝率估計 ${(msg.equity * 100).toFixed(0)}%)`;
      log(`  果蠅[${msg.street}] ${actionLabel(msg.action)} (equity=${msg.equity.toFixed(2)})`);
      break;
    }
    case "hand_result": {
      setActionsEnabled(false);
      msg.log.forEach((line) => log("  " + line));
      if (msg.showdown && msg.fly_hole) {
        renderCards("fly-cards", msg.fly_hole, false);
      }
      const cls = msg.human_payoff > 0 ? "win" : msg.human_payoff < 0 ? "lose" : "";
      log(`  你這手 ${msg.human_payoff > 0 ? "+" : ""}${msg.human_payoff}`, cls);
      el("human-bankroll").textContent = msg.bankroll.human;
      el("fly-bankroll").textContent = msg.bankroll.fly;
      el("status").textContent = "這手結束,準備下一手...";
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
    ws.send(JSON.stringify({ type: "action", action: btn.dataset.action }));
    setActionsEnabled(false);
    el("status").textContent = "已送出動作,等待結果...";
  });
});

setActionsEnabled(false);
connect();
