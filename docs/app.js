/* Blackjack Trainer — web app. Pure vanilla JS, no dependencies. */
(() => {
  'use strict';

  // ---------- Cards ----------
  // pip: blackjack value (Ace = 11). hiLo: Hi-Lo count tag.
  const RANKS = [
    { key: '2', label: '2', pip: 2, hiLo: +1 },
    { key: '3', label: '3', pip: 3, hiLo: +1 },
    { key: '4', label: '4', pip: 4, hiLo: +1 },
    { key: '5', label: '5', pip: 5, hiLo: +1 },
    { key: '6', label: '6', pip: 6, hiLo: +1 },
    { key: '7', label: '7', pip: 7, hiLo: 0 },
    { key: '8', label: '8', pip: 8, hiLo: 0 },
    { key: '9', label: '9', pip: 9, hiLo: 0 },
    { key: 'T', label: '10', pip: 10, hiLo: -1 },
    { key: 'J', label: 'J', pip: 10, hiLo: -1 },
    { key: 'Q', label: 'Q', pip: 10, hiLo: -1 },
    { key: 'K', label: 'K', pip: 10, hiLo: -1 },
    { key: 'A', label: 'A', pip: 11, hiLo: -1 },
  ];
  const SUITS = [
    { s: '♠', red: false }, { s: '♥', red: true },
    { s: '♦', red: true }, { s: '♣', red: false },
  ];

  const rankByPip = p => RANKS.find(r => r.pip === p && r.key !== 'J' && r.key !== 'Q' && r.key !== 'K');

  function card(rank, suit) { return { rank, suit }; }

  // ---------- Hand ----------
  function handTotal(cards) {
    let sum = cards.reduce((a, c) => a + c.rank.pip, 0);
    let aces = cards.filter(c => c.rank.key === 'A').length;
    while (sum > 21 && aces > 0) { sum -= 10; aces--; }
    return sum;
  }
  function handIsSoft(cards) {
    let sum = cards.reduce((a, c) => a + c.rank.pip, 0);
    let aces = cards.filter(c => c.rank.key === 'A').length;
    while (sum > 21 && aces > 0) { sum -= 10; aces--; }
    return aces > 0;
  }
  function handIsPair(cards) {
    return cards.length === 2 && cards[0].rank.pip === cards[1].rank.pip;
  }
  function handIsBlackjack(cards) { return cards.length === 2 && handTotal(cards) === 21; }
  function handDescribe(cards) {
    if (handIsBlackjack(cards)) return 'Blackjack!';
    if (handIsPair(cards)) return `Pair of ${cards[0].rank.label}s`;
    return (handIsSoft(cards) ? 'Soft ' : 'Hard ') + handTotal(cards);
  }

  // ---------- Strategy engine (4–8 deck charts) ----------
  // Codes: H, S, D(double else hit), Ds(double else stand), P(split),
  //        P*(split if DAS else hit), Rh(surr else hit), Rs(surr else stand)
  function hardMove(total, d, h17) {
    if (total <= 8) return 'H';
    if (total === 9) return (d >= 3 && d <= 6) ? 'D' : 'H';
    if (total === 10) return (d >= 2 && d <= 9) ? 'D' : 'H';
    if (total === 11) return h17 ? 'D' : (d === 11 ? 'H' : 'D');
    if (total === 12) return (d >= 4 && d <= 6) ? 'S' : 'H';
    if (total === 13 || total === 14) return (d >= 2 && d <= 6) ? 'S' : 'H';
    if (total === 15) {
      if (d >= 2 && d <= 6) return 'S';
      if (d === 10) return 'Rh';
      if (d === 11 && h17) return 'Rh';
      return 'H';
    }
    if (total === 16) {
      if (d >= 2 && d <= 6) return 'S';
      if (d >= 9) return 'Rh';   // 9, 10, A
      return 'H';                // 7, 8
    }
    return 'S'; // 17+
  }
  function softMove(total, d, h17) {
    if (total === 13 || total === 14) return (d >= 5 && d <= 6) ? 'D' : 'H';
    if (total === 15 || total === 16) return (d >= 4 && d <= 6) ? 'D' : 'H';
    if (total === 17) return (d >= 3 && d <= 6) ? 'D' : 'H';
    if (total === 18) {
      if (d >= 3 && d <= 6) return 'Ds';
      if (h17 && d === 2) return 'Ds';
      if (d === 2 || d === 7 || d === 8) return 'S';
      return 'H';               // 9, 10, A
    }
    if (total === 19) return (h17 && d === 6) ? 'Ds' : 'S';
    return 'S';                 // soft 20+
  }
  function pairMove(pip, isAce, d, das) {
    if (isAce) return 'P';
    if (pip === 10) return 'S';
    if (pip === 9) return (d === 7 || d === 10 || d === 11) ? 'S' : 'P';
    if (pip === 8) return 'P';
    if (pip === 7) return (d >= 2 && d <= 7) ? 'P' : 'H';
    if (pip === 6) return das ? (d >= 2 && d <= 6 ? 'P' : 'H') : (d >= 3 && d <= 6 ? 'P' : 'H');
    if (pip === 5) return (d >= 2 && d <= 9) ? 'D' : 'H';
    if (pip === 4) return (d >= 5 && d <= 6) ? 'P*' : 'H';
    return das ? (d >= 2 && d <= 7 ? 'P' : 'H') : (d >= 4 && d <= 7 ? 'P' : 'H'); // 2s, 3s
  }

  function bookMove(cards, dealerPip, rules) {
    const h17 = rules.h17, das = rules.das;
    if (handIsPair(cards)) {
      return pairMove(cards[0].rank.pip, cards[0].rank.key === 'A', dealerPip, das);
    }
    return handIsSoft(cards)
      ? softMove(handTotal(cards), dealerPip, h17)
      : hardMove(handTotal(cards), dealerPip, h17);
  }

  function resolve(code, { canDouble, canSplit, canSurrender, das }, fallbackAction) {
    switch (code) {
      case 'H': return 'hit';
      case 'S': return 'stand';
      case 'D': return canDouble ? 'double' : 'hit';
      case 'Ds': return canDouble ? 'double' : 'stand';
      case 'Rh': return canSurrender ? 'surrender' : 'hit';
      case 'Rs': return canSurrender ? 'surrender' : 'stand';
      case 'P': return canSplit ? 'split' : fallbackAction();
      case 'P*': return (canSplit && das) ? 'split' : 'hit';
    }
  }

  function recommendation(cards, dealerPip, rules) {
    const two = cards.length === 2;
    const canDouble = two;
    const canSurrender = rules.surrender && two;
    const canSplit = two && handIsPair(cards);
    const code = bookMove(cards, dealerPip, rules);
    const fallback = () => {
      const c = handIsSoft(cards)
        ? softMove(handTotal(cards), dealerPip, rules.h17)
        : hardMove(handTotal(cards), dealerPip, rules.h17);
      if (c === 'S' || c === 'Ds') return canDouble ? 'double' : 'stand';
      if (c === 'D') return canDouble ? 'double' : 'hit';
      return 'hit';
    };
    const action = resolve(code, { canDouble, canSplit, canSurrender, das: rules.das }, fallback);
    return { action, code, explanation: explain(code, cards, dealerPip) };
  }

  function dealerLabel(d) { return d === 11 ? 'an Ace' : `a ${d === 10 ? '10' : d}`; }

  function explain(code, cards, d) {
    const desc = handDescribe(cards).toLowerCase();
    const dl = dealerLabel(d);
    const weak = d >= 2 && d <= 6, strong = d >= 7;
    const total = handTotal(cards);
    switch (code) {
      case 'P':
        if (handIsPair(cards) && cards[0].rank.key === 'A')
          return 'Always split Aces — two fresh hands each starting with 11 beats a single soft/hard 12.';
        if (handIsPair(cards) && cards[0].rank.pip === 8)
          return 'Always split 8s. A hard 16 is the worst hand in blackjack; two hands starting on 8 are far stronger.';
        return `Split. Against ${dl} this pair wins more as two separate hands than as one total of ${total}.`;
      case 'P*':
        return 'Split only because you can double afterward (DAS). Without DAS, just hit.';
      case 'D':
        return `Double down. Strong starting hand vs ${dl}, so press your bet. (If you can't double — e.g. after hitting — just hit.)`;
      case 'Ds':
        return `Double for value against ${dl}. If doubling isn't allowed, stand.`;
      case 'Rh':
        return `Surrender. Against ${dl} your ${desc} loses too often — give up half the bet. If surrender isn't offered, hit.`;
      case 'Rs':
        return 'Surrender if allowed; otherwise stand.';
      case 'S':
        return weak
          ? `Stand. The dealer's ${dl} is a weak "bust" card — let them draw and risk busting instead of you.`
          : 'Stand. Hitting risks busting and your total is already competitive.';
      case 'H':
        return strong
          ? `Hit. The dealer's ${dl} is strong, so a weak total like ${total} must be improved.`
          : 'Hit. Your total is low enough that drawing is the better play.';
    }
  }

  // Visual info for a code (chart colors + action label)
  const CODE_INFO = {
    H:  { cls: 'm-hit',  label: 'Hit' },
    S:  { cls: 'm-stand', label: 'Stand' },
    D:  { cls: 'm-dbl',  label: 'Double' },
    Ds: { cls: 'm-dbl',  label: 'Double' },
    P:  { cls: 'm-split', label: 'Split' },
    'P*': { cls: 'm-split', label: 'Split' },
    Rh: { cls: 'm-surr', label: 'Surrender' },
    Rs: { cls: 'm-surr', label: 'Surrender' },
  };
  const ACTION_LABEL = { hit: 'Hit', stand: 'Stand', double: 'Double', split: 'Split', surrender: 'Surrender' };

  // ---------- Shoe ----------
  function buildShoe(decks) {
    const cards = [];
    for (let i = 0; i < decks; i++)
      for (const su of SUITS) for (const r of RANKS) cards.push(card(r, su));
    for (let i = cards.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [cards[i], cards[j]] = [cards[j], cards[i]];
    }
    return cards;
  }

  // ---------- Persistence ----------
  const LS = {
    get(k, def) { try { return JSON.parse(localStorage.getItem(k)) ?? def; } catch { return def; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  };
  const rules = Object.assign(
    { decks: 6, h17: false, das: true, surrender: true },
    LS.get('bj.rules', {})
  );
  function saveRules() { LS.set('bj.rules', rules); }

  const stats = Object.assign(
    { sC: 0, sT: 0, streak: 0, best: 0, cC: 0, cT: 0 },
    LS.get('bj.stats', {})
  );
  function saveStats() { LS.set('bj.stats', stats); }

  // ---------- Lessons ----------
  const LESSONS = [
    { t: 'The Goal & Hand Values', s: 'Beat the dealer without busting.', b:
`Get a total **closer to 21 than the dealer** without going over ("busting").

• Number cards (2–10) = face value.
• J, Q, K = 10 each.
• Ace = **11 or 1**, whichever helps.

An Ace counting as 11 makes a **soft** hand (A-6 = "soft 17"). If 11 would bust you, the Ace becomes 1 and the hand is **hard**.

A **blackjack** (Ace + 10-value on the first two cards) usually pays **3:2** — avoid 6:5 tables, the house edge is much worse.` },
    { t: 'Hitting & Standing', s: 'Take a card vs. keep your total.', b:
`**Hit** = take a card. **Stand** = keep your total.

The dealer must hit to 17+ then stand. The **dealer's upcard** drives your decisions:

• Dealer **2–6** (weak "bust cards"): they bust often, so stand on stiff totals (12–16) and let them take the risk.
• Dealer **7–Ace** (strong): improve weak hands — hit more.

Always assume the dealer's hidden card is a 10. That's why a 6 looks like a likely bust and a 10 looks like a made 20.` },
    { t: 'Doubling Down', s: 'Double your bet for one card.', b:
`**Double down**: double your bet, take **exactly one** card, then stand.

Double when strong vs a weak dealer:
• **Hard 11** — double vs almost everything.
• **Hard 10** — double vs 2–9.
• **Hard 9** — double vs 3–6.
• **Soft A-2…A-7** — double vs 4/5/6 (range depends on the hand).

You can only double on your **first two cards**. If a play says "double" after you've hit, the fallback is usually hit (or stand for soft 18/19).

Doubling is how you make money: more money in when you have the edge.` },
    { t: 'Splitting Pairs', s: 'Turn one pair into two hands.', b:
`A **pair** can be **split** into two hands, each getting a new card and an equal bet.

The rules you never break:
• **Always split Aces and 8s.** Aces → two hands starting at 11. 8s → escape the worst hand (16).
• **Never split 10s** (20 is great) **or 5s** (play 5-5 as a hard 10 and double).

Everything else depends on the dealer — split small/medium pairs mainly when the dealer is weak (2–6). **DAS** (double after split) adds a few splits like 4-4 vs 5/6.` },
    { t: 'Surrender', s: 'Give up half on hopeless hands.', b:
`**Late surrender** forfeits the hand for **half** your bet (not every table offers it).

Correct only on a few bad spots:
• **Hard 16** vs **9, 10, or Ace**.
• **Hard 15** vs **10** (and vs Ace if the dealer hits soft 17).

Logic: if you'll lose **more than half** the time, giving up half a bet wins long-term.

Note: 8-8 vs 10 is still a **split**, not a surrender.` },
    { t: 'Betting & Bankroll', s: 'Units, bet sizing, not going broke.', b:
`Think in **units** (one base bet), not dollars. Bring **40–100 units** so normal swings don't bust you.

• **Flat betting** (same bet) is safest with basic strategy — lowest variance.
• **Never** chase losses by doubling your bet after a loss ("Martingale"). Table limits + bad runs wipe you out fast.
• Set a **stop-loss** and a **win goal** before you sit, and walk when you hit either.

Basic strategy alone still leaves ~0.5% house edge. The only legal way to flip it is **bet spreading with the count**.` },
    { t: 'Card Counting (Hi-Lo)', s: 'Know when the deck favors you.', b:
`Counting isn't memorizing cards — it's one number telling you if the shoe is rich in **10s & Aces** (good for you) or small cards (good for the dealer).

**Hi-Lo tags:**
• **2,3,4,5,6** → **+1**
• **7,8,9** → **0**
• **10,J,Q,K,A** → **−1**

**Running count:** start at 0, add each card's tag. High positive = many small cards gone = more blackjacks and stronger doubles coming.

**True count = running count ÷ decks remaining.** +6 running with 6 decks left is only true +1; with 1 deck left it's a powerful +6. The true count is what you bet on.

**Use it:** minimum bet at true ≤ +1, raise your bet as the true count climbs. High count = your edge.

Drill the tags and running/true count in the **Count** tab.` },
  ];

  // ---------- Inline markdown (bold + bullets) ----------
  function mdInline(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }
  function mdBlock(text) {
    return text.split('\n\n').map(block => {
      const lines = block.split('\n');
      if (lines.every(l => l.trim().startsWith('•'))) {
        return '<ul>' + lines.map(l => `<li>${mdInline(l.replace(/^\s*•\s*/, ''))}</li>`).join('') + '</ul>';
      }
      return '<p>' + lines.map(mdInline).join('<br>') + '</p>';
    }).join('');
  }

  // ---------- DOM helpers ----------
  const $ = sel => document.querySelector(sel);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  function cardEl(c, faceDown) {
    if (faceDown) {
      const back = el('div', 'card down');
      back.innerHTML = '<span class="back">♣</span>';
      return back;
    }
    const d = el('div', 'card' + (c.suit.red ? ' red' : ''));
    d.innerHTML = `<span class="rk">${c.rank.label}</span><span class="su">${c.suit.s}</span>`;
    return d;
  }

  // ======================================================
  //  TAB RENDERERS
  // ======================================================
  const main = () => $('#main');

  function renderLearn() {
    const root = el('div', 'screen');
    root.appendChild(el('h1', null, 'Learn Blackjack'));
    LESSONS.forEach(les => {
      const item = el('button', 'row');
      item.innerHTML = `<div><div class="row-title">${les.t}</div><div class="row-sub">${les.s}</div></div><span class="chev">›</span>`;
      item.onclick = () => openLesson(les);
      root.appendChild(item);
    });
    swap(root);
  }
  function openLesson(les) {
    const root = el('div', 'screen');
    const back = el('button', 'back', '‹ Learn'); back.onclick = renderLearn;
    root.appendChild(back);
    root.appendChild(el('h1', null, les.t));
    const body = el('div', 'prose'); body.innerHTML = mdBlock(les.b);
    root.appendChild(body);
    swap(root);
  }

  // ---- Strategy chart ----
  const DEALERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
  function dealerCardForPip(p) {
    if (p === 11) return card(RANKS.find(r => r.key === 'A'), SUITS[0]);
    return card(rankByPip(p), SUITS[0]);
  }
  function hardHand(total) {
    if (total <= 11) return [card(rankByPip(total - 2), SUITS[0]), card(rankByPip(2), SUITS[1])];
    return [card(RANKS.find(r => r.key === 'T'), SUITS[0]), card(rankByPip(total - 10), SUITS[1])];
  }
  function renderStrategy() {
    const root = el('div', 'screen');
    root.appendChild(el('h1', null, 'Strategy Chart'));
    root.appendChild(el('p', 'hint', 'Tap any cell for the play and the reasoning. Rows = your hand, columns = dealer upcard. Current rules: <b>' + rulesSummary() + '</b>'));

    const sections = [
      { title: 'Hard Totals', rows: range(8, 17).map(t => ({ label: t === 17 ? '17+' : '' + t, cards: hardHand(t) })) },
      { title: 'Soft Totals', rows: range(13, 20).map(t => ({ label: 'A-' + (t - 11), cards: [card(RANKS.find(r => r.key === 'A'), SUITS[0]), card(rankByPip(t - 11), SUITS[1])] })) },
      { title: 'Pairs', rows: ['2','3','4','5','6','7','8','9','T','A'].map(k => { const r = RANKS.find(x => x.key === k); return { label: r.label + '-' + r.label, cards: [card(r, SUITS[0]), card(r, SUITS[3])] }; }) },
    ];
    sections.forEach(sec => {
      root.appendChild(el('h2', null, sec.title));
      const wrap = el('div', 'chart-wrap');
      const tbl = el('table', 'chart');
      const head = el('tr');
      head.appendChild(el('th', 'corner', ''));
      DEALERS.forEach(d => head.appendChild(el('th', null, d === 11 ? 'A' : d)));
      tbl.appendChild(head);
      sec.rows.forEach(row => {
        const tr = el('tr');
        tr.appendChild(el('th', 'rowlbl', row.label));
        DEALERS.forEach(d => {
          const code = bookMove(row.cards, d, rules);
          const td = el('td', CODE_INFO[code].cls, code);
          td.onclick = () => openDecision(row.cards, d);
          tr.appendChild(td);
        });
        tbl.appendChild(tr);
      });
      wrap.appendChild(tbl);
      root.appendChild(wrap);
    });
    root.appendChild(legend());
    swap(root);
  }
  function legend() {
    const box = el('div', 'legend');
    box.appendChild(el('h2', null, 'Legend'));
    [['H','Hit','m-hit'],['S','Stand','m-stand'],['D','Double','m-dbl'],['P','Split','m-split'],['Rh','Surrender','m-surr']]
      .forEach(([c, n, cls]) => {
        const r = el('div', 'leg-row');
        r.appendChild(el('span', 'swatch ' + cls, c));
        r.appendChild(el('span', null, ' ' + n));
        box.appendChild(r);
      });
    box.appendChild(el('p', 'hint', 'Ds = double else stand · P* = split only with DAS · Rh/Rs = surrender else hit/stand'));
    return box;
  }
  function openDecision(cards, d) {
    const rec = recommendation(cards, d, rules);
    const sheet = el('div', 'sheet-bg');
    const card2 = el('div', 'sheet');
    const you = el('div', 'duo');
    const yc = el('div', 'mini'); cards.forEach(c => yc.appendChild(cardEl(c)));
    you.appendChild(label('You', yc, handDescribe(cards)));
    const dc = el('div', 'mini');
    dc.appendChild(cardEl(dealerCardForPip(d))); dc.appendChild(cardEl(null, true));
    you.appendChild(label('Dealer', dc, 'Shows ' + (d === 11 ? 'A' : d)));
    card2.appendChild(you);
    const badge = el('div', 'big-action ' + CODE_INFO[rec.code].cls, ACTION_LABEL[rec.action].toUpperCase());
    card2.appendChild(badge);
    card2.appendChild(el('p', 'why', rec.explanation));
    const close = el('button', 'btn', 'Close'); close.onclick = () => sheet.remove();
    card2.appendChild(close);
    sheet.appendChild(card2);
    sheet.onclick = e => { if (e.target === sheet) sheet.remove(); };
    document.body.appendChild(sheet);
  }
  function label(top, node, bottom) {
    const w = el('div', 'lab');
    w.appendChild(el('div', 'lab-top', top));
    w.appendChild(node);
    w.appendChild(el('div', 'lab-bot', bottom));
    return w;
  }

  // ---- Trainer ----
  let tShoe = [], tPlayer = [], tDealer = null, tAnswered = false;
  function trainerDeal() {
    if (tShoe.length < 15) tShoe = buildShoe(rules.decks);
    do {
      tPlayer = [tShoe.pop(), tShoe.pop()];
      tDealer = tShoe.pop();
    } while (handIsBlackjack(tPlayer));
    tAnswered = false;
    renderTrainer();
  }
  function renderTrainer() {
    if (!tDealer) { tShoe = buildShoe(rules.decks); }
    if (!tPlayer.length) { trainerDeal(); return; }
    const root = el('div', 'screen');
    root.appendChild(el('h1', null, 'Trainer'));
    // scorebar
    const sb = el('div', 'scorebar');
    sb.appendChild(stat('Accuracy', stats.sT ? Math.round(stats.sC / stats.sT * 100) + '%' : '—'));
    sb.appendChild(stat('Streak', stats.streak));
    sb.appendChild(stat('Best', stats.best));
    sb.appendChild(stat('Hands', stats.sT));
    root.appendChild(sb);

    root.appendChild(el('div', 'sub', 'DEALER SHOWS'));
    const drow = el('div', 'hand');
    drow.appendChild(cardEl(tDealer)); drow.appendChild(cardEl(null, true));
    root.appendChild(drow);

    root.appendChild(el('div', 'sub', 'YOUR HAND'));
    const prow = el('div', 'hand');
    tPlayer.forEach(c => prow.appendChild(cardEl(c)));
    root.appendChild(prow);
    root.appendChild(el('div', 'desc', handDescribe(tPlayer)));

    const area = el('div', 'area');
    if (!tAnswered) {
      area.appendChild(el('div', 'sub', "What's the correct play?"));
      const grid = el('div', 'btn-grid');
      const acts = ['hit', 'stand'];
      if (tPlayer.length === 2) {
        acts.push('double');
        if (handIsPair(tPlayer)) acts.push('split');
        if (rules.surrender) acts.push('surrender');
      }
      acts.forEach(a => {
        const b = el('button', 'btn ghost', ACTION_LABEL[a]);
        b.onclick = () => answerTrainer(a);
        grid.appendChild(b);
      });
      area.appendChild(grid);
    }
    root.appendChild(area);
    swap(root);
    root._feedback = area; // for re-render target not needed
  }
  function answerTrainer(action) {
    const rec = recommendation(tPlayer, tDealer.rank.pip, rules);
    const correct = action === rec.action;
    stats.sT++;
    if (correct) { stats.sC++; stats.streak++; stats.best = Math.max(stats.best, stats.streak); }
    else stats.streak = 0;
    saveStats();
    tAnswered = true;
    renderTrainer();
    // append feedback
    const area = main().querySelector('.area');
    const fb = el('div', 'feedback');
    fb.appendChild(el('div', 'verdict ' + (correct ? 'ok' : 'no'),
      correct ? '✓ Correct!' : `✗ You chose ${ACTION_LABEL[action]} — correct is ${ACTION_LABEL[rec.action]}`));
    fb.appendChild(el('p', 'why', rec.explanation));
    const next = el('button', 'btn', 'Next Hand'); next.onclick = trainerDeal;
    fb.appendChild(next);
    area.appendChild(fb);
  }
  function stat(label, val) {
    const w = el('div', 'stat');
    w.appendChild(el('div', 'stat-v', val));
    w.appendChild(el('div', 'stat-l', label));
    return w;
  }

  // ---- Counting ----
  let countMode = 'tags';
  function renderCount() {
    const root = el('div', 'screen');
    root.appendChild(el('h1', null, 'Card Counting'));
    const seg = el('div', 'seg');
    ['tags', 'running'].forEach(m => {
      const b = el('button', 'seg-b' + (countMode === m ? ' on' : ''), m === 'tags' ? 'Card Tags' : 'Running Count');
      b.onclick = () => { countMode = m; renderCount(); };
      seg.appendChild(b);
    });
    root.appendChild(seg);
    root.appendChild(countMode === 'tags' ? tagDrill() : runningDrill());
    swap(root);
  }
  // tag drill
  let tagCard = null, tagAnswered = false;
  function tagDrill() {
    const box = el('div', 'center');
    if (!tagCard) nextTag(false);
    box.appendChild(el('p', 'hint', 'Hi-Lo: 2–6 = +1 · 7–9 = 0 · 10/J/Q/K/A = −1'));
    const big = cardEl(tagCard); big.classList.add('big');
    box.appendChild(big);
    const fb = el('div', 'tag-fb', tagAnswered === false ? 'Tag this card' :
      (tagAnswered.ok ? `Correct (${tagStr(tagCard.rank.hiLo)})` : `Nope — ${tagCard.rank.label} is ${tagStr(tagCard.rank.hiLo)}`));
    if (tagAnswered) fb.classList.add(tagAnswered.ok ? 'ok' : 'no');
    box.appendChild(fb);
    const g = el('div', 'tag-btns');
    [[+1, '+1', 'p'], [0, '0', 'z'], [-1, '−1', 'n']].forEach(([v, lab, cls]) => {
      const b = el('button', 'tagb ' + cls, lab);
      b.onclick = () => {
        if (tagAnswered) return;
        const ok = v === tagCard.rank.hiLo;
        stats.cT++; if (ok) stats.cC++; saveStats();
        tagAnswered = { ok };
        renderCount();
        setTimeout(() => { if (countMode === 'tags') nextTag(true); }, 700);
      };
      g.appendChild(b);
    });
    box.appendChild(g);
    box.appendChild(el('p', 'hint', `Accuracy ${stats.cT ? Math.round(stats.cC / stats.cT * 100) : 0}% · ${stats.cT} cards`));
    return box;
  }
  function nextTag(rerender) {
    tagCard = card(RANKS[Math.floor(Math.random() * RANKS.length)], SUITS[Math.floor(Math.random() * 4)]);
    tagAnswered = false;
    if (rerender) renderCount();
  }
  function tagStr(v) { return v > 0 ? '+1' : v < 0 ? '−1' : '0'; }

  // running drill
  let rcPhase = 'idle', rcSeq = [], rcIdx = 0, rcLen = 15, rcGuessR = 0, rcGuessT = 0;
  function rcSeen() { return rcSeq.slice(0, rcIdx).reduce((a, c) => a + c.rank.hiLo, 0); }
  function rcDecksRem() { return Math.max(0.5, (rules.decks * 52 - rcIdx) / 52); }
  function rcTrue() { return Math.round(rcSeen() / rcDecksRem()); }
  function runningDrill() {
    const box = el('div', 'center');
    if (rcPhase === 'idle') {
      box.appendChild(el('p', 'hint', 'Flip through cards one at a time, keep a running count in your head, then enter the running count and true count.'));
      const stepRow = el('div', 'stepper');
      const minus = el('button', 'btn small', '−'); const plus = el('button', 'btn small', '+');
      const lbl = el('span', 'step-lbl', 'Cards: ' + rcLen);
      minus.onclick = () => { rcLen = Math.max(5, rcLen - 5); lbl.textContent = 'Cards: ' + rcLen; };
      plus.onclick = () => { rcLen = Math.min(52, rcLen + 5); lbl.textContent = 'Cards: ' + rcLen; };
      stepRow.append(minus, lbl, plus);
      box.appendChild(stepRow);
      const start = el('button', 'btn', 'Start Round');
      start.onclick = () => { rcSeq = buildShoe(rules.decks).slice(0, rcLen); rcIdx = 0; rcGuessR = 0; rcGuessT = 0; rcPhase = 'dealing'; renderCount(); };
      box.appendChild(start);
    } else if (rcPhase === 'dealing') {
      box.appendChild(el('p', 'hint', `Card ${rcIdx} of ${rcSeq.length}`));
      if (rcIdx > 0) { const b = cardEl(rcSeq[rcIdx - 1]); b.classList.add('big'); box.appendChild(b); }
      else box.appendChild(el('div', 'placeholder', 'Tap Next'));
      const nb = el('button', 'btn', rcIdx < rcSeq.length ? 'Next Card' : 'Done — Enter Count');
      nb.onclick = () => { if (rcIdx < rcSeq.length) rcIdx++; else rcPhase = 'answering'; renderCount(); };
      box.appendChild(nb);
    } else if (rcPhase === 'answering') {
      box.appendChild(el('h2', null, rcSeq.length + ' cards shown'));
      box.appendChild(numStepper('Running count', () => rcGuessR, v => rcGuessR = v));
      box.appendChild(numStepper('True count', () => rcGuessT, v => rcGuessT = v));
      box.appendChild(el('p', 'hint', `(${rcDecksRem().toFixed(1)} decks remaining)`));
      const ck = el('button', 'btn', 'Check');
      ck.onclick = () => { const ok = rcGuessR === rcSeen(); stats.cT++; if (ok) stats.cC++; saveStats(); rcPhase = 'revealed'; renderCount(); };
      box.appendChild(ck);
    } else { // revealed
      box.appendChild(resultRow('Running count', rcGuessR, rcSeen()));
      box.appendChild(resultRow('True count', rcGuessT, rcTrue()));
      box.appendChild(el('p', 'hint', `True count = running (${rcSeen()}) ÷ decks remaining (${rcDecksRem().toFixed(1)}).`));
      const nr = el('button', 'btn', 'New Round'); nr.onclick = () => { rcPhase = 'idle'; renderCount(); };
      box.appendChild(nr);
    }
    return box;
  }
  function numStepper(label, get, set) {
    const row = el('div', 'stepper');
    const minus = el('button', 'btn small', '−'); const plus = el('button', 'btn small', '+');
    const lbl = el('span', 'step-lbl', `${label}: ${get()}`);
    minus.onclick = () => { set(get() - 1); lbl.textContent = `${label}: ${get()}`; };
    plus.onclick = () => { set(get() + 1); lbl.textContent = `${label}: ${get()}`; };
    row.append(minus, lbl, plus);
    return row;
  }
  function resultRow(label, guess, actual) {
    const ok = guess === actual;
    const r = el('div', 'res ' + (ok ? 'ok' : 'no'));
    r.innerHTML = `<span>${ok ? '✓' : '✗'} ${label}</span><span class="mono">you ${guess} · actual ${actual}</span>`;
    return r;
  }

  // ---- Settings ----
  function renderSettings() {
    const root = el('div', 'screen');
    root.appendChild(el('h1', null, 'Settings'));
    root.appendChild(el('h2', null, 'Table Rules'));
    const decks = numStepper('Decks', () => rules.decks, v => { rules.decks = Math.min(8, Math.max(1, v)); saveRules(); tShoe = []; });
    root.appendChild(decks);
    root.appendChild(toggle('Dealer hits soft 17 (H17)', 'h17'));
    root.appendChild(toggle('Double after split (DAS)', 'das'));
    root.appendChild(toggle('Late surrender offered', 'surrender'));
    root.appendChild(el('p', 'hint', 'These change the correct basic-strategy plays. Defaults (6 decks, stand soft 17, DAS, late surrender) match a typical modern shoe game.'));

    root.appendChild(el('h2', null, 'Your Progress'));
    root.appendChild(kv('Strategy accuracy', `${stats.sT ? Math.round(stats.sC / stats.sT * 100) : 0}% (${stats.sT} hands)`));
    root.appendChild(kv('Best streak', stats.best));
    root.appendChild(kv('Counting accuracy', `${stats.cT ? Math.round(stats.cC / stats.cT * 100) : 0}% (${stats.cT} cards)`));
    const rs = el('button', 'btn danger', 'Reset strategy stats');
    rs.onclick = () => { stats.sC = stats.sT = stats.streak = stats.best = 0; saveStats(); renderSettings(); };
    const rc = el('button', 'btn danger', 'Reset counting stats');
    rc.onclick = () => { stats.cC = stats.cT = 0; saveStats(); renderSettings(); };
    root.append(rs, rc);
    root.appendChild(el('p', 'hint foot', 'For learning only. Card counting is legal, but casinos may ask advantage players to leave. Gamble responsibly.'));
    swap(root);
  }
  function toggle(label, key) {
    const row = el('label', 'toggle');
    row.appendChild(el('span', null, label));
    const sw = el('span', 'switch' + (rules[key] ? ' on' : ''));
    sw.appendChild(el('span', 'knob'));
    row.appendChild(sw);
    row.onclick = () => { rules[key] = !rules[key]; saveRules(); tShoe = []; renderSettings(); };
    return row;
  }
  function kv(k, v) { const r = el('div', 'kv'); r.innerHTML = `<span>${k}</span><span class="mono">${v}</span>`; return r; }

  // ---------- utils ----------
  function range(a, b) { const out = []; for (let i = a; i <= b; i++) out.push(i); return out; }
  function rulesSummary() {
    const p = [rules.decks + 'D', rules.h17 ? 'H17' : 'S17'];
    if (rules.das) p.push('DAS'); if (rules.surrender) p.push('LS');
    return p.join(' · ');
  }
  function swap(node) { const m = main(); m.innerHTML = ''; m.appendChild(node); m.scrollTop = 0; }

  // ---------- Tab bar ----------
  const TABS = [
    { id: 'learn', label: 'Learn', icon: '📘', render: renderLearn },
    { id: 'strategy', label: 'Strategy', icon: '🃏', render: renderStrategy },
    { id: 'trainer', label: 'Trainer', icon: '🎯', render: renderTrainer },
    { id: 'count', label: 'Count', icon: '🔢', render: renderCount },
    { id: 'settings', label: 'Settings', icon: '⚙️', render: renderSettings },
  ];
  function buildTabBar() {
    const bar = $('#tabbar');
    TABS.forEach(t => {
      const b = el('button', 'tab', `<span class="ti">${t.icon}</span><span class="tl">${t.label}</span>`);
      b.onclick = () => selectTab(t.id);
      b.dataset.id = t.id;
      bar.appendChild(b);
    });
  }
  function selectTab(id) {
    document.querySelectorAll('#tabbar .tab').forEach(b => b.classList.toggle('on', b.dataset.id === id));
    TABS.find(t => t.id === id).render();
  }

  // ---------- boot ----------
  buildTabBar();
  selectTab('strategy');
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }
})();
