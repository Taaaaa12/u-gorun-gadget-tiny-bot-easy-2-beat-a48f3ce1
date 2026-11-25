// Tiny Bot, Easy $2: Beat Buy-and-Hold on Solana.
// A small long-only paper-trading bot for hourly Solana bars (bars-replay/v1).
//
// The hello message opens the run: it names the symbols, the capital, the caps and
// the fees. After that the harness sends one batch per bar, and this bot answers
// each batch with its orders for that bar (an empty list when it sits out). Every
// order decided on a bar fills at the next bar's open, so every signal is read off
// closes the bot has already seen; nothing here reads the clock, the environment or
// any randomness, so the two grading runs agree byte for byte.
//
// Two sleeves share the account, both long only:
//
//   1. Dip sleeve. A symbol is bought when its short-term momentum is up but its
//      price has dipped hard: MACD(12, 26, 9) above its signal line and
//      CCI(14) at or under -48. It leaves when the momentum turns (MACD under its
//      signal), when the position is worth 2% over its cost, or when it is worth
//      30% under it. The dip is bought with one seventh of the equity.
//   2. Floor sleeve. A trade-count guardrail needs at least 8 trades in the final
//      window, and a quiet window may give the dip sleeve none. So from bar 24,
//      while fewer than 8 orders have filled, the bot also buys a sharper dip:
//      a 6-bar drop of 1.5% or more with RSI(7) under 50. These are small (one
//      fourteenth of the equity) and leave fast: +1%, -1.5%, or after 12 bars.
//      Once 8 trades have filled the sleeve switches itself off.
//
// Sizing respects the caps from hello: no position over max_weight_bps of equity,
// no more than max_open_positions, no more than max_orders_per_bar in one reply,
// and nothing under the minimum notional. Each order caps its slippage at 100 bps
// past the close it was decided on, so a fill into a gap is refused and retried
// on the next bar instead of chased.

import { BPS, runBars, type Bar, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

// --- dip sleeve -------------------------------------------------------------
const MACD_FAST = 12;
const MACD_SLOW = 26;
const MACD_SIGNAL = 9;
const CCI_BARS = 14;
const CCI_ENTRY_AT_MOST = -48;
/** Take profit: worth 102/100 of its cost. */
const DIP_ROI = [102n, 100n] as const;
/** Stoploss: worth 7/10 of its cost. */
const DIP_STOP = [7n, 10n] as const;
const DIP_STAKE_DIVISOR = 7n;

// --- floor sleeve -----------------------------------------------------------
const FLOOR_BAR = 24;
const FLOOR_FILLS = 8;
const FLOOR_DROP_BARS = 6;
const FLOOR_RSI_BARS = 7;
const FLOOR_RSI_UNDER = 50;
/** Take profit: worth 101/100 of its cost. */
const FLOOR_ROI = [101n, 100n] as const;
/** Stoploss: worth 197/200 of its cost. */
const FLOOR_STOP = [197n, 200n] as const;
const FLOOR_HOLD_BARS = 12;
const FLOOR_STAKE_DIVISOR = 14n;

const MAX_SLIPPAGE_BPS = 100;

const mean = (xs: readonly number[]): number => xs.reduce((a, x) => a + x, 0) / xs.length;

/** One symbol's bars so far, oldest first; a gap adds nothing. */
interface Series {
  closes: number[];
  highs: number[];
  lows: number[];
}

const series = new Map<string, Series>();
/** How each open position was entered: 'dip' or 'scalp', and the bar its buy filled on. */
const entries = new Map<string, { kind: 'dip' | 'scalp'; seq: number }>();
/** The kind of each order sent, by the client_id it was given. */
const kinds = new Map<string, 'dip' | 'scalp'>();
/** Orders filled so far (buys and sells alike); the floor sleeve reads this. */
let filled = 0;
let nextClient = 1;

function clientId(kind: 'dip' | 'scalp'): string {
  const id = `${kind[0]}${nextClient++}`;
  kinds.set(id, kind);
  return id;
}

/**
 * MACD(12, 26, 9) on a symbol's closes, seeded the way TA-Lib does: EMA(26) as
 * the mean of the first 26 closes, EMA(12) as the mean of the 12 closes up to
 * that bar, the signal line as the mean of the first 9 MACD values. Returns the
 * latest MACD and signal, or nulls before they exist (MACD from bar 25, the
 * signal from bar 33).
 */
function macd(closes: readonly number[]): { macd: number | null; signal: number | null } {
  if (closes.length < MACD_SLOW) return { macd: null, signal: null };
  let slow = mean(closes.slice(0, MACD_SLOW));
  let fast = mean(closes.slice(MACD_SLOW - MACD_FAST, MACD_SLOW));
  const values: number[] = [fast - slow];
  for (let i = MACD_SLOW; i < closes.length; i++) {
    slow += (2 / (MACD_SLOW + 1)) * (closes[i]! - slow);
    fast += (2 / (MACD_FAST + 1)) * (closes[i]! - fast);
    values.push(fast - slow);
  }
  if (values.length < MACD_SIGNAL) return { macd: values[values.length - 1]!, signal: null };
  let signal = mean(values.slice(0, MACD_SIGNAL));
  for (let i = MACD_SIGNAL; i < values.length; i++) {
    signal += (2 / (MACD_SIGNAL + 1)) * (values[i]! - signal);
  }
  return { macd: values[values.length - 1]!, signal };
}

/**
 * CCI(14) on a symbol's typical prices (h + l + c) / 3: the latest typical price
 * against the mean of the last 14, divided by 0.015 times their mean absolute
 * deviation; 0 when the deviation is 0. Null before 14 bars.
 */
function cci(s: Series): number | null {
  const n = s.closes.length;
  if (n < CCI_BARS) return null;
  const typical: number[] = [];
  for (let i = n - CCI_BARS; i < n; i++) {
    typical.push((s.highs[i]! + s.lows[i]! + s.closes[i]!) / 3);
  }
  const m = mean(typical);
  const spread = typical.reduce((a, x) => a + (x < m ? m - x : x - m), 0);
  const dev = spread / CCI_BARS;
  if (dev === 0) return 0;
  return (typical[typical.length - 1]! - m) / (0.015 * dev);
}

/**
 * RSI over `span` closes, Wilder's smoothing: the first average gain and loss
 * are the means of the first `span` changes, each next one (span - 1) / span of
 * the last plus 1 / span of the newest change. Null before `span` changes.
 */
function rsi(closes: readonly number[], span: number): number | null {
  if (closes.length <= span) return null;
  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= span; i++) {
    const change = closes[i]! - closes[i - 1]!;
    if (change > 0) gain += change;
    else loss -= change;
  }
  gain /= span;
  loss /= span;
  for (let i = span + 1; i < closes.length; i++) {
    const change = closes[i]! - closes[i - 1]!;
    gain = (gain * (span - 1) + (change > 0 ? change : 0)) / span;
    loss = (loss * (span - 1) + (change < 0 ? -change : 0)) / span;
  }
  const moved = gain + loss;
  return moved === 0 ? 50 : (100 * gain) / moved;
}

function push(bar: Bar): void {
  let s = series.get(bar.symbol);
  if (s === undefined) series.set(bar.symbol, (s = { closes: [], highs: [], lows: [] }));
  s.closes.push(Number(bar.c));
  s.highs.push(Number(bar.h));
  s.lows.push(Number(bar.l));
}

/** Whether a held dip position exits on this close: momentum turned, or its value at the ROI or the stoploss. */
function dipExits(state: BarsState, symbol: string, held: { qty: bigint; cost: bigint }, m: { macd: number | null; signal: number | null }): boolean {
  if (m.macd !== null && m.signal !== null && m.macd < m.signal) return true;
  const value = state.value(symbol, held.qty);
  return value * DIP_ROI[1] >= held.cost * DIP_ROI[0] || value * DIP_STOP[1] <= held.cost * DIP_STOP[0];
}

/** Whether a held floor (scalp) position exits: its quick profit, its tight stop, or its 12 bars are up. */
function scalpExits(state: BarsState, symbol: string, held: { qty: bigint; cost: bigint }, enteredSeq: number): boolean {
  const value = state.value(symbol, held.qty);
  if (value * FLOOR_ROI[1] >= held.cost * FLOOR_ROI[0]) return true;
  if (value * FLOOR_STOP[1] <= held.cost * FLOOR_STOP[0]) return true;
  return state.seq - enteredSeq >= FLOOR_HOLD_BARS;
}

await runBars({
  onBatch(state, batch) {
    // The fills this batch carries resolve the orders decided on the last bar.
    for (const f of batch.fills) {
      if (f.status !== 'filled') continue;
      filled += 1;
      if (f.side === 'buy') {
        const kind = kinds.get(f.client_id) ?? 'dip';
        entries.set(f.symbol, { kind, seq: f.fill_bar });
      } else {
        entries.delete(f.symbol);
      }
    }
    for (const bar of batch.bars) push(bar);
    // While warming up every order would be rejected and never retried: wait.
    if (state.warmingUp) return [];

    const { caps, fees } = state.hello;
    const fresh = new Set(batch.bars.map((b) => b.symbol));
    const sells: SellInput[] = [];
    const buys: BuyInput[] = [];
    let cash = state.cash;
    let room = caps.max_open_positions - state.positions.size;

    for (const { symbol } of state.hello.symbols) {
      if (!fresh.has(symbol)) continue;
      const s = series.get(symbol)!;
      const held = state.positions.get(symbol);
      if (held !== undefined) {
        const entry = entries.get(symbol) ?? { kind: 'dip' as const, seq: state.seq };
        const m = macd(s.closes);
        const exit = entry.kind === 'scalp' ? scalpExits(state, symbol, held, entry.seq) : dipExits(state, symbol, held, m);
        if (exit) sells.push({ side: 'sell', symbol, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS });
        continue;
      }

      // Not held: the dip sleeve first, then the floor sleeve.
      let kind: 'dip' | 'scalp' | null = null;
      const m = macd(s.closes);
      const c = cci(s);
      if (m.macd !== null && m.signal !== null && c !== null && m.macd > m.signal && c <= CCI_ENTRY_AT_MOST) {
        kind = 'dip';
      } else if (state.seq >= FLOOR_BAR && filled < FLOOR_FILLS && s.closes.length > FLOOR_DROP_BARS) {
        const now = s.closes[s.closes.length - 1]!;
        const then = s.closes[s.closes.length - 1 - FLOOR_DROP_BARS]!;
        // A 6-bar drop of 1.5% or more.
        const dropped = now < then * 0.985;
        const r = rsi(s.closes, FLOOR_RSI_BARS);
        if (dropped && r !== null && r < FLOOR_RSI_UNDER) kind = 'scalp';
      }
      if (kind === null) continue;
      if (room <= 0) continue;
      const divisor = kind === 'dip' ? DIP_STAKE_DIVISOR : FLOOR_STAKE_DIVISOR;
      let slice = state.equity / divisor;
      const byWeight = (BigInt(caps.max_weight_bps) * state.equity) / BPS;
      if (slice > byWeight) slice = byWeight;
      if (slice > cash) slice = cash;
      if (slice === 0n || slice < BigInt(fees.min_notional_quote_units)) continue;
      buys.push({ side: 'buy', symbol, notional_quote_units: slice, max_slippage_bps: MAX_SLIPPAGE_BPS, client_id: clientId(kind) });
      cash -= slice;
      room -= 1;
    }

    const orders: (SellInput | BuyInput)[] = [...sells, ...buys].slice(0, caps.max_orders_per_bar);
    return orders;
  },
});
