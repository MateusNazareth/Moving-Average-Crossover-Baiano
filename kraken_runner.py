"""
=============================================================================
Kraken Integration — BTC/SOL Scalping Strategy
=============================================================================
Supports two execution modes selectable at runtime:
  • PAPER  — simulates orders locally, zero risk, real market data
  • LIVE   — places real orders on Kraken via the REST API

Pairs supported: BTC/USD, BTC/USDT, SOL/USD, SOL/USDT

Requirements:
    pip install ccxt pandas numpy python-dotenv

Setup:
    1. Create a .env file in the same directory as this script:
           KRAKEN_API_KEY=your_api_key_here
           KRAKEN_API_SECRET=your_api_secret_here
           TRADING_MODE=paper          # or "live"
           PAIRS=BTC/USD,SOL/USD       # comma-separated pairs to trade

    2. Kraken API key permissions needed (in Kraken dashboard):
           ✅ Query Funds
           ✅ Query Open Orders & Trades
           ✅ Create & Modify Orders
           ✅ Cancel/Close Orders
           ❌ Withdraw Funds  (keep this OFF for safety)

    3. Run:
           python kraken_runner.py

WARNING: Live trading involves real financial risk.
         Always test thoroughly in paper mode first.
=============================================================================
"""

import os
import time
import logging
import json
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ── Import our strategy engine ────────────────────────────────────────────────
# The strategy file must be in the same directory.
from btc_sol_scalping_strategy import ScalpingStrategy, CONFIG


# =============================================================================
# LOGGING SETUP
# Logs to both console and a timestamped file so you have a full audit trail.
# =============================================================================
log_filename = f"kraken_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                      # print to terminal
        logging.FileHandler(log_filename, mode="a"),  # save to file
    ],
)
log = logging.getLogger("KrakenBot")


# =============================================================================
# ENVIRONMENT & CONFIGURATION
# =============================================================================
load_dotenv()  # reads .env file into os.environ

# ── Kraken credentials ────────────────────────────────────────────────────────
KRAKEN_API_KEY    = os.getenv("KRAKEN_API_KEY", "")
KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

# ── Runtime settings ──────────────────────────────────────────────────────────
# TRADING_MODE: "paper" for simulation, "live" for real orders
TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()

# PAIRS: comma-separated list of markets to trade simultaneously
RAW_PAIRS    = os.getenv("PAIRS", "BTC/USD,BTC/USDT,SOL/USD,SOL/USDT")
PAIRS        = [p.strip() for p in RAW_PAIRS.split(",")]

# How many 1-minute candles to fetch per cycle (must be ≥ 50 for indicators to warm up)
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", "200"))

# Seconds to wait between each full scan cycle (60 = once per minute)
LOOP_INTERVAL = int(os.getenv("LOOP_INTERVAL", "60"))

# Starting virtual equity for paper trading (USD)
PAPER_STARTING_EQUITY = float(os.getenv("PAPER_EQUITY", "10000"))

# Strategy config (can be overridden per-pair below if desired)
STRATEGY_CONFIG = {
    **CONFIG,                    # inherit defaults from strategy file
    "commission": 0.026,         # Kraken maker fee = 0.16%, taker = 0.26% — use taker as conservative estimate
}


# =============================================================================
# PAPER TRADING LEDGER
# Tracks virtual positions and equity without touching real funds.
# =============================================================================
class PaperLedger:
    """
    Simulates account state for paper trading.

    Tracks:
      - equity       : current virtual USD balance
      - positions    : dict of { symbol: { direction, qty, entry, sl, tp, trail_* } }
      - trade_log    : list of completed trade dicts
    """

    def __init__(self, starting_equity: float):
        self.equity     = starting_equity
        self.positions  = {}   # keyed by symbol (e.g. "BTC/USD")
        self.trade_log  = []

    def open_position(self, symbol: str, direction: str, price: float,
                      qty: float, sl: float, tp: float,
                      trail_points: float, trail_offset: float):
        """Record a new simulated entry."""
        commission_cost = price * qty * (STRATEGY_CONFIG["commission"] / 100)
        self.equity -= commission_cost   # deduct entry commission from equity

        self.positions[symbol] = {
            "direction":       direction,
            "entry":           price,
            "qty":             qty,
            "sl":              sl,
            "tp":              tp,
            "trail_points":    trail_points,
            "trail_offset":    trail_offset,
            "trail_activated": False,
            "opened_at":       datetime.now(timezone.utc).isoformat(),
        }
        log.info(f"[PAPER] OPEN {direction.upper()} {symbol} @ {price:.4f} "
                 f"| SL={sl:.4f}  TP={tp:.4f}  Qty={qty:.6f}")

    def update_and_check_exit(self, symbol: str, high: float, low: float) -> Optional[dict]:
        """
        Called each candle. Updates trailing stop and checks if SL/TP was hit.
        Returns trade result dict if closed, else None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        sl, tp = pos["sl"], pos["tp"]
        direction = pos["direction"]

        # ── Trailing stop update ───────────────────────────────────────────────
        if STRATEGY_CONFIG["use_trailing"]:
            trail_pts = pos["trail_points"]
            trail_off = pos["trail_offset"]
            if direction == "long":
                profit_pts = high - pos["entry"]
                if profit_pts >= trail_pts:
                    new_sl = high - trail_off
                    sl = max(sl, new_sl)   # trail only moves UP for longs
                    pos["trail_activated"] = True
            else:
                profit_pts = pos["entry"] - low
                if profit_pts >= trail_pts:
                    new_sl = low + trail_off
                    sl = min(sl, new_sl)   # trail only moves DOWN for shorts
                    pos["trail_activated"] = True
            pos["sl"] = sl

        # ── Exit condition check ───────────────────────────────────────────────
        tp_hit   = (direction == "long"  and high >= tp) or \
                   (direction == "short" and low  <= tp)
        stop_hit = (direction == "long"  and low  <= sl) or \
                   (direction == "short" and high >= sl)

        if not (tp_hit or stop_hit):
            return None

        # Determine exit price (TP takes priority if both hit same bar)
        exit_price = tp if tp_hit else sl
        qty        = pos["qty"]

        # Calculate PnL
        if direction == "long":
            gross_pnl = (exit_price - pos["entry"]) * qty
        else:
            gross_pnl = (pos["entry"] - exit_price) * qty

        commission_cost = exit_price * qty * (STRATEGY_CONFIG["commission"] / 100)
        net_pnl = gross_pnl - commission_cost

        self.equity += net_pnl   # update virtual account balance

        result = {
            "symbol":      symbol,
            "direction":   direction,
            "entry":       pos["entry"],
            "exit":        exit_price,
            "qty":         qty,
            "result":      "TP" if tp_hit else "SL",
            "net_pnl_usd": round(net_pnl, 4),
            "equity":      round(self.equity, 2),
            "opened_at":   pos["opened_at"],
            "closed_at":   datetime.now(timezone.utc).isoformat(),
        }

        self.trade_log.append(result)
        del self.positions[symbol]   # mark as closed

        log.info(f"[PAPER] CLOSE {direction.upper()} {symbol} @ {exit_price:.4f} "
                 f"| Result={result['result']}  PnL={net_pnl:+.2f} USD  "
                 f"Equity={self.equity:.2f}")
        return result

    def summary(self):
        """Print paper trading performance summary."""
        if not self.trade_log:
            log.info("[PAPER] No completed trades yet.")
            return
        df    = pd.DataFrame(self.trade_log)
        wins  = df[df["net_pnl_usd"] > 0]
        loss  = df[df["net_pnl_usd"] <= 0]
        log.info("=" * 60)
        log.info("  PAPER TRADING SUMMARY")
        log.info("=" * 60)
        log.info(f"  Trades      : {len(df)}  (W:{len(wins)} / L:{len(loss)})")
        log.info(f"  Win rate    : {len(wins)/len(df)*100:.1f}%")
        log.info(f"  Total PnL   : ${df['net_pnl_usd'].sum():.2f}")
        log.info(f"  Equity now  : ${self.equity:.2f}")
        log.info("=" * 60)


# =============================================================================
# KRAKEN BOT — MAIN CLASS
# =============================================================================
class KrakenBot:
    """
    Orchestrates the full live-trading / paper-trading loop.

    For each configured pair every LOOP_INTERVAL seconds:
      1. Fetch the latest 1-minute OHLCV candles from Kraken
      2. Run the ScalpingStrategy to generate signals
      3. Execute or simulate orders based on TRADING_MODE
      4. Monitor and close open positions when SL/TP is hit
    """

    def __init__(self):
        # ── Connect to Kraken via ccxt ─────────────────────────────────────────
        self.exchange = ccxt.kraken({
            "apiKey":  KRAKEN_API_KEY,
            "secret":  KRAKEN_API_SECRET,
            "enableRateLimit": True,   # respect Kraken's rate limits automatically
            "options": {
                "defaultType": "spot",  # use "future" for Kraken Futures (separate account)
            },
        })

        # ── Mode banner ────────────────────────────────────────────────────────
        if TRADING_MODE == "live":
            log.warning("⚠️  LIVE MODE — real orders will be placed on Kraken!")
        else:
            log.info("📄  PAPER MODE — simulating trades, no real orders.")

        # ── Paper ledger (only used in paper mode) ─────────────────────────────
        self.paper = PaperLedger(PAPER_STARTING_EQUITY)

        # ── Live position tracker: { symbol: { direction, qty, sl, tp, ... } }
        # Used in live mode to track what we've opened so we can manage exits.
        self.live_positions: dict = {}

        log.info(f"Trading pairs : {PAIRS}")
        log.info(f"Candle limit  : {CANDLE_LIMIT}")
        log.info(f"Loop interval : {LOOP_INTERVAL}s")

    # -------------------------------------------------------------------------
    # DATA FETCHING
    # -------------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the most recent `CANDLE_LIMIT` 1-minute candles from Kraken.
        Returns a DataFrame with columns: open, high, low, close, volume.
        The last (incomplete) candle is dropped to avoid acting on partial data.
        """
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe="1m", limit=CANDLE_LIMIT + 1)
        except ccxt.NetworkError as e:
            log.error(f"Network error fetching {symbol}: {e}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error fetching {symbol}: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Drop the last row — it's the current, still-forming candle
        df = df.iloc[:-1]

        log.debug(f"Fetched {len(df)} candles for {symbol}")
        return df

    # -------------------------------------------------------------------------
    # SIGNAL GENERATION
    # -------------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame) -> dict:
        """
        Run the ScalpingStrategy on the OHLCV data and return the signal
        from the LAST completed candle.

        Returns a dict with keys:
          long_signal, short_signal, close, long_sl, long_tp,
          short_sl, short_tp, trail_points, trail_offset
        """
        if df.empty or len(df) < 50:
            # Need at least 50 bars for indicators to warm up
            return {"long_signal": False, "short_signal": False}

        strat = ScalpingStrategy(df, STRATEGY_CONFIG)
        # Run only indicator + signal computation (no backtest needed here)
        strat._compute_indicators()
        strat._compute_structure_breaks()
        strat._compute_order_blocks()
        strat._compute_fvg()
        strat._compute_entry_signals()
        strat._compute_exit_levels()

        last = strat.df.iloc[-1]   # most recent completed candle

        return {
            "long_signal":  bool(last["long_signal"]),
            "short_signal": bool(last["short_signal"]),
            "close":        float(last["close"]),
            "long_sl":      float(last["long_sl"]),
            "long_tp":      float(last["long_tp"]),
            "short_sl":     float(last["short_sl"]),
            "short_tp":     float(last["short_tp"]),
            "trail_points": float(last["trail_points"]) if STRATEGY_CONFIG["use_trailing"] else float("nan"),
            "trail_offset": float(last["trail_offset"]) if STRATEGY_CONFIG["use_trailing"] else float("nan"),
            "atr":          float(last["atr"]),
        }

    # -------------------------------------------------------------------------
    # POSITION SIZING
    # -------------------------------------------------------------------------
    def compute_qty(self, symbol: str, price: float, sl: float) -> float:
        """
        Risk-based position sizing.
        Risks `risk_percent` % of available equity per trade.

        Formula:
            risk_usd   = equity × risk_percent / 100
            stop_dist  = |price - sl|
            qty        = risk_usd / stop_dist

        This ensures a consistent dollar loss regardless of which pair is traded.
        """
        risk_pct = STRATEGY_CONFIG["risk_percent"] / 100

        if TRADING_MODE == "paper":
            equity = self.paper.equity
        else:
            equity = self._get_live_equity()

        risk_usd  = equity * risk_pct
        stop_dist = abs(price - sl)

        if stop_dist == 0:
            log.warning(f"Stop distance is 0 for {symbol} — skipping")
            return 0.0

        qty = risk_usd / stop_dist

        # ── Kraken minimum order sizes ─────────────────────────────────────────
        # These are the current minimums; check Kraken docs if changed.
        min_qty = {"BTC": 0.0001, "SOL": 0.5}.get(symbol.split("/")[0], 0.001)
        qty = max(qty, min_qty)

        return round(qty, 6)

    def _get_live_equity(self) -> float:
        """Fetch available USD/USDT balance from Kraken account."""
        try:
            balance = self.exchange.fetch_balance()
            # Kraken returns balances under currency codes
            usd = balance.get("USD", {}).get("free", 0) or 0
            usdt = balance.get("USDT", {}).get("free", 0) or 0
            return float(usd) + float(usdt)
        except Exception as e:
            log.error(f"Could not fetch live equity: {e}")
            return 0.0

    # -------------------------------------------------------------------------
    # ORDER EXECUTION — PAPER
    # -------------------------------------------------------------------------
    def paper_execute(self, symbol: str, signal: dict):
        """Handle paper order entry and exit for one symbol."""
        last_high = signal["close"]   # approximation; real high from candle is in df
        last_low  = signal["close"]

        # ── Check exits on open paper positions ────────────────────────────────
        if symbol in self.paper.positions:
            # We'll re-check properly in next cycle with real OHLCV;
            # here we use close as a proxy (conservative — real SL/TP fires on H/L)
            self.paper.update_and_check_exit(symbol, last_high, last_low)
            return   # don't open a new trade while one is open

        # ── Open new position ──────────────────────────────────────────────────
        if signal["long_signal"] or signal["short_signal"]:
            direction = "long" if signal["long_signal"] else "short"
            price     = signal["close"]
            sl        = signal["long_sl"]  if direction == "long"  else signal["short_sl"]
            tp        = signal["long_tp"]  if direction == "long"  else signal["short_tp"]
            qty       = self.compute_qty(symbol, price, sl)

            if qty <= 0:
                return

            self.paper.open_position(
                symbol=symbol, direction=direction, price=price,
                qty=qty, sl=sl, tp=tp,
                trail_points=signal["trail_points"],
                trail_offset=signal["trail_offset"],
            )

    def paper_check_exits(self, symbol: str, df: pd.DataFrame):
        """
        Check paper position exit against actual candle H/L data.
        Call this every loop with fresh OHLCV data for accuracy.
        """
        if symbol not in self.paper.positions or df.empty:
            return
        last = df.iloc[-1]
        self.paper.update_and_check_exit(symbol, float(last["high"]), float(last["low"]))

    # -------------------------------------------------------------------------
    # ORDER EXECUTION — LIVE
    # -------------------------------------------------------------------------
    def live_execute(self, symbol: str, signal: dict):
        """
        Place real orders on Kraken.

        Entry: market order for immediate fill (scalping requires speed).
        SL/TP: submitted as a stop-loss order and a limit order simultaneously
               using Kraken's `createOrder` with conditional close parameters.

        Note: Kraken spot does not support native OCO (One-Cancels-Other) orders
              in the same way as futures. We use separate orders and cancel
              manually when one fills. For true OCO, use Kraken Futures.
        """
        has_position = symbol in self.live_positions

        # ── Check open position for manual SL/TP tracking ─────────────────────
        if has_position:
            self._live_check_exit(symbol)
            return

        # ── Entry signal ───────────────────────────────────────────────────────
        if not (signal["long_signal"] or signal["short_signal"]):
            return

        direction = "long" if signal["long_signal"] else "short"
        price     = signal["close"]
        sl        = signal["long_sl"]  if direction == "long"  else signal["short_sl"]
        tp        = signal["long_tp"]  if direction == "long"  else signal["short_tp"]
        qty       = self.compute_qty(symbol, price, sl)

        if qty <= 0:
            return

        side = "buy" if direction == "long" else "sell"

        log.info(f"[LIVE] Placing {side.upper()} {symbol}  qty={qty}  price≈{price:.4f}")

        try:
            # ── Entry: market order ────────────────────────────────────────────
            entry_order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=qty,
            )
            log.info(f"[LIVE] Entry order placed: {entry_order['id']}")

            # ── Stop-loss order ────────────────────────────────────────────────
            sl_side = "sell" if direction == "long" else "buy"
            sl_order = self.exchange.create_order(
                symbol=symbol,
                type="stop_loss",          # Kraken: "stop-loss" order
                side=sl_side,
                amount=qty,
                price=sl,
                params={"ordertype": "stop-loss", "price": sl},
            )
            log.info(f"[LIVE] SL order placed: {sl_order['id']}  @ {sl:.4f}")

            # ── Take-profit order ──────────────────────────────────────────────
            tp_side = "sell" if direction == "long" else "buy"
            tp_order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=tp_side,
                amount=qty,
                price=tp,
            )
            log.info(f"[LIVE] TP order placed: {tp_order['id']}  @ {tp:.4f}")

            # Track the open position and associated order IDs
            self.live_positions[symbol] = {
                "direction":    direction,
                "qty":          qty,
                "entry":        price,
                "sl":           sl,
                "tp":           tp,
                "sl_order_id":  sl_order["id"],
                "tp_order_id":  tp_order["id"],
                "opened_at":    datetime.now(timezone.utc).isoformat(),
            }

        except ccxt.InsufficientFunds as e:
            log.error(f"[LIVE] Insufficient funds for {symbol}: {e}")
        except ccxt.InvalidOrder as e:
            log.error(f"[LIVE] Invalid order for {symbol}: {e}")
        except ccxt.ExchangeError as e:
            log.error(f"[LIVE] Exchange error for {symbol}: {e}")

    def _live_check_exit(self, symbol: str):
        """
        Polls Kraken to see if either the SL or TP order has been filled.
        If one fills, cancels the other to avoid double-exit.
        """
        pos = self.live_positions.get(symbol)
        if not pos:
            return

        try:
            sl_order = self.exchange.fetch_order(pos["sl_order_id"], symbol)
            tp_order = self.exchange.fetch_order(pos["tp_order_id"], symbol)

            sl_filled = sl_order["status"] == "closed"
            tp_filled = tp_order["status"] == "closed"

            if tp_filled:
                log.info(f"[LIVE] TP HIT for {symbol} — cancelling SL order")
                self._safe_cancel(pos["sl_order_id"], symbol)
                del self.live_positions[symbol]

            elif sl_filled:
                log.info(f"[LIVE] SL HIT for {symbol} — cancelling TP order")
                self._safe_cancel(pos["tp_order_id"], symbol)
                del self.live_positions[symbol]

        except Exception as e:
            log.error(f"[LIVE] Error checking exit for {symbol}: {e}")

    def _safe_cancel(self, order_id: str, symbol: str):
        """Cancel a Kraken order, ignoring errors if already filled/cancelled."""
        try:
            self.exchange.cancel_order(order_id, symbol)
            log.info(f"[LIVE] Cancelled order {order_id}")
        except Exception as e:
            log.warning(f"[LIVE] Could not cancel order {order_id}: {e}")

    # -------------------------------------------------------------------------
    # CONNECTIVITY CHECK
    # -------------------------------------------------------------------------
    def check_connectivity(self) -> bool:
        """Verify API credentials and connectivity before starting the loop."""
        try:
            self.exchange.load_markets()
            log.info("✅ Connected to Kraken successfully.")

            # Validate that all configured pairs exist on Kraken
            for pair in PAIRS:
                if pair not in self.exchange.markets:
                    log.warning(f"⚠️  Pair {pair} not found on Kraken — it will be skipped.")

            if TRADING_MODE == "live":
                balance = self.exchange.fetch_balance()
                usd  = balance.get("USD", {}).get("free", 0)
                usdt = balance.get("USDT", {}).get("free", 0)
                log.info(f"💰 Account balance — USD: {usd}  USDT: {usdt}")

            return True

        except ccxt.AuthenticationError:
            log.error("❌ Authentication failed — check your API key and secret in .env")
            return False
        except ccxt.NetworkError as e:
            log.error(f"❌ Network error connecting to Kraken: {e}")
            return False

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    def run(self):
        """
        Main trading loop. Runs indefinitely until interrupted (Ctrl+C).

        Each cycle:
          For every configured pair:
            1. Fetch fresh candle data
            2. Check if any open position should be exited
            3. Run strategy to look for new entry signals
            4. Execute paper or live orders
          Then sleep until the next cycle.
        """
        log.info("🚀 Starting Kraken Scalping Bot...")

        if not self.check_connectivity():
            log.error("Aborting — fix connectivity issues before restarting.")
            return

        cycle = 0
        try:
            while True:
                cycle += 1
                cycle_start = time.time()
                log.info(f"\n{'─'*60}")
                log.info(f"  CYCLE #{cycle}  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                log.info(f"{'─'*60}")

                for symbol in PAIRS:
                    # Skip pairs not available on Kraken
                    if symbol not in self.exchange.markets:
                        continue

                    log.info(f"📊 Processing {symbol} ...")

                    # 1. Fetch fresh OHLCV data
                    df = self.fetch_ohlcv(symbol)
                    if df.empty:
                        log.warning(f"  No data for {symbol} — skipping")
                        continue

                    # 2. Check exits on existing positions FIRST (before new entry)
                    if TRADING_MODE == "paper":
                        self.paper_check_exits(symbol, df)
                    else:
                        self._live_check_exit(symbol)

                    # 3. Compute strategy signals on latest data
                    signal = self.get_signal(df)

                    log.info(
                        f"  Close={signal.get('close', '?'):.4f}  "
                        f"Long={'✅' if signal['long_signal'] else '❌'}  "
                        f"Short={'✅' if signal['short_signal'] else '❌'}"
                    )

                    # 4. Execute (paper or live)
                    if TRADING_MODE == "paper":
                        self.paper_execute(symbol, signal)
                    else:
                        self.live_execute(symbol, signal)

                # ── Print paper summary every 10 cycles ────────────────────────
                if TRADING_MODE == "paper" and cycle % 10 == 0:
                    self.paper.summary()

                # ── Sleep until next cycle ─────────────────────────────────────
                elapsed = time.time() - cycle_start
                sleep_time = max(0, LOOP_INTERVAL - elapsed)
                log.info(f"⏱  Cycle took {elapsed:.1f}s — sleeping {sleep_time:.0f}s")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            log.info("\n🛑 Bot stopped by user.")
            if TRADING_MODE == "paper":
                self.paper.summary()
            else:
                log.info(f"Open live positions: {list(self.live_positions.keys())}")
                log.info("⚠️  Remember to manually close any open Kraken positions!")

        except Exception as e:
            log.exception(f"Unexpected error: {e}")
            raise


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    bot = KrakenBot()
    bot.run()
