"""
Round 3 backtester — runs against prices_round_3_day_{0,1,2}.csv
Usage:
  python3 round3/backtest_r3.py                  # test improved trader
  python3 round3/backtest_r3.py --compare        # improved vs original side-by-side
"""

import csv, io, json, sys, importlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, ".")   # so datamodel imports work from repo root

from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation

DATA_DIR = "ROUND_3"
DAYS = [0, 1, 2]

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}
PRODUCTS = list(POSITION_LIMITS.keys())


# ── CSV helpers ──────────────────────────────────────────────────────────────
def _f(v) -> Optional[float]:
    try: return float(v)
    except: return None

def _i(v) -> Optional[int]:
    f = _f(v)
    return int(f) if f is not None else None

def load_prices() -> Dict[int, Dict[str, dict]]:
    data: Dict[int, Dict[str, dict]] = {}
    for day in DAYS:
        with open(f"{DATA_DIR}/prices_round_3_day_{day}.csv") as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = _i(row["timestamp"])
                if ts is None: continue
                key = day * 1_000_000 + ts
                data.setdefault(key, {})[row["product"]] = row
    return data

def load_trades() -> Dict[int, List[dict]]:
    data: Dict[int, List[dict]] = defaultdict(list)
    for day in DAYS:
        with open(f"{DATA_DIR}/trades_round_3_day_{day}.csv") as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = _i(row["timestamp"])
                if ts is None: continue
                # day field is in the prices CSV but not trades — infer from filename
                data[day * 1_000_000 + ts].append(row)
    return data

def build_od(row: dict) -> OrderDepth:
    od = OrderDepth()
    for i in range(1, 4):
        p, v = _i(row.get(f"bid_price_{i}")), _i(row.get(f"bid_volume_{i}"))
        if p is not None and v is not None: od.buy_orders[p] = v
        p, v = _i(row.get(f"ask_price_{i}")), _i(row.get(f"ask_volume_{i}"))
        if p is not None and v is not None: od.sell_orders[p] = -v
    return od


# ── Order matching ────────────────────────────────────────────────────────────
def match(orders: List[Order], od: OrderDepth, pos: int, limit: int) -> List[Tuple[int,int]]:
    fills = []
    for order in orders:
        if order.quantity > 0:   # buy
            rem = min(order.quantity, limit - (pos + sum(q for _,q in fills if q > 0)))
            for ap in sorted(od.sell_orders):
                if rem <= 0 or order.price < ap: break
                qty = min(rem, -od.sell_orders[ap])
                if qty > 0: fills.append((ap, qty)); rem -= qty
        elif order.quantity < 0:  # sell
            rem = min(-order.quantity, limit + (pos + sum(q for _,q in fills)))
            for bp in sorted(od.buy_orders, reverse=True):
                if rem <= 0 or order.price > bp: break
                qty = min(rem, od.buy_orders[bp])
                if qty > 0: fills.append((bp, -qty)); rem -= qty
    return fills


# ── Main backtest engine ──────────────────────────────────────────────────────
def run_backtest(module, label: str, verbose=False) -> dict:
    trader = module.Trader()

    prices  = load_prices()
    trades  = load_trades()
    timestamps = sorted(prices.keys())

    positions: Dict[str, int] = defaultdict(int)
    cash = 0.0
    trader_data_str = ""
    pnl_history = []
    fills_by_product: Dict[str, int] = defaultdict(int)
    vol_by_product:   Dict[str, int] = defaultdict(int)

    for ts in timestamps:
        price_rows = prices[ts]
        raw_ts = ts % 1_000_000

        # Build order depths
        listings = {p: Listing(p, p, "XIRECS") for p in PRODUCTS}
        order_depths = {p: (build_od(price_rows[p]) if p in price_rows else OrderDepth())
                        for p in PRODUCTS}

        # Market trades
        mt: Dict[Symbol, List[Trade]] = defaultdict(list)
        for t in trades.get(ts, []):
            sym = t.get("symbol", "")
            if sym in POSITION_LIMITS:
                mt[sym].append(Trade(sym,
                    int(_f(t["price"]) or 0),
                    int(_f(t["quantity"]) or 0),
                    t.get("buyer",""), t.get("seller",""), raw_ts))

        state = TradingState(
            traderData=trader_data_str,
            timestamp=raw_ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=defaultdict(list),
            market_trades=dict(mt),
            position=dict(positions),
            observations=Observation({}, {}),
        )

        # Run trader (suppress stdout from logger)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result, _, trader_data_str = trader.run(state)
        except Exception as e:
            sys.stdout = old_stdout
            if verbose: print(f"[{ts}] ERROR: {e}")
            continue
        finally:
            sys.stdout = old_stdout

        # Match orders
        for product, orders in result.items():
            if not orders: continue
            od  = order_depths.get(product, OrderDepth())
            lim = POSITION_LIMITS.get(product, 300)
            for price, qty in match(orders, od, positions[product], lim):
                positions[product] += qty
                cash -= price * qty
                fills_by_product[product] += 1
                vol_by_product[product]   += abs(qty)
                if verbose:
                    print(f"[{ts}] {'BUY' if qty>0 else 'SELL'} {product} {abs(qty)}@{price}")

        # Mark-to-market PnL
        mtm = cash
        for prod, pos in positions.items():
            if prod in price_rows:
                mid = _f(price_rows[prod].get("mid_price"))
                if mid: mtm += pos * mid
        pnl_history.append((ts, mtm))

    final_pnl = pnl_history[-1][1] if pnl_history else 0.0

    # Max drawdown
    peak, max_dd = 0.0, 0.0
    for _, pnl in pnl_history:
        if pnl > peak: peak = pnl
        if peak - pnl > max_dd: max_dd = peak - pnl

    # PnL by day
    day_pnl = {}
    prev = 0.0
    for day in DAYS:
        lo, hi = day*1_000_000, (day+1)*1_000_000
        day_pts = [p for ts,p in pnl_history if lo <= ts < hi]
        if day_pts:
            day_pnl[day] = day_pts[-1] - prev
            prev = day_pts[-1]

    return dict(
        label=label,
        final_pnl=final_pnl,
        max_drawdown=max_dd,
        day_pnl=day_pnl,
        fills_by_product=dict(fills_by_product),
        vol_by_product=dict(vol_by_product),
        final_positions=dict(positions),
        pnl_history=pnl_history,
    )


def print_result(r: dict):
    print(f"\n{'='*65}")
    print(f"  {r['label']}")
    print(f"{'='*65}")
    print(f"  Final PnL   : {r['final_pnl']:>12,.1f}")
    print(f"  Max Drawdown: {r['max_drawdown']:>12,.1f}")
    print(f"\n  PnL by day:")
    for day, pnl in sorted(r['day_pnl'].items()):
        tte = 8 - day
        print(f"    Day {day} (TTE={tte}d): {pnl:>+12,.1f}")
    print(f"\n  Top products by fills:")
    by_fills = sorted(r['fills_by_product'].items(), key=lambda x: -x[1])
    for prod, fills in by_fills[:8]:
        vol = r['vol_by_product'].get(prod, 0)
        print(f"    {prod:<28} fills={fills:>5,}  vol={vol:>6,}")
    print(f"\n  Final positions:")
    for prod, pos in sorted(r['final_positions'].items()):
        if pos != 0:
            print(f"    {prod:<28} {pos:>+6}")
    print(f"{'='*65}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    sys.path.insert(0, "round3")

    if mode == "--compare":
        # Run both versions
        import trader_round3 as improved
        import trader_round3_original as original

        print("\nRunning improved trader...")
        r_imp = run_backtest(improved, "IMPROVED (all 6 strikes + deep OTM)")
        print("Running original trader...")
        r_org = run_backtest(original, "ORIGINAL (VEV_5200 only + smile fit)")

        print_result(r_imp)
        print_result(r_org)

        print(f"\n{'='*65}")
        print(f"  COMPARISON")
        print(f"{'='*65}")
        diff = r_imp['final_pnl'] - r_org['final_pnl']
        sign = "+" if diff >= 0 else ""
        print(f"  Improved PnL : {r_imp['final_pnl']:>12,.1f}")
        print(f"  Original PnL : {r_org['final_pnl']:>12,.1f}")
        print(f"  Difference   : {sign}{diff:>11,.1f}  ({'BETTER' if diff>0 else 'WORSE'})")
        print(f"{'='*65}")
    else:
        import trader_round3 as trader
        print("Running improved trader backtest...")
        r = run_backtest(trader, "IMPROVED trader_round3.py")
        print_result(r)
