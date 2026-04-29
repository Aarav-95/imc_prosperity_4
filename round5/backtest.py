"""
Backtester for round5/trader.py against the 3-day round-5 price CSVs.
Fill model: aggressive and passive fills against the 3-level order book.
  - Buy order at price P fills all ask levels where ask_px <= P (lowest first)
  - Sell order at price P fills all bid levels where bid_px >= P (highest first)
PnL: realized cash + mark-to-market on open inventory at each timestamp's mid.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from collections import defaultdict
from datamodel import OrderDepth, TradingState, Order

DATA_DIR = os.path.join(os.path.dirname(__file__), "../ROUND_5")
DAYS = [2, 3, 4]
LIMIT = 10


def build_order_depths(row_group):
    """Build {product: OrderDepth} for one timestamp from a slice of rows."""
    ods = {}
    for _, row in row_group.iterrows():
        od = OrderDepth()
        for lvl in [1, 2, 3]:
            bp, bv = row.get(f"bid_price_{lvl}"), row.get(f"bid_volume_{lvl}")
            ap, av = row.get(f"ask_price_{lvl}"), row.get(f"ask_volume_{lvl}")
            if pd.notna(bp) and pd.notna(bv):
                od.buy_orders[int(bp)] = int(bv)
            if pd.notna(ap) and pd.notna(av):
                od.sell_orders[int(ap)] = -int(av)  # Prosperity convention: negative
        ods[row["product"]] = od
    return ods


def fill_orders(orders, order_depths, positions, cash):
    """
    Attempt to fill each order against the snapshot order book.
    Returns (positions, cash, fill_log).
    fill_log: list of (product, price, qty_signed)
    """
    fills = []
    for product, order_list in orders.items():
        od = order_depths.get(product)
        if od is None:
            continue
        pos = positions[product]
        for order in order_list:
            remaining = abs(order.quantity)
            is_buy = order.quantity > 0
            if is_buy:
                # Fill against ask levels where ask_px <= order.price
                for ask_px in sorted(od.sell_orders.keys()):
                    if ask_px > order.price or remaining == 0:
                        break
                    avail = -od.sell_orders[ask_px]   # convert back to positive
                    fill_qty = min(remaining, avail, LIMIT - pos)
                    if fill_qty <= 0:
                        continue
                    pos += fill_qty
                    cash -= ask_px * fill_qty
                    remaining -= fill_qty
                    fills.append((product, ask_px, fill_qty))
            else:
                # Fill against bid levels where bid_px >= order.price
                for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_px < order.price or remaining == 0:
                        break
                    avail = od.buy_orders[bid_px]
                    fill_qty = min(remaining, avail, LIMIT + pos)
                    if fill_qty <= 0:
                        continue
                    pos -= fill_qty
                    cash += bid_px * fill_qty
                    remaining -= fill_qty
                    fills.append((product, bid_px, -fill_qty))
        positions[product] = pos
    return positions, cash, fills


def build_market_trades(trades_df, ts):
    """Build {product: [Trade, ...]} for one timestamp from the trades CSV."""
    from datamodel import Trade
    result = defaultdict(list)
    for _, row in trades_df[trades_df["timestamp"] == ts].iterrows():
        result[row["symbol"]].append(
            Trade(row["symbol"], int(row["price"]), int(row["quantity"]),
                  buyer=row.get("buyer") or None, seller=row.get("seller") or None,
                  timestamp=int(ts))
        )
    return dict(result)


def run_backtest(verbose=False):
    from round5.trader import Trader

    trader = Trader()
    positions = defaultdict(int)
    cash = defaultdict(float)   # per-product realized cash
    trader_data = ""
    total_pnl_ts = []

    for day in DAYS:
        price_file  = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
        trades_file = os.path.join(DATA_DIR, f"trades_round_5_day_{day}.csv")
        df = pd.read_csv(price_file, sep=";")
        trades_df = pd.read_csv(trades_file, sep=";") if os.path.exists(trades_file) else pd.DataFrame()

        timestamps = sorted(df["timestamp"].unique())
        print(f"\n=== Day {day} | {len(timestamps)} timestamps | {df['product'].nunique()} products ===")

        day_fills = 0
        for ts in timestamps:
            ts_rows = df[df["timestamp"] == ts]
            order_depths = build_order_depths(ts_rows)
            market_trades = build_market_trades(trades_df, ts) if not trades_df.empty else {}

            state = TradingState(
                traderData=trader_data,
                timestamp=ts,
                listings={},
                order_depths=order_depths,
                own_trades={},
                market_trades=market_trades,
                position=dict(positions),
                observations=None,
            )

            try:
                result, conversions, trader_data = trader.run(state)
            except Exception as e:
                print(f"  ERROR at ts={ts}: {e}")
                trader_data = ""
                continue

            positions, _, fills = fill_orders(result, order_depths, positions, 0)
            # Update per-product realized cash
            for prod, px, qty in fills:
                cash[prod] -= px * qty   # qty positive = bought, so cash decreases
            day_fills += len(fills)

            # Mark-to-market: sum over all products
            mtm = 0.0
            for prod, pos in positions.items():
                mid_rows = ts_rows[ts_rows["product"] == prod]
                if not mid_rows.empty:
                    mid_px = mid_rows.iloc[0]["mid_price"]
                    mtm += pos * mid_px
            realized = sum(cash.values())
            total_pnl_ts.append((day, ts, realized + mtm))

        end_realized = sum(cash.values())
        end_mtm = 0.0
        last_ts_rows = df[df["timestamp"] == timestamps[-1]]
        for prod, pos in positions.items():
            mid_rows = last_ts_rows[last_ts_rows["product"] == prod]
            if not mid_rows.empty:
                end_mtm += pos * mid_rows.iloc[0]["mid_price"]
        print(f"  Day {day} fills: {day_fills}  |  EOD PnL: {end_realized + end_mtm:,.1f}  "
              f"(realized {end_realized:,.1f}  +  mtm {end_mtm:,.1f})")

    # Final summary
    print("\n=== FINAL SUMMARY ===")
    final_realized = sum(cash.values())
    print(f"Total realized cash: {final_realized:,.1f}")

    # Per-product breakdown
    prod_pnl = {}
    for prod, c in cash.items():
        last_price = None
        for day in reversed(DAYS):
            pf = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
            tmp = pd.read_csv(pf, sep=";")
            row = tmp[(tmp["product"] == prod)].tail(1)
            if not row.empty:
                last_price = row.iloc[0]["mid_price"]
                break
        mtm_val = positions[prod] * (last_price or 0)
        prod_pnl[prod] = c + mtm_val

    sorted_pnl = sorted(prod_pnl.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Product':<40} {'PnL':>10}  {'Pos':>5}")
    print("-" * 58)
    for prod, pnl in sorted_pnl:
        pos = positions[prod]
        marker = " ***" if abs(pnl) > 500 else ""
        print(f"{prod:<40} {pnl:>10.1f}  {pos:>5}{marker}")
    total = sum(v for v in prod_pnl.values())
    print("-" * 58)
    print(f"{'TOTAL':<40} {total:>10.1f}")

    # PnL curve
    pnl_df = pd.DataFrame(total_pnl_ts, columns=["day", "ts", "pnl"])
    print(f"\nPeak PnL: {pnl_df['pnl'].max():,.1f}  |  Final PnL: {pnl_df['pnl'].iloc[-1]:,.1f}")
    print(f"Drawdown: {(pnl_df['pnl'].max() - pnl_df['pnl'].min()):,.1f}")

    return pnl_df, prod_pnl


if __name__ == "__main__":
    run_backtest(verbose="--verbose" in sys.argv)
