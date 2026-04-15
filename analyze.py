"""
Analysis of ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT using only stdlib.
"""
import csv
import math
from collections import defaultdict

def read_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows

def to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

# ── Load all price data ────────────────────────────────────────────
all_prices = []
for day in [-2, -1, 0]:
    all_prices.extend(read_csv(f"data/prices_round_1_day_{day}.csv"))

all_trades = []
for day in [-2, -1, 0]:
    all_trades.extend(read_csv(f"data/trades_round_1_day_{day}.csv"))

for product in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
    print("=" * 70)
    print(f"  PRODUCT: {product}")
    print("=" * 70)

    rows = [r for r in all_prices if r["product"] == product]
    trades = [r for r in all_trades if r["symbol"] == product]

    # ── Mid price stats ────────────────────────────────────────────
    mids = [to_float(r["mid_price"]) for r in rows]
    mids = [m for m in mids if m is not None]

    n = len(mids)
    mean_mid = sum(mids) / n
    median_mid = sorted(mids)[n // 2]
    variance = sum((x - mean_mid) ** 2 for x in mids) / n
    std_mid = math.sqrt(variance)

    print(f"\n--- Mid Price Stats ({n} data points) ---")
    print(f"  Mean:     {mean_mid:.2f}")
    print(f"  Median:   {median_mid:.2f}")
    print(f"  Std Dev:  {std_mid:.2f}")
    print(f"  Min:      {min(mids):.2f}")
    print(f"  Max:      {max(mids):.2f}")
    print(f"  Range:    {max(mids) - min(mids):.2f}")

    # ── Tick-to-tick returns ───────────────────────────────────────
    returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    r_mean = sum(returns) / len(returns)
    r_var = sum((x - r_mean) ** 2 for x in returns) / len(returns)
    r_std = math.sqrt(r_var)

    print(f"\n--- Returns (tick-to-tick mid Δ) ---")
    print(f"  Mean return:     {r_mean:.4f}")
    print(f"  Std of returns:  {r_std:.4f}")
    print(f"  Max up move:     {max(returns):.2f}")
    print(f"  Max down move:   {min(returns):.2f}")

    # Skewness
    if r_std > 0:
        skew = sum(((x - r_mean) / r_std) ** 3 for x in returns) / len(returns)
        kurt = sum(((x - r_mean) / r_std) ** 4 for x in returns) / len(returns) - 3
        print(f"  Skewness:        {skew:.4f}")
        print(f"  Excess kurtosis: {kurt:.4f}")

    # ── Autocorrelation ────────────────────────────────────────────
    def autocorr(series, lag):
        n = len(series)
        if n <= lag:
            return float('nan')
        m = sum(series) / n
        num = sum((series[i] - m) * (series[i - lag] - m) for i in range(lag, n))
        den = sum((x - m) ** 2 for x in series)
        return num / den if den != 0 else 0

    ac1 = autocorr(returns, 1)
    ac5 = autocorr(returns, 5)
    ac10 = autocorr(returns, 10)
    ac20 = autocorr(returns, 20)

    label1 = "(mean-reverting)" if ac1 < -0.05 else "(trending)" if ac1 > 0.05 else "(neutral)"
    print(f"\n--- Autocorrelation of Returns ---")
    print(f"  Lag-1:   {ac1:+.4f}  {label1}")
    print(f"  Lag-5:   {ac5:+.4f}")
    print(f"  Lag-10:  {ac10:+.4f}")
    print(f"  Lag-20:  {ac20:+.4f}")

    # ── Spread analysis ────────────────────────────────────────────
    spreads = []
    for r in rows:
        b1 = to_float(r["bid_price_1"])
        a1 = to_float(r["ask_price_1"])
        if b1 is not None and a1 is not None:
            spreads.append(a1 - b1)

    if spreads:
        print(f"\n--- Bid-Ask Spread ---")
        print(f"  Mean:   {sum(spreads)/len(spreads):.2f}")
        print(f"  Median: {sorted(spreads)[len(spreads)//2]:.2f}")
        print(f"  Min:    {min(spreads):.2f}")
        print(f"  Max:    {max(spreads):.2f}")

    # ── Top-of-book volume ─────────────────────────────────────────
    bv1s = [to_float(r["bid_volume_1"]) for r in rows]
    av1s = [to_float(r["ask_volume_1"]) for r in rows]
    bv1s = [v for v in bv1s if v is not None]
    av1s = [v for v in av1s if v is not None]
    print(f"\n--- Top-of-Book Volume ---")
    print(f"  Avg bid_vol_1: {sum(bv1s)/len(bv1s):.1f}" if bv1s else "  No bids")
    print(f"  Avg ask_vol_1: {sum(av1s)/len(av1s):.1f}" if av1s else "  No asks")

    # ── Book depth ─────────────────────────────────────────────────
    total = len(rows)
    has_b2 = sum(1 for r in rows if r.get("bid_price_2", "") != "") / total * 100
    has_b3 = sum(1 for r in rows if r.get("bid_price_3", "") != "") / total * 100
    has_a2 = sum(1 for r in rows if r.get("ask_price_2", "") != "") / total * 100
    has_a3 = sum(1 for r in rows if r.get("ask_price_3", "") != "") / total * 100
    print(f"\n--- Book Depth (% ticks with level present) ---")
    print(f"  Bid L2: {has_b2:.1f}%   Bid L3: {has_b3:.1f}%")
    print(f"  Ask L2: {has_a2:.1f}%   Ask L3: {has_a3:.1f}%")

    # ── Per-day trend ──────────────────────────────────────────────
    days = sorted(set(r["day"] for r in rows))
    print(f"\n--- Per-Day Trend ---")
    for d in days:
        day_mids = [to_float(r["mid_price"]) for r in rows if r["day"] == d]
        day_mids = [m for m in day_mids if m is not None]
        if len(day_mids) >= 2:
            start, end = day_mids[0], day_mids[-1]
            print(f"  Day {d}: {start:.1f} → {end:.1f}  (Δ = {end - start:+.1f})")

    # ── Trade data ─────────────────────────────────────────────────
    print(f"\n--- Market Trades ---")
    print(f"  Total trades: {len(trades)}")
    if trades:
        tprices = [to_float(t["price"]) for t in trades]
        tqtys = [to_float(t["quantity"]) for t in trades]
        tprices = [p for p in tprices if p is not None]
        tqtys = [q for q in tqtys if q is not None]
        print(f"  Avg price:    {sum(tprices)/len(tprices):.2f}")
        print(f"  Avg quantity: {sum(tqtys)/len(tqtys):.1f}")
        print(f"  Total volume: {sum(tqtys):.0f}")

    # ── Mid price distribution ─────────────────────────────────────
    print(f"\n--- Mid Price Distribution (10 buckets) ---")
    lo, hi = min(mids), max(mids)
    if lo == hi:
        print(f"  All values = {lo}")
    else:
        nbins = 10
        width = (hi - lo) / nbins
        buckets = [0] * nbins
        for m in mids:
            idx = min(int((m - lo) / width), nbins - 1)
            buckets[idx] += 1
        max_count = max(buckets)
        for i in range(nbins):
            edge_lo = lo + i * width
            edge_hi = lo + (i + 1) * width
            bar = "█" * int(buckets[i] / max_count * 30) if max_count > 0 else ""
            print(f"  {edge_lo:10.1f} - {edge_hi:10.1f}: {buckets[i]:5d} {bar}")

    # ── Mean crossings ─────────────────────────────────────────────
    pct_above = sum(1 for m in mids if m > mean_mid) / n * 100
    pct_below = sum(1 for m in mids if m < mean_mid) / n * 100
    centered = [m - mean_mid for m in mids]
    crossings = sum(
        1 for i in range(1, len(centered))
        if (centered[i - 1] > 0 and centered[i] < 0) or (centered[i - 1] < 0 and centered[i] > 0)
    )
    print(f"\n--- Oscillation around mean ({mean_mid:.1f}) ---")
    print(f"  % above mean: {pct_above:.1f}%")
    print(f"  % below mean: {pct_below:.1f}%")
    print(f"  Mean crossings: {crossings} (out of {n} ticks)")
    print(f"  Crossing rate:  {crossings/n*100:.1f}%")

    # ── Moving average analysis (EMA 20 vs EMA 50) ─────────────────
    def ema(series, span):
        alpha = 2.0 / (span + 1)
        result = [series[0]]
        for i in range(1, len(series)):
            result.append(alpha * series[i] + (1 - alpha) * result[-1])
        return result

    if len(mids) > 50:
        ema20 = ema(mids, 20)
        ema50 = ema(mids, 50)
        # Count EMA crossovers
        cross_count = 0
        for i in range(51, len(mids)):
            prev_diff = ema20[i-1] - ema50[i-1]
            curr_diff = ema20[i] - ema50[i]
            if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                cross_count += 1
        pct_ema20_above = sum(1 for i in range(50, len(mids)) if ema20[i] > ema50[i]) / (len(mids) - 50) * 100
        print(f"\n--- EMA(20) vs EMA(50) ---")
        print(f"  EMA crossovers:      {cross_count}")
        print(f"  % time EMA20 > EMA50: {pct_ema20_above:.1f}%")

    print()
