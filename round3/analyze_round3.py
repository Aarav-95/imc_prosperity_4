"""
Round 3 Analysis: VELVETFRUIT_EXTRACT, VEV_* vouchers, HYDROGEL_PACK
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

def autocorr(series, lag):
    n = len(series)
    if n <= lag:
        return float('nan')
    m = sum(series) / n
    num = sum((series[i] - m) * (series[i - lag] - m) for i in range(lag, n))
    den = sum((x - m) ** 2 for x in series)
    return num / den if den != 0 else 0

def percentile(sorted_list, p):
    idx = p / 100 * (len(sorted_list) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_list[lo]
    return sorted_list[lo] * (hi - idx) + sorted_list[hi] * (idx - lo)

def linear_regression(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxx = sum(xi * xi for xi in x)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0, sy / n if n > 0 else 0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept

# ── Load all Round 3 data ──────────────────────────────────────────
all_prices = []
all_trades = []
for day in [0, 1, 2]:
    all_prices.extend(read_csv(f"data/Round 3/prices_round_3_day_{day}.csv"))
    all_trades.extend(read_csv(f"data/Round 3/trades_round_3_day_{day}.csv"))

products = sorted(set(r["product"] for r in all_prices))
print(f"Products found: {products}")
print(f"Total price rows: {len(all_prices)}")
print(f"Total trade rows: {len(all_trades)}")

# Separate products into categories
underlying = "VELVETFRUIT_EXTRACT"
vouchers = sorted([p for p in products if p.startswith("VEV_")])
other = [p for p in products if p != underlying and not p.startswith("VEV_")]

print(f"\nUnderlying: {underlying}")
print(f"Vouchers:   {vouchers}")
print(f"Other:      {other}")

# ══════════════════════════════════════════════════════════════════
# 1) ANALYZE EACH PRODUCT INDIVIDUALLY
# ══════════════════════════════════════════════════════════════════

for product in products:
    print("\n" + "=" * 70)
    print(f"  PRODUCT: {product}")
    print("=" * 70)

    rows = [r for r in all_prices if r["product"] == product]
    trades = [r for r in all_trades if r["symbol"] == product]

    # ── Mid price stats ────────────────────────────────────────
    mids = [to_float(r["mid_price"]) for r in rows]
    mids = [m for m in mids if m is not None]
    timestamps = [to_float(r["timestamp"]) for r in rows if to_float(r["mid_price"]) is not None]

    if not mids:
        print("  No mid price data!")
        continue

    n = len(mids)
    mean_mid = sum(mids) / n
    sorted_mids = sorted(mids)
    median_mid = sorted_mids[n // 2]
    variance = sum((x - mean_mid) ** 2 for x in mids) / n
    std_mid = math.sqrt(variance)

    print(f"\n--- Mid Price Stats ({n} data points) ---")
    print(f"  Mean:     {mean_mid:.4f}")
    print(f"  Median:   {median_mid:.4f}")
    print(f"  Std Dev:  {std_mid:.4f}")
    print(f"  Min:      {min(mids):.4f}")
    print(f"  Max:      {max(mids):.4f}")
    print(f"  Range:    {max(mids) - min(mids):.4f}")
    print(f"  Q1:       {percentile(sorted_mids, 25):.4f}")
    print(f"  Q3:       {percentile(sorted_mids, 75):.4f}")

    # ── Tick-to-tick returns ───────────────────────────────────
    returns = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    if returns:
        r_mean = sum(returns) / len(returns)
        r_var = sum((x - r_mean) ** 2 for x in returns) / len(returns)
        r_std = math.sqrt(r_var)

        print(f"\n--- Returns (tick-to-tick mid Δ) ---")
        print(f"  Mean return:     {r_mean:.6f}")
        print(f"  Std of returns:  {r_std:.4f}")
        print(f"  Max up move:     {max(returns):.4f}")
        print(f"  Max down move:   {min(returns):.4f}")

        if r_std > 0:
            skew = sum(((x - r_mean) / r_std) ** 3 for x in returns) / len(returns)
            kurt = sum(((x - r_mean) / r_std) ** 4 for x in returns) / len(returns) - 3
            print(f"  Skewness:        {skew:.4f}")
            print(f"  Excess kurtosis: {kurt:.4f}")

        # ── Autocorrelation ────────────────────────────────────
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

    # ── Spread analysis ────────────────────────────────────────
    spreads = []
    for r in rows:
        b1 = to_float(r["bid_price_1"])
        a1 = to_float(r["ask_price_1"])
        if b1 is not None and a1 is not None:
            spreads.append(a1 - b1)

    if spreads:
        print(f"\n--- Bid-Ask Spread ---")
        print(f"  Mean:   {sum(spreads)/len(spreads):.4f}")
        print(f"  Median: {sorted(spreads)[len(spreads)//2]:.4f}")
        print(f"  Min:    {min(spreads):.4f}")
        print(f"  Max:    {max(spreads):.4f}")

    # ── Top-of-book volume ─────────────────────────────────────
    bv1s = [to_float(r["bid_volume_1"]) for r in rows]
    av1s = [to_float(r["ask_volume_1"]) for r in rows]
    bv1s = [v for v in bv1s if v is not None]
    av1s = [v for v in av1s if v is not None]
    print(f"\n--- Top-of-Book Volume ---")
    print(f"  Avg bid_vol_1: {sum(bv1s)/len(bv1s):.1f}" if bv1s else "  No bids")
    print(f"  Avg ask_vol_1: {sum(av1s)/len(av1s):.1f}" if av1s else "  No asks")

    # ── Book depth ─────────────────────────────────────────────
    total = len(rows)
    has_b2 = sum(1 for r in rows if r.get("bid_price_2", "") != "") / total * 100
    has_b3 = sum(1 for r in rows if r.get("bid_price_3", "") != "") / total * 100
    has_a2 = sum(1 for r in rows if r.get("ask_price_2", "") != "") / total * 100
    has_a3 = sum(1 for r in rows if r.get("ask_price_3", "") != "") / total * 100
    print(f"\n--- Book Depth (% ticks with level present) ---")
    print(f"  Bid L2: {has_b2:.1f}%   Bid L3: {has_b3:.1f}%")
    print(f"  Ask L2: {has_a2:.1f}%   Ask L3: {has_a3:.1f}%")

    # ── Per-day trend ──────────────────────────────────────────
    days = sorted(set(r["day"] for r in rows))
    print(f"\n--- Per-Day Trend ---")
    for d in days:
        day_mids = [to_float(r["mid_price"]) for r in rows if r["day"] == d]
        day_mids = [m for m in day_mids if m is not None]
        if len(day_mids) >= 2:
            start, end = day_mids[0], day_mids[-1]
            day_mean = sum(day_mids) / len(day_mids)
            day_var = sum((x - day_mean) ** 2 for x in day_mids) / len(day_mids)
            day_std = math.sqrt(day_var)
            print(f"  Day {d}: {start:.2f} → {end:.2f}  (Δ = {end - start:+.2f}, mean={day_mean:.2f}, std={day_std:.2f})")

    # ── Slope per tick (linear regression on mid vs timestamp) ──
    if len(mids) > 10 and timestamps:
        slope, intercept = linear_regression(timestamps, mids)
        print(f"\n--- Linear Trend ---")
        print(f"  Slope per timestamp: {slope:.6f}")
        print(f"  Intercept:           {intercept:.4f}")
        print(f"  Implied daily move:  {slope * 10000:.2f} (approx)")

    # ── Trade data ─────────────────────────────────────────────
    print(f"\n--- Market Trades ---")
    print(f"  Total trades: {len(trades)}")
    if trades:
        tprices = [to_float(t["price"]) for t in trades]
        tqtys = [to_float(t["quantity"]) for t in trades]
        tprices = [p for p in tprices if p is not None]
        tqtys = [q for q in tqtys if q is not None]
        if tprices:
            print(f"  Avg price:    {sum(tprices)/len(tprices):.4f}")
        if tqtys:
            print(f"  Avg quantity: {sum(tqtys)/len(tqtys):.1f}")
            print(f"  Total volume: {sum(tqtys):.0f}")

        # Trades per day
        trades_by_day = defaultdict(list)
        for t in trades:
            ts = to_float(t["timestamp"])
            # Try to figure out the day from the prices data
            trades_by_day["all"].append(t)
        
        # Buyer/seller patterns
        buyers = defaultdict(int)
        sellers = defaultdict(int)
        for t in trades:
            if t.get("buyer"):
                buyers[t["buyer"]] += 1
            if t.get("seller"):
                sellers[t["seller"]] += 1
        if buyers:
            print(f"\n  Top buyers: {sorted(buyers.items(), key=lambda x: -x[1])[:5]}")
        if sellers:
            print(f"  Top sellers: {sorted(sellers.items(), key=lambda x: -x[1])[:5]}")

    # ── Mid price distribution ─────────────────────────────────
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
            print(f"  {edge_lo:10.2f} - {edge_hi:10.2f}: {buckets[i]:5d} {bar}")


# ══════════════════════════════════════════════════════════════════
# 2) VOUCHER / OPTIONS ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  VOUCHER / OPTIONS ANALYSIS")
print("=" * 70)

# Extract strike prices from voucher names
strikes = {}
for v in vouchers:
    strike = int(v.split("_")[1])
    strikes[v] = strike
    print(f"  {v}: Strike = {strike}")

# Get underlying price series per timestamp
underlying_by_ts = {}
for r in all_prices:
    if r["product"] == underlying:
        ts = to_float(r["timestamp"])
        day = r["day"]
        mid = to_float(r["mid_price"])
        if ts is not None and mid is not None:
            key = (day, int(ts))
            underlying_by_ts[key] = mid

# Analyze voucher vs underlying relationship
print(f"\n--- Voucher Mid Price vs Intrinsic Value ---")
print(f"  (Intrinsic = max(0, Underlying - Strike) for calls)")
print()

for v in vouchers:
    strike = strikes[v]
    v_rows = [r for r in all_prices if r["product"] == v]
    
    intrinsics = []
    time_values = []
    voucher_mids_list = []
    underlying_mids_list = []
    
    for r in v_rows:
        ts = to_float(r["timestamp"])
        day = r["day"]
        v_mid = to_float(r["mid_price"])
        key = (day, int(ts)) if ts is not None else None
        
        if key and key in underlying_by_ts and v_mid is not None:
            u_mid = underlying_by_ts[key]
            intrinsic = max(0, u_mid - strike)
            tv = v_mid - intrinsic
            intrinsics.append(intrinsic)
            time_values.append(tv)
            voucher_mids_list.append(v_mid)
            underlying_mids_list.append(u_mid)
    
    if intrinsics:
        avg_intrinsic = sum(intrinsics) / len(intrinsics)
        avg_tv = sum(time_values) / len(time_values)
        avg_v_mid = sum(voucher_mids_list) / len(voucher_mids_list)
        avg_u_mid = sum(underlying_mids_list) / len(underlying_mids_list)
        
        # What % of the time is it ITM
        itm_pct = sum(1 for i in intrinsics if i > 0) / len(intrinsics) * 100
        
        print(f"  {v} (K={strike}):")
        print(f"    Avg voucher mid:  {avg_v_mid:.4f}")
        print(f"    Avg underlying:   {avg_u_mid:.2f}")
        print(f"    Avg intrinsic:    {avg_intrinsic:.4f}")
        print(f"    Avg time value:   {avg_tv:.4f}")
        print(f"    ITM %:            {itm_pct:.1f}%")
        print(f"    Min time value:   {min(time_values):.4f}")
        print(f"    Max time value:   {max(time_values):.4f}")
        print()

# ── Delta estimation: how much does voucher price change per unit of underlying? ──
print(f"\n--- Delta Estimation (Voucher Δ / Underlying Δ) ---")
for v in vouchers:
    strike = strikes[v]
    v_rows = [r for r in all_prices if r["product"] == v]
    
    # Build aligned time series
    v_by_ts = {}
    for r in v_rows:
        ts = to_float(r["timestamp"])
        day = r["day"]
        mid = to_float(r["mid_price"])
        if ts is not None and mid is not None:
            key = (day, int(ts))
            v_by_ts[key] = mid
    
    # Find common timestamps
    common_keys = sorted(set(v_by_ts.keys()) & set(underlying_by_ts.keys()))
    
    if len(common_keys) > 10:
        v_series = [v_by_ts[k] for k in common_keys]
        u_series = [underlying_by_ts[k] for k in common_keys]
        
        v_returns = [v_series[i] - v_series[i-1] for i in range(1, len(v_series))]
        u_returns = [u_series[i] - u_series[i-1] for i in range(1, len(u_series))]
        
        # Delta = cov(v_ret, u_ret) / var(u_ret)
        v_mean = sum(v_returns) / len(v_returns)
        u_mean = sum(u_returns) / len(u_returns)
        cov = sum((v - v_mean) * (u - u_mean) for v, u in zip(v_returns, u_returns)) / len(v_returns)
        u_var = sum((u - u_mean) ** 2 for u in u_returns) / len(u_returns)
        delta = cov / u_var if u_var > 0 else 0
        
        # Correlation
        v_std_r = math.sqrt(sum((v - v_mean)**2 for v in v_returns) / len(v_returns))
        u_std_r = math.sqrt(u_var)
        corr = cov / (v_std_r * u_std_r) if v_std_r > 0 and u_std_r > 0 else 0
        
        print(f"  {v} (K={strike}): delta={delta:.4f}, corr={corr:.4f}")


# ══════════════════════════════════════════════════════════════════
# 3) CROSS-PRODUCT CORRELATIONS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  CROSS-PRODUCT CORRELATIONS")
print("=" * 70)

# Build mid-price series indexed by (day, timestamp)
product_series = {}
for p in products:
    ts_map = {}
    for r in all_prices:
        if r["product"] == p:
            ts = to_float(r["timestamp"])
            day = r["day"]
            mid = to_float(r["mid_price"])
            if ts is not None and mid is not None:
                key = (day, int(ts))
                ts_map[key] = mid
    product_series[p] = ts_map

# Compute pairwise correlations for key products
key_products = [underlying] + other + vouchers[:3]  # underlying, HYDROGEL, first few vouchers
for i, p1 in enumerate(key_products):
    for p2 in key_products[i+1:]:
        common = sorted(set(product_series[p1].keys()) & set(product_series[p2].keys()))
        if len(common) > 100:
            s1 = [product_series[p1][k] for k in common]
            s2 = [product_series[p2][k] for k in common]
            # Returns correlation
            r1 = [s1[i] - s1[i-1] for i in range(1, len(s1))]
            r2 = [s2[i] - s2[i-1] for i in range(1, len(s2))]
            m1 = sum(r1) / len(r1)
            m2 = sum(r2) / len(r2)
            cov = sum((a - m1)*(b - m2) for a, b in zip(r1, r2)) / len(r1)
            std1 = math.sqrt(sum((a - m1)**2 for a in r1) / len(r1))
            std2 = math.sqrt(sum((b - m2)**2 for b in r2) / len(r2))
            corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0
            # Level correlation
            lm1 = sum(s1) / len(s1)
            lm2 = sum(s2) / len(s2)
            lcov = sum((a - lm1)*(b - lm2) for a, b in zip(s1, s2)) / len(s1)
            lstd1 = math.sqrt(sum((a - lm1)**2 for a in s1) / len(s1))
            lstd2 = math.sqrt(sum((b - lm2)**2 for b in s2) / len(s2))
            lcorr = lcov / (lstd1 * lstd2) if lstd1 > 0 and lstd2 > 0 else 0
            print(f"  {p1:25s} vs {p2:25s}: return_corr={corr:+.4f}, level_corr={lcorr:+.4f}")


# ══════════════════════════════════════════════════════════════════
# 4) TRADE FLOW ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  TRADE FLOW ANALYSIS")
print("=" * 70)

trade_products = sorted(set(t["symbol"] for t in all_trades))
for product in trade_products:
    trades = [t for t in all_trades if t["symbol"] == product]
    print(f"\n  {product}: {len(trades)} trades")
    
    prices = [to_float(t["price"]) for t in trades]
    qtys = [to_float(t["quantity"]) for t in trades]
    prices = [p for p in prices if p is not None]
    qtys = [q for q in qtys if q is not None]
    
    if prices:
        print(f"    Price range: {min(prices):.2f} - {max(prices):.2f}")
        print(f"    Avg price:   {sum(prices)/len(prices):.2f}")
    if qtys:
        print(f"    Total vol:   {sum(qtys):.0f}")
        print(f"    Avg size:    {sum(qtys)/len(qtys):.1f}")
    
    # Trade timing
    trade_ts = [to_float(t["timestamp"]) for t in trades]
    trade_ts = [t for t in trade_ts if t is not None]
    if trade_ts:
        print(f"    First trade at t={min(trade_ts):.0f}, Last at t={max(trade_ts):.0f}")
        # Trade frequency
        if len(trade_ts) > 1:
            gaps = [trade_ts[i] - trade_ts[i-1] for i in range(1, len(trade_ts)) if trade_ts[i] > trade_ts[i-1]]
            if gaps:
                print(f"    Avg gap between trades: {sum(gaps)/len(gaps):.0f} ticks")

    # Currency info
    currencies = set(t.get("currency", "") for t in trades)
    if currencies and currencies != {""}:
        print(f"    Currencies: {currencies}")

print("\n\nDone!")
