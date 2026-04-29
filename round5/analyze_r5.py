"""
Round 5 — Comprehensive analysis of 50 products across 10 families.
Segmentation, correlation, lead/lag, spread, and volatility profiling.
"""
import csv
import math
import sys
from collections import defaultdict

DATA_DIR = "../data/Round 5"
DAYS = [2, 3, 4]

# ═══════════════════════════════════════════════════════════════════
# 1. Load price data → per-product time series of mid prices
# ═══════════════════════════════════════════════════════════════════

def load_prices():
    """Returns {day: {product: [(timestamp, mid, bid1, ask1, bid_vol1, ask_vol1)]}}"""
    data = {}
    for day in DAYS:
        path = f"{DATA_DIR}/prices_round_5_day_{day}.csv"
        day_data = defaultdict(list)
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                product = row["product"]
                ts = int(row["timestamp"])
                mid = float(row["mid_price"]) if row["mid_price"] else None
                bid1 = float(row["bid_price_1"]) if row["bid_price_1"] else None
                ask1 = float(row["ask_price_1"]) if row["ask_price_1"] else None
                bv1 = int(row["bid_volume_1"]) if row["bid_volume_1"] else 0
                av1 = int(row["ask_volume_1"]) if row["ask_volume_1"] else 0
                if mid is not None:
                    day_data[product].append((ts, mid, bid1, ask1, bv1, av1))
        data[day] = dict(day_data)
    return data

def load_trades():
    """Returns {day: {symbol: [(timestamp, price, quantity, buyer, seller)]}}"""
    data = {}
    for day in DAYS:
        path = f"{DATA_DIR}/trades_round_5_day_{day}.csv"
        day_data = defaultdict(list)
        with open(path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                sym = row["symbol"]
                ts = int(row["timestamp"])
                price = float(row["price"])
                qty = int(row["quantity"])
                buyer = row.get("buyer", "")
                seller = row.get("seller", "")
                day_data[sym].append((ts, price, qty, buyer, seller))
        data[day] = dict(day_data)
    return data

# ═══════════════════════════════════════════════════════════════════
# 2. Extract product families
# ═══════════════════════════════════════════════════════════════════

def get_family(product):
    """Group products into families by prefix."""
    prefixes = [
        "GALAXY_SOUNDS", "MICROCHIP", "OXYGEN_SHAKE", "PANEL",
        "PEBBLES", "ROBOT", "SLEEP_POD", "SNACKPACK",
        "TRANSLATOR", "UV_VISOR"
    ]
    for p in prefixes:
        if product.startswith(p):
            return p
    return "UNKNOWN"

# ═══════════════════════════════════════════════════════════════════
# 3. Statistical helpers
# ═══════════════════════════════════════════════════════════════════

def compute_returns(mids):
    """Compute tick-to-tick returns from mid-price series."""
    returns = []
    for i in range(1, len(mids)):
        if mids[i-1] != 0:
            returns.append((mids[i] - mids[i-1]) / mids[i-1])
    return returns

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def stdev(xs):
    if len(xs) < 2:
        return 0
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / (len(xs) - 1))

def correlation(xs, ys):
    """Pearson correlation between two same-length series."""
    n = min(len(xs), len(ys))
    if n < 10:
        return 0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / ((n - 1) * sx * sy)

def autocorrelation(xs, lag=1):
    """Compute autocorrelation at given lag."""
    if len(xs) < lag + 10:
        return 0
    return correlation(xs[:-lag], xs[lag:])

def cross_correlation_lagged(xs, ys, max_lag=10):
    """Compute cross-correlation at lags -max_lag to +max_lag.
    Positive lag means xs leads ys."""
    results = {}
    n = min(len(xs), len(ys))
    if n < 20:
        return results
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = xs[:n - lag]
            b = ys[lag:n]
        else:
            a = xs[-lag:n]
            b = ys[:n + lag]
        results[lag] = correlation(a, b)
    return results

# ═══════════════════════════════════════════════════════════════════
# 4. Main Analysis
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("ROUND 5 — 50 PRODUCT ANALYSIS")
    print("=" * 80)
    
    print("\nLoading data...")
    prices = load_prices()
    trades = load_trades()
    
    all_products = sorted(set(p for day_data in prices.values() for p in day_data))
    families = defaultdict(list)
    for p in all_products:
        families[get_family(p)].append(p)
    
    print(f"\nProducts: {len(all_products)}")
    print(f"Families: {len(families)}")
    for fam, members in sorted(families.items()):
        print(f"  {fam}: {members}")
    
    # ═══════════════════════════════════════════════════════════════
    # A. Per-product statistics (aggregated across days)
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("A. PER-PRODUCT STATISTICS")
    print("=" * 80)
    
    product_stats = {}
    
    for product in all_products:
        all_mids = []
        all_spreads = []
        all_returns = []
        total_drift = 0
        
        for day in DAYS:
            series = prices[day].get(product, [])
            if not series:
                continue
            mids = [s[1] for s in series]
            all_mids.extend(mids)
            
            # Spreads
            for ts, mid, bid, ask, bv, av in series:
                if bid is not None and ask is not None:
                    all_spreads.append(ask - bid)
            
            # Returns
            rets = compute_returns(mids)
            all_returns.extend(rets)
            
            # Day drift (end - start)
            if len(mids) >= 2:
                total_drift += mids[-1] - mids[0]
        
        if not all_mids:
            continue
        
        avg_mid = mean(all_mids)
        avg_spread = mean(all_spreads) if all_spreads else 0
        spread_pct = (avg_spread / avg_mid * 100) if avg_mid else 0
        ret_vol = stdev(all_returns) * 10000 if all_returns else 0  # in bps
        ac1 = autocorrelation(all_returns, 1)
        ac5 = autocorrelation(all_returns, 5)
        total_range = max(all_mids) - min(all_mids)
        range_pct = (total_range / avg_mid * 100) if avg_mid else 0
        
        product_stats[product] = {
            "family": get_family(product),
            "avg_mid": avg_mid,
            "avg_spread": avg_spread,
            "spread_pct": spread_pct,
            "ret_vol_bps": ret_vol,
            "ac1": ac1,
            "ac5": ac5,
            "total_drift": total_drift,
            "drift_pct": (total_drift / avg_mid * 100) if avg_mid else 0,
            "total_range": total_range,
            "range_pct": range_pct,
            "n_ticks": len(all_mids),
        }
    
    # Print sorted by family
    print(f"\n{'Product':<38} {'AvgMid':>8} {'Spread':>7} {'Spr%':>6} {'Vol(bp)':>8} {'AC(1)':>7} {'AC(5)':>7} {'Drift%':>8} {'Range%':>8}")
    print("-" * 115)
    for fam in sorted(families.keys()):
        for p in sorted(families[fam]):
            s = product_stats.get(p)
            if not s:
                continue
            print(f"{p:<38} {s['avg_mid']:>8.1f} {s['avg_spread']:>7.1f} {s['spread_pct']:>5.2f}% {s['ret_vol_bps']:>8.2f} {s['ac1']:>7.3f} {s['ac5']:>7.3f} {s['drift_pct']:>7.2f}% {s['range_pct']:>7.2f}%")
        print()
    
    # ═══════════════════════════════════════════════════════════════
    # B. Family-level aggregates
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("B. FAMILY-LEVEL SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Family':<22} {'#Prod':>5} {'AvgMid':>8} {'AvgSpread':>9} {'AvgVol':>8} {'AvgAC1':>8} {'AvgDrift%':>10} {'AvgRange%':>10}")
    print("-" * 95)
    
    family_stats = {}
    for fam in sorted(families.keys()):
        members_stats = [product_stats[p] for p in families[fam] if p in product_stats]
        if not members_stats:
            continue
        n = len(members_stats)
        avg_mid = mean([s["avg_mid"] for s in members_stats])
        avg_spread = mean([s["avg_spread"] for s in members_stats])
        avg_vol = mean([s["ret_vol_bps"] for s in members_stats])
        avg_ac1 = mean([s["ac1"] for s in members_stats])
        avg_drift = mean([s["drift_pct"] for s in members_stats])
        avg_range = mean([s["range_pct"] for s in members_stats])
        
        family_stats[fam] = {
            "n": n, "avg_mid": avg_mid, "avg_spread": avg_spread,
            "avg_vol": avg_vol, "avg_ac1": avg_ac1, "avg_drift": avg_drift,
            "avg_range": avg_range,
        }
        
        print(f"{fam:<22} {n:>5} {avg_mid:>8.1f} {avg_spread:>9.1f} {avg_vol:>8.2f} {avg_ac1:>8.4f} {avg_drift:>9.2f}% {avg_range:>9.2f}%")
    
    # ═══════════════════════════════════════════════════════════════
    # C. Intra-family correlation matrix
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("C. INTRA-FAMILY CORRELATION (mid-price levels, day 2)")
    print("=" * 80)
    
    for fam in sorted(families.keys()):
        members = sorted(families[fam])
        if len(members) < 2:
            continue
        
        print(f"\n── {fam} ({len(members)} products) ──")
        
        # Build aligned mid-price series for day 2
        series_map = {}
        for p in members:
            day_series = prices[2].get(p, [])
            if day_series:
                series_map[p] = {s[0]: s[1] for s in day_series}
        
        if len(series_map) < 2:
            continue
        
        # Get common timestamps
        all_ts = sorted(set.intersection(*[set(d.keys()) for d in series_map.values()]))
        if len(all_ts) < 20:
            continue
        
        # Build return series
        return_series = {}
        for p in members:
            if p not in series_map:
                continue
            mids = [series_map[p][ts] for ts in all_ts]
            return_series[p] = compute_returns(mids)
        
        # Print correlation matrix
        short_names = {p: p.replace(fam + "_", "") for p in members}
        max_name = max(len(short_names[p]) for p in members if p in return_series)
        
        header = " " * (max_name + 2) + "  ".join(f"{short_names[p]:>{max(6, len(short_names[p]))}}" for p in members if p in return_series)
        print(header)
        
        for p1 in members:
            if p1 not in return_series:
                continue
            row = f"{short_names[p1]:<{max_name + 2}}"
            for p2 in members:
                if p2 not in return_series:
                    continue
                c = correlation(return_series[p1], return_series[p2])
                w = max(6, len(short_names[p2]))
                row += f"{c:>{w}.3f}  "
            print(row)
    
    # ═══════════════════════════════════════════════════════════════
    # D. Cross-family correlation (family-level baskets)
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("D. CROSS-FAMILY CORRELATION (average mid returns, day 2)")
    print("=" * 80)
    
    # Build family-level return series (average of member returns)
    family_returns = {}
    for fam in sorted(families.keys()):
        members = sorted(families[fam])
        member_series = {}
        for p in members:
            day_series = prices[2].get(p, [])
            if day_series:
                member_series[p] = {s[0]: s[1] for s in day_series}
        
        if not member_series:
            continue
        
        common_ts = sorted(set.intersection(*[set(d.keys()) for d in member_series.values()]))
        if len(common_ts) < 20:
            continue
        
        # Average mid across family members at each timestamp
        avg_mids = []
        for ts in common_ts:
            avg_mids.append(mean([member_series[p][ts] for p in member_series]))
        
        family_returns[fam] = compute_returns(avg_mids)
    
    fam_names = sorted(family_returns.keys())
    max_fn = max(len(f) for f in fam_names)
    
    header = " " * (max_fn + 2) + "  ".join(f"{f[:8]:>8}" for f in fam_names)
    print(header)
    for f1 in fam_names:
        row = f"{f1:<{max_fn + 2}}"
        for f2 in fam_names:
            c = correlation(family_returns[f1], family_returns[f2])
            row += f"{c:>8.3f}  "
        print(row)
    
    # ═══════════════════════════════════════════════════════════════
    # E. Lead-lag analysis within highly correlated families
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("E. LEAD-LAG ANALYSIS (intra-family, day 2)")
    print("=" * 80)
    
    for fam in sorted(families.keys()):
        members = sorted(families[fam])
        if len(members) < 2:
            continue
        
        # Build aligned return series
        series_map = {}
        for p in members:
            day_series = prices[2].get(p, [])
            if day_series:
                series_map[p] = {s[0]: s[1] for s in day_series}
        
        if len(series_map) < 2:
            continue
        
        common_ts = sorted(set.intersection(*[set(d.keys()) for d in series_map.values()]))
        if len(common_ts) < 50:
            continue
        
        return_series = {}
        for p in series_map:
            mids = [series_map[p][ts] for ts in common_ts]
            return_series[p] = compute_returns(mids)
        
        print(f"\n── {fam} ──")
        
        # For each pair, find the lag with maximum cross-correlation
        pairs_analyzed = []
        for i, p1 in enumerate(members):
            for p2 in members[i+1:]:
                if p1 not in return_series or p2 not in return_series:
                    continue
                cc = cross_correlation_lagged(return_series[p1], return_series[p2], max_lag=5)
                if not cc:
                    continue
                best_lag = max(cc, key=cc.get)
                best_cc = cc[best_lag]
                lag0_cc = cc.get(0, 0)
                
                short1 = p1.replace(fam + "_", "")
                short2 = p2.replace(fam + "_", "")
                
                if best_lag != 0 and abs(best_cc) > abs(lag0_cc) + 0.02:
                    if best_lag > 0:
                        leader, lagger = short1, short2
                    else:
                        leader, lagger = short2, short1
                    pairs_analyzed.append((abs(best_lag), best_cc, lag0_cc, leader, lagger))
                    print(f"  {leader:>20} LEADS {lagger:<20} by {abs(best_lag)} ticks  (lag-0 corr={lag0_cc:.3f}, best corr={best_cc:.3f})")
        
        if not pairs_analyzed:
            print(f"  No significant lead-lag detected (all pairs peak at lag=0)")
    
    # ═══════════════════════════════════════════════════════════════
    # F. Trade activity analysis
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("F. TRADE ACTIVITY (all days combined)")
    print("=" * 80)
    
    trade_stats = defaultdict(lambda: {"total_volume": 0, "total_trades": 0, "buyers": defaultdict(int), "sellers": defaultdict(int)})
    
    for day in DAYS:
        for sym, trade_list in trades[day].items():
            for ts, price, qty, buyer, seller in trade_list:
                trade_stats[sym]["total_volume"] += qty
                trade_stats[sym]["total_trades"] += 1
                if buyer:
                    trade_stats[sym]["buyers"][buyer] += qty
                if seller:
                    trade_stats[sym]["sellers"][seller] += qty
    
    print(f"\n{'Product':<38} {'Trades':>7} {'Volume':>8} {'Top Buyer':>20} {'Top Seller':>20}")
    print("-" * 100)
    
    for fam in sorted(families.keys()):
        for p in sorted(families[fam]):
            ts_info = trade_stats.get(p, {"total_trades": 0, "total_volume": 0, "buyers": {}, "sellers": {}})
            top_buyer = max(ts_info["buyers"], key=ts_info["buyers"].get) if ts_info["buyers"] else "-"
            top_seller = max(ts_info["sellers"], key=ts_info["sellers"].get) if ts_info["sellers"] else "-"
            print(f"{p:<38} {ts_info['total_trades']:>7} {ts_info['total_volume']:>8} {top_buyer:>20} {top_seller:>20}")
        print()
    
    # ═══════════════════════════════════════════════════════════════
    # G. Intra-family spread (price dispersion)
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("G. INTRA-FAMILY PRICE RELATIONSHIPS (day 2, first & last tick)")
    print("=" * 80)
    
    for fam in sorted(families.keys()):
        members = sorted(families[fam])
        if len(members) < 2:
            continue
        
        print(f"\n── {fam} ──")
        print(f"  {'Product':<38} {'Start Mid':>10} {'End Mid':>10} {'Drift':>10}")
        
        for p in members:
            series = prices[2].get(p, [])
            if not series:
                continue
            start_mid = series[0][1]
            end_mid = series[-1][1]
            drift = end_mid - start_mid
            print(f"  {p:<38} {start_mid:>10.1f} {end_mid:>10.1f} {drift:>+10.1f}")
    
    # ═══════════════════════════════════════════════════════════════
    # H. Panel size analysis (special — dimension-based products)
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("H. PANEL SIZE ANALYSIS — checking if price ∝ area")
    print("=" * 80)
    
    panel_areas = {
        "PANEL_1X2": 2, "PANEL_1X4": 4, "PANEL_2X2": 4,
        "PANEL_2X4": 8, "PANEL_4X4": 16
    }
    
    print(f"\n  {'Panel':<15} {'Area':>5} {'AvgMid':>10} {'Price/Area':>12}")
    for p, area in sorted(panel_areas.items(), key=lambda x: x[1]):
        s = product_stats.get(p)
        if s:
            print(f"  {p:<15} {area:>5} {s['avg_mid']:>10.1f} {s['avg_mid']/area:>12.1f}")
    
    # ═══════════════════════════════════════════════════════════════
    # I. Mean-reversion opportunities (negative AC1)
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("I. MEAN REVERSION RANKING (by AC(1), most negative = strongest MR)")
    print("=" * 80)
    
    ranked = sorted(product_stats.items(), key=lambda x: x[1]["ac1"])
    print(f"\n{'Rank':>4} {'Product':<38} {'AC(1)':>8} {'AC(5)':>8} {'Vol(bp)':>8} {'Spread':>7} {'Family':<20}")
    print("-" * 100)
    for i, (p, s) in enumerate(ranked):
        flag = "★" if s["ac1"] < -0.1 else ""
        print(f"{i+1:>4} {p:<38} {s['ac1']:>8.4f} {s['ac5']:>8.4f} {s['ret_vol_bps']:>8.2f} {s['avg_spread']:>7.1f} {s['family']:<20} {flag}")
    
    # ═══════════════════════════════════════════════════════════════
    # J. Consistency check across days
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("J. DAY-OVER-DAY CONSISTENCY (AC1 per day)")
    print("=" * 80)
    
    print(f"\n{'Product':<38} {'AC1_D2':>8} {'AC1_D3':>8} {'AC1_D4':>8} {'Stable?':>8}")
    print("-" * 80)
    
    for p in sorted(all_products):
        ac_days = []
        for day in DAYS:
            series = prices[day].get(p, [])
            if len(series) < 30:
                ac_days.append(None)
                continue
            mids = [s[1] for s in series]
            rets = compute_returns(mids)
            ac_days.append(autocorrelation(rets, 1))
        
        if all(a is not None for a in ac_days):
            # Stable if all same sign and within 0.1 of each other
            signs = [1 if a > 0 else -1 for a in ac_days]
            same_sign = (signs[0] == signs[1] == signs[2])
            spread_ac = max(ac_days) - min(ac_days)
            stable = "YES" if same_sign and spread_ac < 0.15 else "no"
            print(f"{p:<38} {ac_days[0]:>8.4f} {ac_days[1]:>8.4f} {ac_days[2]:>8.4f} {stable:>8}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
