"""
Implied Volatility / Moneyness Analysis for VEV_* Vouchers

Backs out IV from each voucher using Black-Scholes, then maps IV vs moneyness
to find the volatility smile and identify mispricings.
"""
import csv
import math
from collections import defaultdict

# ── Black-Scholes helpers (stdlib only) ─────────────────────────────

def norm_cdf(x):
    """Standard normal CDF using Abramowitz & Stegun approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2)
    return 0.5 * (1.0 + sign * y)

def bs_call_price(S, K, T, sigma, r=0.0):
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_delta(S, K, T, sigma, r=0.0):
    """Black-Scholes delta for a call."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_vega(S, K, T, sigma, r=0.0):
    """Black-Scholes vega (dC/dσ)."""
    if T <= 0 or sigma <= 0:
        return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)

def implied_vol(C_market, S, K, T, r=0.0, tol=1e-6, max_iter=100):
    """
    Solve for implied volatility using Newton-Raphson with bisection fallback.
    Returns None if IV cannot be determined (e.g. price below intrinsic).
    """
    intrinsic = max(0, S - K * math.exp(-r * T))
    if C_market < intrinsic - 0.01:
        return None  # below intrinsic, no valid IV
    if C_market >= S:
        return None  # above the upper bound

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * C_market / S if T > 0 else 0.5
    sigma = max(0.01, min(sigma, 5.0))

    # Newton-Raphson
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, sigma, r)
        v = bs_vega(S, K, T, sigma, r)
        if v < 1e-10:
            break
        diff = price - C_market
        sigma -= diff / v
        sigma = max(0.001, min(sigma, 10.0))
        if abs(diff) < tol:
            return sigma

    # Fallback: bisection
    lo, hi = 0.001, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        price = bs_call_price(S, K, T, mid, r)
        if price < C_market:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid

    return (lo + hi) / 2


# ── Load data ──────────────────────────────────────────────────────

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

all_prices = []
for day_idx in [0, 1, 2]:
    all_prices.extend(read_csv(f"data/Round 3/prices_round_3_day_{day_idx}.csv"))

# Products
underlying_name = "VELVETFRUIT_EXTRACT"
vouchers = ["VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100",
            "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500",
            "VEV_6000", "VEV_6500"]
strikes = {v: int(v.split("_")[1]) for v in vouchers}

# Build time-indexed data: (day, timestamp) -> mid_price
price_by_ts = defaultdict(dict)  # (day, ts) -> {product: mid}
for r in all_prices:
    ts = to_float(r["timestamp"])
    day = r["day"]
    mid = to_float(r["mid_price"])
    product = r["product"]
    if ts is not None and mid is not None:
        key = (day, int(ts))
        price_by_ts[key][product] = mid


# ══════════════════════════════════════════════════════════════════
# TTE ESTIMATION
# ══════════════════════════════════════════════════════════════════

# Each day has ~10,000 ticks (timestamps 0 to 999,900 step 100)
# We need to figure out TTE. Let's try a range and find what gives
# the smoothest IV smile.

TICKS_PER_DAY = 10_000

print("=" * 70)
print("  TTE ESTIMATION")
print("=" * 70)

# Try different TTE assumptions and see which gives most consistent IV
# Using VEV_5300 (near ATM) as the calibration anchor
for total_days in [3, 4, 5, 7, 10]:
    # On day 0, tick 0: TTE = total_days (in day units)
    # We'll compute IV in annualized terms (multiply daily vol by sqrt(252))
    sample_key = ("0", 0)
    if sample_key in price_by_ts:
        S = price_by_ts[sample_key].get(underlying_name)
        for v in ["VEV_5200", "VEV_5300"]:
            C = price_by_ts[sample_key].get(v)
            K = strikes[v]
            if S and C:
                T_years = total_days / 252
                iv = implied_vol(C, S, K, T_years)
                if iv:
                    daily_vol_pct = iv / math.sqrt(252) * 100
                    print(f"  TTE={total_days}d  {v}: IV={iv*100:.1f}% ann, daily_vol={daily_vol_pct:.2f}%")

# Check actual daily vol from underlying data
print(f"\n  Actual underlying daily returns analysis:")
for day_label in ["0", "1", "2"]:
    day_mids = []
    for ts in range(0, 1000000, 100):
        key = (day_label, ts)
        if key in price_by_ts and underlying_name in price_by_ts[key]:
            day_mids.append(price_by_ts[key][underlying_name])
    if len(day_mids) > 1:
        rets = [(day_mids[i] - day_mids[i-1]) / day_mids[i-1] for i in range(1, len(day_mids))]
        daily_std = math.sqrt(sum(r**2 for r in rets) / len(rets)) * math.sqrt(TICKS_PER_DAY)
        ann_vol = daily_std * math.sqrt(252)
        print(f"  Day {day_label}: realized_daily_vol={daily_std*100:.2f}%, annualized={ann_vol*100:.1f}%")


# ══════════════════════════════════════════════════════════════════
# IV SURFACE ANALYSIS
# ══════════════════════════════════════════════════════════════════

# Use TTE = 4 days for now (we'll refine based on calibration above)
TOTAL_DAYS_TO_EXPIRY = 5  # Total days from day 0 tick 0 to expiry

print("\n" + "=" * 70)
print(f"  IV SURFACE (TTE base = {TOTAL_DAYS_TO_EXPIRY} days)")
print("=" * 70)

# Compute IV for each voucher across all timestamps
iv_data = []  # list of (day, ts, voucher, S, K, C, moneyness, IV, delta)

for key in sorted(price_by_ts.keys()):
    day, ts = key
    day_int = int(day)
    prices = price_by_ts[key]
    S = prices.get(underlying_name)
    if S is None:
        continue

    # TTE decreases as we move forward in time
    elapsed_days = day_int + ts / (TICKS_PER_DAY * 100)
    tte_days = max(0.01, TOTAL_DAYS_TO_EXPIRY - elapsed_days)
    T_years = tte_days / 252

    for v in vouchers:
        C = prices.get(v)
        K = strikes[v]
        if C is None or C <= 0:
            continue

        moneyness = math.log(S / K) if S > 0 and K > 0 else 0
        iv = implied_vol(C, S, K, T_years)

        if iv is not None and 0.001 < iv < 10.0:
            delta = bs_delta(S, K, T_years, iv)
            iv_data.append({
                "day": day, "ts": ts, "voucher": v,
                "S": S, "K": K, "C": C,
                "moneyness": moneyness,
                "iv": iv, "delta": delta,
                "tte_days": tte_days
            })

print(f"\n  Total IV data points: {len(iv_data)}")

# ── Per-voucher IV summary ─────────────────────────────────────────
print(f"\n--- Average IV by Voucher ---")
print(f"  {'Voucher':<12} {'Strike':>6} {'Avg IV%':>8} {'Std IV%':>8} {'Min IV%':>8} {'Max IV%':>8} {'Avg Delta':>10} {'Moneyness':>10} {'N':>6}")

voucher_ivs = defaultdict(list)
for d in iv_data:
    voucher_ivs[d["voucher"]].append(d)

for v in vouchers:
    if v in voucher_ivs:
        ivs = [d["iv"] for d in voucher_ivs[v]]
        deltas = [d["delta"] for d in voucher_ivs[v]]
        moneyness = [d["moneyness"] for d in voucher_ivs[v]]
        mean_iv = sum(ivs) / len(ivs)
        std_iv = math.sqrt(sum((x - mean_iv)**2 for x in ivs) / len(ivs)) if len(ivs) > 1 else 0
        mean_delta = sum(deltas) / len(deltas)
        mean_m = sum(moneyness) / len(moneyness)
        print(f"  {v:<12} {strikes[v]:>6} {mean_iv*100:>8.2f} {std_iv*100:>8.2f} {min(ivs)*100:>8.2f} {max(ivs)*100:>8.2f} {mean_delta:>10.4f} {mean_m:>10.4f} {len(ivs):>6}")
    else:
        print(f"  {v:<12} {strikes[v]:>6}  -- NO IV DATA --")


# ── IV vs Moneyness (the smile) at specific timestamps ─────────────
print(f"\n--- Volatility Smile Snapshots ---")
print(f"  (IV at fixed timestamps across all strikes)")

sample_timestamps = [
    ("0", 0), ("0", 500000),
    ("1", 0), ("1", 500000),
    ("2", 0), ("2", 500000)
]

for sample_day, sample_ts in sample_timestamps:
    key = (sample_day, sample_ts)
    if key not in price_by_ts or underlying_name not in price_by_ts[key]:
        continue

    S = price_by_ts[key][underlying_name]
    day_int = int(sample_day)
    elapsed_days = day_int + sample_ts / (TICKS_PER_DAY * 100)
    tte_days = max(0.01, TOTAL_DAYS_TO_EXPIRY - elapsed_days)
    T_years = tte_days / 252

    print(f"\n  Day {sample_day}, t={sample_ts} (S={S:.1f}, TTE={tte_days:.2f}d)")
    print(f"  {'Voucher':<12} {'K':>6} {'Price':>8} {'Intrinsic':>10} {'TV':>8} {'IV%':>8} {'Delta':>8} {'Moneyness':>10}")

    for v in vouchers:
        C = price_by_ts[key].get(v)
        K = strikes[v]
        if C is None:
            continue
        intrinsic = max(0, S - K)
        tv = C - intrinsic
        moneyness = math.log(S / K)
        iv = implied_vol(C, S, K, T_years)
        delta = bs_delta(S, K, T_years, iv) if iv else None

        iv_str = f"{iv*100:.2f}" if iv else "N/A"
        delta_str = f"{delta:.4f}" if delta else "N/A"
        print(f"  {v:<12} {K:>6} {C:>8.2f} {intrinsic:>10.2f} {tv:>8.2f} {iv_str:>8} {delta_str:>8} {moneyness:>10.4f}")


# ── Per-day IV evolution for key strikes ────────────────────────────
print(f"\n--- IV Evolution Over Time (sampled every 1000 ticks) ---")
key_vouchers = ["VEV_5000", "VEV_5200", "VEV_5300", "VEV_5400"]

for v in key_vouchers:
    print(f"\n  {v} (K={strikes[v]}):")
    print(f"  {'Day':>4} {'Tick':>8} {'S':>8} {'C':>8} {'IV%':>8} {'Delta':>8}")
    for day_label in ["0", "1", "2"]:
        for ts in range(0, 1000000, 100000):  # sample every ~1000 ticks
            key = (day_label, ts)
            if key not in price_by_ts:
                continue
            S = price_by_ts[key].get(underlying_name)
            C = price_by_ts[key].get(v)
            K = strikes[v]
            if S is None or C is None:
                continue
            day_int = int(day_label)
            elapsed_days = day_int + ts / (TICKS_PER_DAY * 100)
            tte_days = max(0.01, TOTAL_DAYS_TO_EXPIRY - elapsed_days)
            T_years = tte_days / 252
            iv = implied_vol(C, S, K, T_years)
            delta = bs_delta(S, K, T_years, iv) if iv else None
            iv_str = f"{iv*100:.2f}" if iv else "N/A"
            delta_str = f"{delta:.4f}" if delta else "N/A"
            print(f"  {day_label:>4} {ts:>8} {S:>8.1f} {C:>8.2f} {iv_str:>8} {delta_str:>8}")


# ── Find IV outliers ───────────────────────────────────────────────
print(f"\n" + "=" * 70)
print(f"  OUTLIER DETECTION: IV DEVIATIONS FROM SMILE")
print(f"=" * 70)

# For each timestamp, fit a simple model to the IV smile and find deviations
# Group by timestamp, compute mean/std of IVs, flag outliers

# First, compute per-timestamp smile structure
ts_smiles = defaultdict(dict)  # (day, ts) -> {voucher: iv}
for d in iv_data:
    ts_smiles[(d["day"], d["ts"])][d["voucher"]] = d["iv"]

# For each timestamp, compute the mean IV and find which vouchers deviate
outlier_counts = defaultdict(lambda: {"above": 0, "below": 0, "total": 0})
for key, smile in ts_smiles.items():
    if len(smile) < 3:
        continue
    ivs = list(smile.values())
    mean_iv = sum(ivs) / len(ivs)
    std_iv = math.sqrt(sum((x - mean_iv)**2 for x in ivs) / len(ivs))
    if std_iv < 0.001:
        continue
    for v, iv in smile.items():
        z = (iv - mean_iv) / std_iv
        outlier_counts[v]["total"] += 1
        if z > 1.5:
            outlier_counts[v]["above"] += 1
        elif z < -1.5:
            outlier_counts[v]["below"] += 1

print(f"\n  How often each voucher's IV is an outlier (|z| > 1.5 from smile mean):")
print(f"  {'Voucher':<12} {'IV High %':>10} {'IV Low %':>10} {'Signal':>20}")
for v in vouchers:
    if outlier_counts[v]["total"] > 0:
        n = outlier_counts[v]["total"]
        pct_high = outlier_counts[v]["above"] / n * 100
        pct_low = outlier_counts[v]["below"] / n * 100
        signal = ""
        if pct_high > 30:
            signal = "→ SELL VOL (overpriced)"
        elif pct_low > 30:
            signal = "→ BUY VOL (underpriced)"
        print(f"  {v:<12} {pct_high:>9.1f}% {pct_low:>9.1f}% {signal:>20}")


# ── Pairwise IV comparison for nearby strikes ─────────────────────
print(f"\n--- IV Spread Between Adjacent Strikes ---")
adjacent_pairs = [
    ("VEV_5000", "VEV_5100"), ("VEV_5100", "VEV_5200"),
    ("VEV_5200", "VEV_5300"), ("VEV_5300", "VEV_5400"),
    ("VEV_5400", "VEV_5500"),
]

for v1, v2 in adjacent_pairs:
    spreads = []
    for key, smile in ts_smiles.items():
        if v1 in smile and v2 in smile:
            spreads.append(smile[v1] - smile[v2])
    if spreads:
        mean_sp = sum(spreads) / len(spreads)
        std_sp = math.sqrt(sum((x - mean_sp)**2 for x in spreads) / len(spreads))
        print(f"  {v1} - {v2}: mean_iv_diff={mean_sp*100:+.2f}%, std={std_sp*100:.2f}%")


# ── BS Fair Value vs Market Price ──────────────────────────────────
print(f"\n" + "=" * 70)
print(f"  BS FAIR VALUE VS MARKET (using median IV)")
print(f"=" * 70)
print(f"  If we use the median IV across all strikes as 'true' vol,")
print(f"  which vouchers are over/under priced?")

# Compute median IV across all data
all_ivs = [d["iv"] for d in iv_data]
all_ivs_sorted = sorted(all_ivs)
median_iv = all_ivs_sorted[len(all_ivs_sorted) // 2]
print(f"\n  Median IV across all strikes: {median_iv*100:.2f}%")

# For the most recent timestamp, compute BS fair value using median IV
last_key = max(price_by_ts.keys())
S = price_by_ts[last_key].get(underlying_name)
day_int = int(last_key[0])
ts = last_key[1]
elapsed_days = day_int + ts / (TICKS_PER_DAY * 100)
tte_days = max(0.01, TOTAL_DAYS_TO_EXPIRY - elapsed_days)
T_years = tte_days / 252

print(f"  Last timestamp: Day {last_key[0]}, t={last_key[1]} (S={S:.1f}, TTE={tte_days:.2f}d)")
print(f"\n  {'Voucher':<12} {'Market':>8} {'BS Fair':>8} {'Diff':>8} {'Signal':>20}")

for v in vouchers:
    C_market = price_by_ts[last_key].get(v)
    K = strikes[v]
    if C_market is None:
        continue
    bs_fair = bs_call_price(S, K, T_years, median_iv)
    diff = C_market - bs_fair
    signal = ""
    if diff > 2:
        signal = "OVERPRICED → SELL"
    elif diff < -2:
        signal = "UNDERPRICED → BUY"
    print(f"  {v:<12} {C_market:>8.2f} {bs_fair:>8.2f} {diff:>+8.2f} {signal:>20}")


print(f"\n\nDone!")
