"""
Aether Crystal options pricing & portfolio optimization.

Underlying: GBM with zero risk-neutral drift, sigma = 251% annualized.
Discrete grid: 4 steps per trading day, 252 trading days per year.
S0 = 50 (mid of 49.975/50.025).
2W = 40 steps, 3W = 60 steps.

Requirements: numpy, scipy
    pip install numpy scipy

Runtime: ~1-2 minutes total on a normal laptop.
"""

import numpy as np
from scipy.stats import norm

# ============================================================================
# CONSTANTS
# ============================================================================
S0 = 50.0
SIGMA = 2.51                         # 251% annualized vol
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY  # 1008
DT = 1.0 / STEPS_PER_YEAR

STEPS_2W = 2 * 5 * STEPS_PER_DAY     # 40
STEPS_3W = 3 * 5 * STEPS_PER_DAY     # 60

CONTRACT_SIZE = 3000                 # PnL multiplier
N_EVAL = 100                         # number of sims used to mark PnL


# ============================================================================
# PATH SIMULATION (with antithetic variates)
# ============================================================================
def simulate_paths(n_half, n_steps, seed=None):
    """
    Simulate GBM paths with antithetic variates for variance reduction.
    Returns array of shape (2*n_half, n_steps+1).
    """
    if seed is not None:
        np.random.seed(seed)
    z = np.random.standard_normal((n_half, n_steps))
    z_full = np.concatenate([z, -z], axis=0)
    log_inc = -0.5 * SIGMA**2 * DT + SIGMA * np.sqrt(DT) * z_full
    log_S = np.log(S0) + np.concatenate(
        [np.zeros((2 * n_half, 1)), np.cumsum(log_inc, axis=1)], axis=1
    )
    return np.exp(log_S)


# ============================================================================
# PRICING: compute fair values for all instruments
# ============================================================================
print("=" * 90)
print("STEP 1: PRICING ALL INSTRUMENTS")
print("=" * 90)

N_HALF = 500_000   # 1M total paths
print(f"Simulating {2*N_HALF:,} paths over {STEPS_3W} steps...")
paths = simulate_paths(N_HALF, STEPS_3W, seed=7)

S_2W = paths[:, STEPS_2W]
S_3W = paths[:, STEPS_3W]
S_min_3W = paths[:, 1:STEPS_3W + 1].min(axis=1)  # min over discrete grid

# Build payoff arrays for all instruments
payoffs = {
    '50C_3W':   np.maximum(S_3W - 50, 0),
    '50P_3W':   np.maximum(50 - S_3W, 0),
    '35P_3W':   np.maximum(35 - S_3W, 0),
    '40P_3W':   np.maximum(40 - S_3W, 0),
    '45P_3W':   np.maximum(45 - S_3W, 0),
    '60C_3W':   np.maximum(S_3W - 60, 0),
    '50P_2W':   np.maximum(50 - S_2W, 0),
    '50C_2W':   np.maximum(S_2W - 50, 0),
    # Chooser: at t=2W, holder picks side that's ITM at that time (per spec)
    'CHOOSER':  np.where(S_2W > 50,
                         np.maximum(S_3W - 50, 0),
                         np.maximum(50 - S_3W, 0)),
    # Binary put: pays 10 if S_3W < 40, else 0
    'BIN_P':    np.where(S_3W < 40, 10.0, 0.0),
    # Knock-out put: K=45, barrier=35. Knocked out if S ever < 35.
    'KO_P':     np.where(S_min_3W >= 35, np.maximum(45 - S_3W, 0), 0.0),
}

# Market quotes (bid, ask)
quotes = {
    '50C_3W':   (12.00, 12.05),
    '50P_3W':   (12.00, 12.05),
    '35P_3W':   ( 4.33,  4.35),
    '40P_3W':   ( 6.50,  6.55),
    '45P_3W':   ( 9.05,  9.10),
    '60C_3W':   ( 8.80,  8.85),
    '50P_2W':   ( 9.70,  9.75),
    '50C_2W':   ( 9.70,  9.75),
    'CHOOSER':  (22.20, 22.30),
    'BIN_P':    ( 5.00,  5.10),
    'KO_P':     ( 0.15,  0.175),
}

# Available volume per instrument
volumes = {
    '50C_3W': 50, '50P_3W': 50, '35P_3W': 50, '40P_3W': 50, '45P_3W': 50,
    '60C_3W': 50, '50P_2W': 50, '50C_2W': 50, 'CHOOSER': 50, 'BIN_P': 50,
    'KO_P': 500,
}

names = list(payoffs.keys())
fair = {n: payoffs[n].mean() for n in names}
sd = {n: payoffs[n].std(ddof=1) for n in names}

def stderr(x):
    return x.std(ddof=1) / np.sqrt(len(x))

print(f"\n{'Instrument':<10} {'Fair':>9} {'StdErr':>8} {'Bid':>7} {'Ask':>7} "
      f"{'BuyEdge':>9} {'SellEdge':>9} {'Edge%':>7} {'Action':>6}")
print("-" * 90)
for n in names:
    f = fair[n]
    se = stderr(payoffs[n])
    bid, ask = quotes[n]
    buy_edge = f - ask
    sell_edge = bid - f
    best = max(buy_edge, sell_edge)
    action = "BUY" if buy_edge > sell_edge else "SELL"
    if best <= 0:
        action = "skip"
    mid = (bid + ask) / 2
    edge_pct = best / mid * 100
    print(f"{n:<10} {f:>9.4f} {se:>8.4f} {bid:>7.3f} {ask:>7.3f} "
          f"{buy_edge:>+9.4f} {sell_edge:>+9.4f} {edge_pct:>+6.2f}% {action:>6}")

# Show key probabilities for context
print(f"\nKey probabilities under the GBM model:")
print(f"  P(S_3W < 35) = {(S_3W < 35).mean():.4f}")
print(f"  P(S_3W < 40) = {(S_3W < 40).mean():.4f}    -> binary put pays out")
print(f"  P(S_3W < 45) = {(S_3W < 45).mean():.4f}")
print(f"  P(S_3W < 50) = {(S_3W < 50).mean():.4f}")
print(f"  P(S_3W > 60) = {(S_3W > 60).mean():.4f}")
print(f"  P(min_3W < 35) = {(S_min_3W < 35).mean():.4f}    -> knock-out probability")


# ============================================================================
# CORRELATION STRUCTURE (for hedging)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 2: CORRELATION MATRIX OF PAYOFFS")
print("=" * 90)

M = np.column_stack([payoffs[n] for n in names])
C = np.corrcoef(M, rowvar=False)
print(f"{'':>10} " + " ".join(f"{n:>8}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>10} " + " ".join(f"{C[i,j]:>+8.3f}" for j in range(len(names))))

# Covariance of MARKS (= Cov(payoff) / N_EVAL since marks are 100-sim averages)
Cov_payoff = np.cov(M, rowvar=False)
Cov_mark = Cov_payoff / N_EVAL


# ============================================================================
# PORTFOLIO ANALYSIS HELPERS
# ============================================================================
bid_arr = np.array([quotes[n][0] for n in names])
ask_arr = np.array([quotes[n][1] for n in names])
vol_arr = np.array([volumes[n] for n in names])
fair_arr = np.array([fair[n] for n in names])

def portfolio_stats(qty):
    """
    Given signed qty vector (long > 0, short < 0), return (E[PnL], std[PnL]).
    Long entries fill at ask, short entries fill at bid.
    """
    q = np.array(qty)
    entry = np.where(q > 0, ask_arr, np.where(q < 0, bid_arr, 0.0))
    e_pnl = (q * (fair_arr - entry) * CONTRACT_SIZE).sum()
    w = q * CONTRACT_SIZE
    var = w @ Cov_mark @ w
    return e_pnl, np.sqrt(var) if var > 0 else 0.0


def coord_descent(lam, max_iter=50):
    """
    Maximize E[PnL] - lambda * std[PnL] over integer positions.
    """
    qty = np.zeros(len(names), dtype=int)
    def obj(q):
        e, s = portfolio_stats(q)
        return e - lam * s
    cur = obj(qty)
    for _ in range(max_iter):
        improved = False
        for i in range(len(names)):
            best, best_obj = qty[i], cur
            for q_i in range(-vol_arr[i], vol_arr[i] + 1):
                qty_test = qty.copy()
                qty_test[i] = q_i
                v = obj(qty_test)
                if v > best_obj + 1e-6:
                    best_obj, best = v, q_i
            if best != qty[i]:
                qty[i], cur = best, best_obj
                improved = True
        if not improved:
            break
    return qty


def run_mc_portfolio(qty, n_trials=20_000, n_eval=100, seed=2026):
    """
    Run full Monte Carlo: each trial draws 100 paths, computes mark for each
    instrument as the 100-path average payoff, then computes total portfolio PnL.
    Returns array of n_trials PnL realizations.
    """
    np.random.seed(seed)
    pnls = np.empty(n_trials)
    for t in range(n_trials):
        z = np.random.standard_normal((n_eval, STEPS_3W))
        log_inc = -0.5 * SIGMA**2 * DT + SIGMA * np.sqrt(DT) * z
        log_S = np.log(S0) + np.concatenate(
            [np.zeros((n_eval, 1)), np.cumsum(log_inc, axis=1)], axis=1
        )
        Sp = np.exp(log_S)
        s2 = Sp[:, STEPS_2W]
        s3 = Sp[:, STEPS_3W]
        smn = Sp[:, 1:STEPS_3W + 1].min(axis=1)
        marks = np.array([
            np.maximum(s3 - 50, 0).mean(),                    # 50C_3W
            np.maximum(50 - s3, 0).mean(),                    # 50P_3W
            np.maximum(35 - s3, 0).mean(),                    # 35P_3W
            np.maximum(40 - s3, 0).mean(),                    # 40P_3W
            np.maximum(45 - s3, 0).mean(),                    # 45P_3W
            np.maximum(s3 - 60, 0).mean(),                    # 60C_3W
            np.maximum(50 - s2, 0).mean(),                    # 50P_2W
            np.maximum(s2 - 50, 0).mean(),                    # 50C_2W
            np.where(s2 > 50, np.maximum(s3 - 50, 0),
                              np.maximum(50 - s3, 0)).mean(), # CHOOSER
            np.where(s3 < 40, 10.0, 0.0).mean(),              # BIN_P
            np.where(smn >= 35, np.maximum(45 - s3, 0),
                                0.0).mean(),                  # KO_P
        ])
        entry = np.where(qty > 0, ask_arr, np.where(qty < 0, bid_arr, 0.0))
        pnls[t] = (qty * (marks - entry) * CONTRACT_SIZE).sum()
    return pnls


# ============================================================================
# OPTIMIZER: lambda sweep to find the risk-return frontier
# ============================================================================
print("\n" + "=" * 90)
print("STEP 3: LAMBDA SWEEP (E[PnL] - lambda * std[PnL])")
print("=" * 90)
print(f"{'lambda':>7} {'E[PnL]':>10} {'std':>10} {'Sharpe':>7} {'P(loss)*':>9}  positions")
print("-" * 90)

lam_results = {}
for lam in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]:
    q = coord_descent(lam)
    e, s = portfolio_stats(q)
    sh = e / s if s > 0 else 0.0
    p_loss = norm.cdf(-e / s) if s > 0 else 0.0
    lam_results[lam] = q
    pos_str = " ".join(f"{names[i]}={q[i]:+d}" for i in range(len(names)) if q[i] != 0)
    print(f"{lam:>7.2f} {e:>10.0f} {s:>10.0f} {sh:>7.3f} {p_loss:>9.4f}  {pos_str}")


# ============================================================================
# FINAL CHOICE: lambda = 0.3 portfolio (good balance)
# ============================================================================
print("\n" + "=" * 90)
print("STEP 4: FINAL PORTFOLIO (lambda = 0.3)")
print("=" * 90)

final_qty = lam_results[0.3]
e_anal, s_anal = portfolio_stats(final_qty)

print(f"{'Instrument':<10} {'Action':<6} {'Qty':>5} {'Entry':>7} "
      f"{'Fair':>7} {'Edge/u':>9} {'$ Contrib':>11}")
print("-" * 70)
total = 0
for i, n in enumerate(names):
    q = final_qty[i]
    if q == 0:
        continue
    bid, ask = quotes[n]
    if q > 0:
        action, entry = "BUY", ask
        edge = fair[n] - ask
    else:
        action, entry = "SELL", bid
        edge = bid - fair[n]
    contrib = abs(q) * edge * CONTRACT_SIZE
    total += contrib
    print(f"{n:<10} {action:<6} {abs(q):>5d} {entry:>7.3f} "
          f"{fair[n]:>7.3f} {edge:>+9.4f} {contrib:>+11,.0f}")
print("-" * 70)
print(f"{'TOTAL E[PnL] (analytical)':<48} {total:>+15,.0f}")
print(f"Analytical std[PnL]:        {s_anal:,.0f}    Sharpe: {e_anal/s_anal:.3f}")


# ============================================================================
# FINAL VERIFICATION via large-scale Monte Carlo
# ============================================================================
print("\n" + "=" * 90)
print("STEP 5: MONTE CARLO VERIFICATION (50,000 trials)")
print("=" * 90)

pnls = run_mc_portfolio(final_qty, n_trials=50_000)

print(f"\nFinal portfolio statistics:")
print(f"  Mean PnL:        ${pnls.mean():>12,.0f}")
print(f"  Std PnL:         ${pnls.std():>12,.0f}")
print(f"  Sharpe:          {pnls.mean()/pnls.std():.3f}")
print(f"  P(PnL > 0):      {(pnls > 0).mean():.3%}")
print(f"  P(PnL > 50k):    {(pnls > 50_000).mean():.3%}")
print(f"  P(PnL > 100k):   {(pnls > 100_000).mean():.3%}")
print(f"  P(PnL < -100k):  {(pnls < -100_000).mean():.3%}")
print(f"  P(PnL < -200k):  {(pnls < -200_000).mean():.3%}")
print(f"\n  Percentiles:")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{q:>2}:  ${np.percentile(pnls, q):>12,.0f}")

# Also compare to the un-hedged max-edge portfolio for context
unhedged_qty = np.zeros(len(names), dtype=int)
unhedged_qty[names.index('KO_P')]    = 500
unhedged_qty[names.index('50P_2W')]  = 50
unhedged_qty[names.index('50C_2W')]  = 50
unhedged_qty[names.index('CHOOSER')] = -50
unhedged_qty[names.index('BIN_P')]   = -50

pnls_unhedged = run_mc_portfolio(unhedged_qty, n_trials=50_000, seed=2027)
print(f"\nFor comparison, un-hedged max-edge portfolio:")
print(f"  Mean: ${pnls_unhedged.mean():,.0f}  Std: ${pnls_unhedged.std():,.0f}  "
      f"Sharpe: {pnls_unhedged.mean()/pnls_unhedged.std():.3f}")
print(f"  P(PnL < -200k): {(pnls_unhedged < -200_000).mean():.3%}  "
      f"(vs {(pnls < -200_000).mean():.3%} hedged)")
print(f"  p5: ${np.percentile(pnls_unhedged, 5):,.0f}  "
      f"(vs ${np.percentile(pnls, 5):,.0f} hedged)")
