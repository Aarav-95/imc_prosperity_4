"""
Manual trading challenge portfolio optimizer.

Fee formula: fee_i = (x_i)^2 * budget  (where x_i is fraction of budget allocated)
Net profit = budget * sum(r_i * x_i - x_i^2)

We allow long (+) and short (-) positions.
Budget constraint: sum(|x_i|) <= 1
"""

import numpy as np
import cvxpy as cp

BUDGET = 1_000_000

products = [
    "scoria_paste",
    "ashes_of_phoenix",
    "lava_cake",
    "obsidian",
    "magma_ink",
    "pyroflex",
    "thermolite",
    "sulfur",
    "volcanic_incense",
]

# Mid-point estimates from news analysis
returns = np.array([
    0.00,   # scoria paste
    -0.025,  # ashes of phoenix (2-3% → use 2.5%)
   -0.65,   # lava cake
    0.20,   # obsidian
    0.00,   # magma ink
   -0.10,   # pyroflex
    0.25,   # thermolite
    0.05,   # sulfur
    0.02,   # volcanic incense
])

n = len(products)

# Decision variable: fraction of budget per product (can be negative = short)
x = cp.Variable(n)

# Objective: maximize net return after fees
# net = sum(r_i * x_i - x_i^2) * budget
# cvxpy minimizes, so negate
fee_cost = cp.sum_squares(x)           # sum of x_i^2
expected_return = returns @ x           # sum of r_i * x_i
objective = cp.Maximize(expected_return - fee_cost)

# Budget constraint: total capital deployed <= 1 (100%)
# Using L1 norm: sum(|x_i|) <= 1
constraints = [cp.norm1(x) <= 1.0]

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"Status: {prob.status}")
print(f"\n{'Product':<22} {'Allocation':>12} {'Exp Return':>12} {'Fee':>12} {'Net':>12}")
print("-" * 72)

total_fee = 0
total_net = 0
total_budget_used = 0

for i, prod in enumerate(products):
    xi = x.value[i]
    alloc_pct = xi * 100
    rev = returns[i] * xi * BUDGET
    fee = xi**2 * BUDGET
    net = rev - fee
    total_fee += fee
    total_net += net
    total_budget_used += abs(xi)
    if abs(xi) > 0.0001:
        direction = "BUY " if xi > 0 else "SELL"
        print(f"{prod:<22} {alloc_pct:>+10.2f}% ({direction:5}) {rev:>+10,.0f}  {fee:>10,.0f}  {net:>+10,.0f}")

print("-" * 72)
print(f"{'TOTAL':<22} {total_budget_used*100:>10.2f}%        {total_net+total_fee:>+10,.0f}  {total_fee:>10,.0f}  {total_net:>+10,.0f}")
print(f"\nNet profit: ${total_net:,.0f}")
print(f"Total fees: ${total_fee:,.0f}")
print(f"Budget used: {total_budget_used*100:.1f}%")
print(f"Objective value: {prob.value:.6f}")

# Sanity check: unconstrained per-product optima
print("\n--- Unconstrained optima (x_i* = r_i/2) ---")
x_unc = returns / 2
total_unc_budget = np.sum(np.abs(x_unc))
net_unc = np.sum(returns * x_unc - x_unc**2) * BUDGET
print(f"Total budget needed: {total_unc_budget*100:.1f}%")
print(f"Net profit if feasible: ${net_unc:,.0f}")
