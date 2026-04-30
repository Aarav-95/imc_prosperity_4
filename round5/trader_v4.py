from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ── SKIP List ────────────────────────────────────────────────────────────────
# Products that lost consistently across BOTH the basic-MM run (+309k) AND
# the microprice run (+218k). These are structurally toxic regardless of
# strategy style — wide spreads, adverse selection, or directional drift that
# overwhelms any passive quoting edge. Skipping them removes ~60k of drag.
SKIP = {
    "ROBOT_MOPPING",         # -17k (v1) / -6k (v2) — consistent bleeder
    "MICROCHIP_TRIANGLE",    # -10k (v1) / -8.6k (v2) — consistent bleeder
    "PANEL_1X2",             # -11k (v1) / -5.3k (v2) — consistent bleeder
    "SLEEP_POD_LAMB_WOOL",   # -8k (v1) / -8.3k (v2) — consistent bleeder
    "PEBBLES_M",             # -7.7k (v1) / -9.6k (v2) — consistent bleeder
}

# ── SNACKPACK Pairs Configuration ────────────────────────────────────────────
# Pairs that are structurally cointegrated across all 3 data days.
# Correlations stable at -0.91 to -0.93 (anticorr) / +0.91 (poscorr).
# "+" = sum is stationary (anticorrelated pair).
# "-" = diff is stationary (positively correlated pair).
SNACKPACK_PAIRS = [
    ("SNACKPACK_CHOCOLATE",  "SNACKPACK_VANILLA",     "+"),
    ("SNACKPACK_RASPBERRY",  "SNACKPACK_STRAWBERRY",  "+"),
    ("SNACKPACK_PISTACHIO",  "SNACKPACK_RASPBERRY",   "+"),
    ("SNACKPACK_PISTACHIO",  "SNACKPACK_STRAWBERRY",  "-"),
]
SNACKPACK_PRODUCTS = {
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
}

# Slow EWMA for pair spread mean (~500-tick lookback).
# Long enough to filter noise, short enough to adapt to per-day mean shifts.
PAIR_ALPHA    = 0.998
PAIR_Z_THRESH = 2.0
PAIR_MAX_LEG  = 5
PAIR_WARMUP   = 500

# ── MM Engine Configuration ───────────────────────────────────────────────────
# Fast EWMA for per-product fair value (~30-tick span).
EWMA_ALPHA   = 0.93
MR_MIN_VAR   = 4.0
MR_K         = 1.5
MR_CAP       = 4
INV_SKEW     = 0.5     # seashells of fair-value shift per unit of inventory


class Trader:
    def __init__(self):
        self.position_limits = {
            "GALAXY_SOUNDS_DARK_MATTER": 10,
            "GALAXY_SOUNDS_BLACK_HOLES": 10,
            "GALAXY_SOUNDS_PLANETARY_RINGS": 10,
            "GALAXY_SOUNDS_SOLAR_WINDS": 10,
            "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
            "SLEEP_POD_SUEDE": 10,
            "SLEEP_POD_LAMB_WOOL": 10,
            "SLEEP_POD_POLYESTER": 10,
            "SLEEP_POD_NYLON": 10,
            "SLEEP_POD_COTTON": 10,
            "MICROCHIP_CIRCLE": 10,
            "MICROCHIP_OVAL": 10,
            "MICROCHIP_SQUARE": 10,
            "MICROCHIP_RECTANGLE": 10,
            "MICROCHIP_TRIANGLE": 10,
            "PEBBLES_XS": 10,
            "PEBBLES_S": 10,
            "PEBBLES_M": 10,
            "PEBBLES_L": 10,
            "PEBBLES_XL": 10,
            "ROBOT_VACUUMING": 10,
            "ROBOT_MOPPING": 10,
            "ROBOT_DISHES": 10,
            "ROBOT_LAUNDRY": 10,
            "ROBOT_IRONING": 10,
            "UV_VISOR_YELLOW": 10,
            "UV_VISOR_AMBER": 10,
            "UV_VISOR_ORANGE": 10,
            "UV_VISOR_RED": 10,
            "UV_VISOR_MAGENTA": 10,
            "TRANSLATOR_SPACE_GRAY": 10,
            "TRANSLATOR_ASTRO_BLACK": 10,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_GRAPHITE_MIST": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "PANEL_1X2": 10,
            "PANEL_2X2": 10,
            "PANEL_1X4": 10,
            "PANEL_2X4": 10,
            "PANEL_4X4": 10,
            "OXYGEN_SHAKE_MORNING_BREATH": 10,
            "OXYGEN_SHAKE_EVENING_BREATH": 10,
            "OXYGEN_SHAKE_MINT": 10,
            "OXYGEN_SHAKE_CHOCOLATE": 10,
            "OXYGEN_SHAKE_GARLIC": 10,
            "SNACKPACK_CHOCOLATE": 10,
            "SNACKPACK_VANILLA": 10,
            "SNACKPACK_PISTACHIO": 10,
            "SNACKPACK_STRAWBERRY": 10,
            "SNACKPACK_RASPBERRY": 10,
        }

        # Only trade products not in the SKIP list
        self.active_products = [
            p for p in self.position_limits if p not in SKIP
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _microprice(self, od: OrderDepth) -> Optional[float]:
        """Volume-weighted fair value between best bid and ask.
        Leans toward the thinner side of the book for a more accurate estimate."""
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bid_vol  = od.buy_orders[best_bid]
        ask_vol  = abs(od.sell_orders[best_ask])
        total    = bid_vol + ask_vol
        if total <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    # ── SNACKPACK Pair Overlay ────────────────────────────────────────────────

    def _compute_pair_targets(
        self, state: TradingState, pair_state: dict, tick: int
    ) -> Dict[str, float]:
        """Compute inventory target adjustments from cointegrated SNACKPACK pairs.
        Returns {product: target_pos_delta}. Only fires after PAIR_WARMUP ticks."""
        targets: Dict[str, float] = {p: 0.0 for p in SNACKPACK_PRODUCTS}
        limit = 10

        for a, b, op in SNACKPACK_PAIRS:
            label = f"{a}_{b}"
            od_a  = state.order_depths.get(a)
            od_b  = state.order_depths.get(b)
            if od_a is None or od_b is None:
                continue
            mid_a = self._microprice(od_a)
            mid_b = self._microprice(od_b)
            if mid_a is None or mid_b is None:
                continue

            spread = (mid_a + mid_b) if op == "+" else (mid_a - mid_b)

            ps      = pair_state.get(label, {"m": spread, "ad": 0.0})
            new_m   = PAIR_ALPHA * ps["m"]  + (1 - PAIR_ALPHA) * spread
            dev     = abs(spread - new_m)
            new_ad  = PAIR_ALPHA * ps["ad"] + (1 - PAIR_ALPHA) * dev
            pair_state[label] = {"m": new_m, "ad": new_ad}

            if tick < PAIR_WARMUP or new_ad < 5.0:
                continue

            z = (spread - new_m) / new_ad
            if abs(z) < PAIR_Z_THRESH:
                continue

            leg  = min(PAIR_MAX_LEG, int(abs(z) - PAIR_Z_THRESH + 1) * 2)
            sign = -1 if z > 0 else +1  # spread too high → short, too low → long

            if op == "+":
                targets[a] += sign * leg
                targets[b] += sign * leg
            else:
                targets[a] += sign * leg
                targets[b] -= sign * leg

        for p in targets:
            targets[p] = max(-limit, min(limit, targets[p]))

        return targets

    # ── Core MM Engine ────────────────────────────────────────────────────────

    def _quote_product(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        limit: int,
        ewma_state: dict,
        target_pos: float,
    ) -> List[Order]:
        """Microprice + EWMA mean-reversion + inventory skew market maker.

        1. Compute microprice as volume-weighted fair value.
        2. Update EWMA mean/variance for a short-horizon mean-reversion signal.
        3. Nudge target_pos against local z-score deviation.
        4. Skew fair value away from inventory (anchored at target_pos).
        5. Half-spread = max(1, book_spread/4) — widens on wide books.
        6. Take any level priced better than skewed_fair ± half immediately.
        7. Quote residual capacity passively around skewed_fair ± half.
        """
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid    = max(od.buy_orders.keys())
        best_ask    = min(od.sell_orders.keys())
        book_spread = best_ask - best_bid
        if book_spread <= 0:
            return []

        fair = self._microprice(od)
        if fair is None:
            return []

        # Update EWMA mean + variance
        prev   = ewma_state.get(product)
        if prev is None:
            ewma_m, ewma_v = fair, 0.0
        else:
            ewma_m = EWMA_ALPHA * prev["m"] + (1 - EWMA_ALPHA) * fair
            ewma_v = EWMA_ALPHA * prev["v"] + (1 - EWMA_ALPHA) * (fair - ewma_m) ** 2
        ewma_state[product] = {"m": ewma_m, "v": ewma_v}

        # Mean-reversion target nudge
        if ewma_v > MR_MIN_VAR:
            z      = (fair - ewma_m) / math.sqrt(ewma_v)
            mr_adj = max(-MR_CAP, min(MR_CAP, -MR_K * z))
        else:
            mr_adj = 0.0

        dyn_target  = max(-limit, min(limit, target_pos + mr_adj))

        # Inventory skew anchored at dynamic target
        skewed_fair = fair - INV_SKEW * (pos - dyn_target)

        half    = max(1.0, book_spread / 4.0)
        our_bid = math.floor(skewed_fair - half)
        our_ask = math.ceil(skewed_fair  + half)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap  = limit - pos
        sell_cap = limit + pos

        # --- True Buy & Hold / Short & Hold ---
        if target_pos == limit:
            if buy_cap > 0:
                # Aggressively take the ask to build position, and never sell
                return [Order(product, best_ask + 1, buy_cap)]
            return []

        if target_pos == -limit:
            if sell_cap > 0:
                # Aggressively hit the bid to build position, and never buy
                return [Order(product, best_bid - 1, -sell_cap)]
            return []
        # --------------------------------------

        orders: List[Order] = []

        # Take any mispriced liquidity immediately
        ask_taken = 0
        if best_ask <= skewed_fair - half and buy_cap > 0:
            avail     = abs(od.sell_orders[best_ask])
            ask_taken = min(avail, buy_cap)
            if ask_taken > 0:
                orders.append(Order(product, best_ask, ask_taken))

        bid_taken = 0
        if best_bid >= skewed_fair + half and sell_cap > 0:
            avail     = od.buy_orders[best_bid]
            bid_taken = min(avail, sell_cap)
            if bid_taken > 0:
                orders.append(Order(product, best_bid, -bid_taken))

        # Quote residual capacity passively
        quote_buy  = max(0, buy_cap  - ask_taken)
        quote_sell = max(0, sell_cap - bid_taken)

        if quote_buy  > 0:
            orders.append(Order(product, int(our_bid),  quote_buy))
        if quote_sell > 0:
            orders.append(Order(product, int(our_ask), -quote_sell))

        return orders

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        ewma_state: dict = trader_data.get("ewma", {})
        pair_state: dict = trader_data.get("pairs", {})
        tick: int        = trader_data.get("tick", 0) + 1

        # Phase 1: compute SNACKPACK pair targets
        snack_targets = self._compute_pair_targets(state, pair_state, tick)

        # Phase 2: quote every non-SKIP product with the MM engine
        for product in self.active_products:
            od = state.order_depths.get(product)
            if not od:
                continue

            pos   = state.position.get(product, 0)
            limit = self.position_limits[product]

            # SNACKPACK: pair overlay drives target; others are pure MM (target=0)
            target_pos = snack_targets.get(product, 0.0)

            # Directional Trend Overrides
            if product == "MICROCHIP_OVAL":
                target_pos = -limit  # Max Short
            elif product == "GALAXY_SOUNDS_BLACK_HOLES":
                target_pos = limit # Max Long
            elif product == "PEBBLES_XL":
                target_pos = limit   # Max Long
            elif product == "PEBBLES_XS":
                target_pos = -limit  # Max Short
            elif product == "OXYGEN_SHAKE_GARLIC":
                target_pos = limit   # Max Long

            orders = self._quote_product(
                product, od, pos, limit, ewma_state, target_pos
            )
            if orders:
                result[product] = orders

        traderData = json.dumps(
            {"ewma": ewma_state, "pairs": pair_state, "tick": tick},
            separators=(",", ":"),
        )
        return result, conversions, traderData
