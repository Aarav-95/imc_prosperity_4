from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ── SNACKPACK Pairs Configuration ──────────────────────────────────────────
# Pairs that are structurally cointegrated across all 3 data days.
# "+" = anticorrelated pair (trade the SUM): sum is stationary.
# "-" = positively correlated pair (trade the DIFF): diff is stationary.
# Correlations are stable at -0.91 to -0.93 / +0.91 across days 2/3/4.
SNACKPACK_PAIRS = [
    ("SNACKPACK_CHOCOLATE",  "SNACKPACK_VANILLA",     "+", "choc_van"),
    ("SNACKPACK_RASPBERRY",  "SNACKPACK_STRAWBERRY",  "+", "ras_str"),
    ("SNACKPACK_PISTACHIO",  "SNACKPACK_RASPBERRY",   "+", "pis_ras"),
    ("SNACKPACK_PISTACHIO",  "SNACKPACK_STRAWBERRY",  "-", "pis_str"),
]
SNACKPACK_PRODUCTS = {
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
}

# Slow EWMA for the pair spread mean (alpha=0.998 ≈ 500-tick lookback).
# Long enough to filter noise, short enough to adapt to per-day mean shifts.
PAIR_ALPHA    = 0.998
PAIR_Z_THRESH = 2.0      # z-score threshold to enter a pair trade
PAIR_MAX_LEG  = 5        # max lots per product per pair
PAIR_WARMUP   = 500      # ticks before pair trading activates

# ── MM Engine Configuration ─────────────────────────────────────────────────
# EWMA for per-product fair value (alpha=0.93 ≈ 30-tick fast lookback).
EWMA_ALPHA   = 0.93
MR_MIN_VAR   = 4.0       # min variance required before mean-reversion fires
MR_K         = 1.5       # how strongly to fade z-score deviations
MR_CAP       = 4         # max inventory shift from mean-reversion overlay
INV_SKEW     = 0.5       # seashells of fair-value shift per unit of inventory


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

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _microprice(self, od: OrderDepth) -> Optional[float]:
        """Volume-weighted fair value between best bid and ask.
        Leans toward whichever side has less depth, giving a more accurate
        estimate of where the next trade will print."""
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

    # ── SNACKPACK Pair Overlay ───────────────────────────────────────────────

    def _compute_pair_targets(
        self,
        state: TradingState,
        pair_state: dict,
        tick: int,
    ) -> Dict[str, float]:
        """Compute target inventory adjustments from cointegrated SNACKPACK pairs.

        Returns a dict {product: target_pos_delta} derived from z-score signals.
        All contributions across pairs are summed and clamped to ±POS_LIMIT.
        Only fires after PAIR_WARMUP ticks so slow EWMA has time to settle.
        """
        targets: Dict[str, float] = {p: 0.0 for p in SNACKPACK_PRODUCTS}

        for a, b, op, label in SNACKPACK_PAIRS:
            od_a = state.order_depths.get(a)
            od_b = state.order_depths.get(b)
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

            # Need warmup AND meaningful sigma before trading
            if tick < PAIR_WARMUP or new_ad < 5.0:
                continue

            z = (spread - new_m) / new_ad
            if abs(z) < PAIR_Z_THRESH:
                continue

            # Scale lot size with z-score distance beyond threshold, cap at max
            leg  = min(PAIR_MAX_LEG, int(abs(z) - PAIR_Z_THRESH + 1) * 2)
            sign = -1 if z > 0 else +1   # spread too high → short, too low → long

            if op == "+":
                targets[a] += sign * leg
                targets[b] += sign * leg
            else:                         # "−" = diff pair
                targets[a] += sign * leg
                targets[b] -= sign * leg

        limit = 10
        for p in targets:
            targets[p] = max(-limit, min(limit, targets[p]))

        return targets

    # ── Core MM Engine ───────────────────────────────────────────────────────

    def _quote_product(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        limit: int,
        ewma_state: dict,
        target_pos: float,
    ) -> List[Order]:
        """Full microprice + EWMA-mean-reversion + inventory-skew MM logic.

        Steps:
          1. Compute microprice as the volume-weighted fair value.
          2. Update EWMA mean and variance for a short-horizon mean-reversion signal.
          3. Derive a z-score and nudge target_pos against local deviations.
          4. Skew fair value away from current inventory (anchored at target_pos).
          5. Set half-spread = max(1, book_spread/4) so quotes widen on wide books.
          6. Take any book levels priced better than our skewed fair ± half.
          7. Quote the residual capacity passively around skewed fair ± half.
        """
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid   = max(od.buy_orders.keys())
        best_ask   = min(od.sell_orders.keys())
        book_spread = best_ask - best_bid
        if book_spread <= 0:
            return []

        fair = self._microprice(od)
        if fair is None:
            return []

        # ── Step 2: update EWMA mean + variance ──────────────────────────
        prev      = ewma_state.get(product)
        if prev is None:
            ewma_m, ewma_v = fair, 0.0
        else:
            ewma_m = EWMA_ALPHA * prev["m"] + (1 - EWMA_ALPHA) * fair
            ewma_v = EWMA_ALPHA * prev["v"] + (1 - EWMA_ALPHA) * (fair - ewma_m) ** 2
        ewma_state[product] = {"m": ewma_m, "v": ewma_v}

        # ── Step 3: mean-reversion target nudge ──────────────────────────
        if ewma_v > MR_MIN_VAR:
            z      = (fair - ewma_m) / math.sqrt(ewma_v)
            mr_adj = max(-MR_CAP, min(MR_CAP, -MR_K * z))
        else:
            mr_adj = 0.0

        dyn_target = max(-limit, min(limit, target_pos + mr_adj))

        # ── Step 4: inventory skew anchored at dynamic target ─────────────
        deviation   = pos - dyn_target
        skewed_fair = fair - INV_SKEW * deviation

        # ── Step 5: half-spread ───────────────────────────────────────────
        half = max(1.0, book_spread / 4.0)

        our_bid = math.floor(skewed_fair - half)
        our_ask = math.ceil(skewed_fair  + half)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap  = limit - pos
        sell_cap = limit + pos

        orders: List[Order] = []

        # ── Step 6: take any mispriced liquidity immediately ──────────────
        ask_taken = 0
        if best_ask <= skewed_fair - half and buy_cap > 0:
            avail      = abs(od.sell_orders[best_ask])
            ask_taken  = min(avail, buy_cap)
            if ask_taken > 0:
                orders.append(Order(product, best_ask, ask_taken))

        bid_taken = 0
        if best_bid >= skewed_fair + half and sell_cap > 0:
            avail      = od.buy_orders[best_bid]
            bid_taken  = min(avail, sell_cap)
            if bid_taken > 0:
                orders.append(Order(product, best_bid, -bid_taken))

        # ── Step 7: quote residual capacity passively ─────────────────────
        quote_buy  = max(0, buy_cap  - ask_taken)
        quote_sell = max(0, sell_cap - bid_taken)

        if quote_buy > 0:
            orders.append(Order(product, int(our_bid), quote_buy))
        if quote_sell > 0:
            orders.append(Order(product, int(our_ask), -quote_sell))

        return orders

    # ── Run ──────────────────────────────────────────────────────────────────

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

        # ── Phase 1: compute SNACKPACK pair targets ───────────────────────
        snack_targets = self._compute_pair_targets(state, pair_state, tick)

        # ── Phase 2: quote every product with the MM engine ──────────────
        for product, limit in self.position_limits.items():
            od = state.order_depths.get(product)
            if not od:
                continue

            pos = state.position.get(product, 0)

            # For SNACKPACK products the pair overlay drives target position;
            # for everything else target_pos = 0 (pure spread capture, no bias).
            target_pos = snack_targets.get(product, 0.0)

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
