from __future__ import annotations
from datamodel import OrderDepth, TradingState, Order
import json
import math

LIMIT = 10

# EWMA mean-reversion overlay (~30-tick span)
EWMA_ALPHA = 0.93
MR_K       = 1.5
MR_CAP     = 4
MR_MIN_VAR = 4.0

# SNACKPACK pair-trade slow EWMA (~500-tick span)
PAIR_ALPHA    = 0.998
PAIR_Z_THRESH = 2.0
PAIR_MAX_LEG  = 5
PAIR_WARMUP   = 500

# op="+" -> spread = a + b (sum-stationary, anticorrelated pair)
# op="-" -> spread = a - b (diff-stationary, positively correlated pair)
SNACKPACK_PAIRS = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",    "+"),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY",  "+"),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",   "+"),
    ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY",  "-"),
]
SNACKPACK_PRODUCTS = {
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",
    "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY",
    "SNACKPACK_PISTACHIO",
}

# PEBBLES basket: microprice sum of all 5 ~= 50000 (implied fair per leg)
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

# Per-product config:
#   target_pos : directional inventory bias (0 = pure MM, ±5 = lean into drift)
#                SNACKPACK products: overridden by pair-trade logic
#   inv_skew   : fair shift per (position - target) / LIMIT seashells
#   size       : lots per quote side
#   min_half   : minimum half-spread to post
CFG: dict[str, dict] = {
    # Tight-spread products — higher size, lower skew
    "TRANSLATOR_ECLIPSE_CHARCOAL":   {"target_pos": +5, "inv_skew":  4, "size": 6, "min_half": 2},
    "TRANSLATOR_ASTRO_BLACK":        {"target_pos": +5, "inv_skew":  4, "size": 6, "min_half": 2},
    "ROBOT_LAUNDRY":                 {"target_pos": -5, "inv_skew":  4, "size": 6, "min_half": 2},
    "SNACKPACK_PISTACHIO":           {"target_pos":  0, "inv_skew":  6, "size": 6, "min_half": 4},
    "SNACKPACK_RASPBERRY":           {"target_pos":  0, "inv_skew":  6, "size": 6, "min_half": 4},
    "SNACKPACK_STRAWBERRY":          {"target_pos":  0, "inv_skew":  6, "size": 6, "min_half": 4},

    # Directional bias tier — target ±5, keep at half-limit so both sides quote
    "PANEL_1X2":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 2},
    "UV_VISOR_AMBER":                {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 2},
    "PEBBLES_M":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SLEEP_POD_SUEDE":               {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "MICROCHIP_RECTANGLE":           {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "GALAXY_SOUNDS_SOLAR_FLAMES":    {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "TRANSLATOR_GRAPHITE_MIST":      {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 2},
    "PANEL_4X4":                     {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 2},
    "PEBBLES_XL":                    {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 4},
    "GALAXY_SOUNDS_DARK_MATTER":     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PANEL_2X2":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 2},
    "ROBOT_IRONING":                 {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "ROBOT_VACUUMING":               {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 2},
    "UV_VISOR_RED":                  {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 2},
    "UV_VISOR_ORANGE":               {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 2},
    "UV_VISOR_YELLOW":               {"target_pos": -3, "inv_skew": 12, "size": 3, "min_half": 2},
    "GALAXY_SOUNDS_PLANETARY_RINGS": {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PANEL_1X4":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 2},

    # Previously skipped products — now with drift-matched target_pos
    "SLEEP_POD_COTTON":              {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PEBBLES_S":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "MICROCHIP_OVAL":                {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "UV_VISOR_MAGENTA":              {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PANEL_2X4":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "ROBOT_MOPPING":                 {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "MICROCHIP_TRIANGLE":            {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "TRANSLATOR_SPACE_GRAY":         {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SLEEP_POD_LAMB_WOOL":           {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "MICROCHIP_SQUARE":              {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "OXYGEN_SHAKE_GARLIC":           {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "OXYGEN_SHAKE_MINT":             {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "OXYGEN_SHAKE_MORNING_BREATH":   {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PEBBLES_XS":                    {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "TRANSLATOR_VOID_BLUE":          {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SLEEP_POD_NYLON":               {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "MICROCHIP_CIRCLE":              {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SNACKPACK_CHOCOLATE":           {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SNACKPACK_VANILLA":             {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "OXYGEN_SHAKE_EVENING_BREATH":   {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "SLEEP_POD_POLYESTER":           {"target_pos": +5, "inv_skew": 15, "size": 4, "min_half": 3},
    "GALAXY_SOUNDS_BLACK_HOLES":     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "PEBBLES_L":                     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
    "GALAXY_SOUNDS_SOLAR_WINDS":     {"target_pos": -5, "inv_skew": 15, "size": 4, "min_half": 3},
}


def microprice(od: OrderDepth) -> float | None:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    bv = od.buy_orders[bb]
    av = abs(od.sell_orders[ba])
    total = bv + av
    if total <= 0:
        return (bb + ba) / 2.0
    return (bb * av + ba * bv) / total


class Trader:
    def run(self, state: TradingState):
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        ewma_state: dict[str, dict] = ts.get("ewma", {})
        pair_state: dict[str, dict] = ts.get("pairs", {})
        tick = ts.get("tick", 0) + 1

        orders: dict[str, list[Order]] = {}
        pos = state.position

        # --- SNACKPACK pair-trade signals ---
        # Slow EWMA tracks spread mean + absolute deviation (sigma proxy).
        # z > PAIR_Z_THRESH => spread stretched high => fade it back.
        snack_target: dict[str, float] = {p: 0.0 for p in SNACKPACK_PRODUCTS}
        pair_active = tick >= PAIR_WARMUP

        for a, b, op in SNACKPACK_PAIRS:
            label = f"{a}_{b}_{op}"
            od_a = state.order_depths.get(a)
            od_b = state.order_depths.get(b)
            if od_a is None or od_b is None:
                continue
            m_a = microprice(od_a)
            m_b = microprice(od_b)
            if m_a is None or m_b is None:
                continue
            spread = (m_a + m_b) if op == "+" else (m_a - m_b)
            ps = pair_state.get(label, {"m": spread, "ad": 0.0})
            new_m  = PAIR_ALPHA * ps["m"]  + (1 - PAIR_ALPHA) * spread
            new_ad = PAIR_ALPHA * ps["ad"] + (1 - PAIR_ALPHA) * abs(spread - new_m)
            pair_state[label] = {"m": new_m, "ad": new_ad}

            if not pair_active or new_ad < 5.0:
                continue
            z = (spread - new_m) / new_ad
            if abs(z) < PAIR_Z_THRESH:
                continue
            leg  = min(PAIR_MAX_LEG, int(abs(z) - PAIR_Z_THRESH + 1) * 2)
            sign = -1 if z > 0 else +1   # z high => spread too wide => fade
            if op == "+":
                snack_target[a] += sign * leg
                snack_target[b] += sign * leg
            else:
                snack_target[a] += sign * leg
                snack_target[b] -= sign * leg

        for p in snack_target:
            snack_target[p] = max(-LIMIT, min(LIMIT, snack_target[p]))

        # --- PEBBLES basket-implied fair values ---
        peb_micro = {p: microprice(state.order_depths[p])
                     for p in PEBBLES if p in state.order_depths}
        peb_fair: dict[str, float] = {}
        if len(peb_micro) == 5 and all(v is not None for v in peb_micro.values()):
            for leg in PEBBLES:
                peb_fair[leg] = 50000 - sum(peb_micro[o] for o in PEBBLES if o != leg)

        # --- Main MM loop ---
        for sym, cfg in CFG.items():
            od = state.order_depths.get(sym)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            # Basket-implied fair for PEBBLES when all 5 legs are live
            if sym in peb_fair:
                fair = peb_fair[sym]
            else:
                fair = microprice(od)
                if fair is None:
                    continue

            bb = max(od.buy_orders)
            ba = min(od.sell_orders)
            book_spread = ba - bb
            if book_spread <= 0:
                continue

            lpos = pos.get(sym, 0)

            # SNACKPACK: pair-trade logic sets target; CFG target_pos is fallback
            if sym in SNACKPACK_PRODUCTS:
                base_target = int(round(snack_target.get(sym, 0)))
            else:
                base_target = cfg["target_pos"]

            # EWMA mean-reversion overlay: lean target against local deviation
            prev = ewma_state.get(sym)
            if prev is None:
                ewma_m = fair
                ewma_v = 0.0
            else:
                ewma_m = EWMA_ALPHA * prev["m"] + (1 - EWMA_ALPHA) * fair
                ewma_v = EWMA_ALPHA * prev["v"] + (1 - EWMA_ALPHA) * (fair - ewma_m) ** 2
            ewma_state[sym] = {"m": ewma_m, "v": ewma_v}

            mr_adj = 0.0
            if ewma_v > MR_MIN_VAR:
                z = (fair - ewma_m) / math.sqrt(ewma_v)
                mr_adj = max(-MR_CAP, min(MR_CAP, -MR_K * z))

            target = max(-LIMIT, min(LIMIT, base_target + mr_adj))

            # Inventory skew: shift fair so quotes lean toward closing gap to target
            inv_shift   = cfg["inv_skew"] * (lpos - target) / LIMIT
            skewed_fair = fair - inv_shift

            half    = max(cfg["min_half"], book_spread / 4.0)
            bid_px  = math.floor(skewed_fair - half)
            ask_px  = math.ceil(skewed_fair + half)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            buy_cap  = LIMIT - lpos
            sell_cap = LIMIT + lpos
            buy_qty  = min(cfg["size"], max(0, buy_cap))
            sell_qty = min(cfg["size"], max(0, sell_cap))

            ords: list[Order] = []

            # Take-the-cross: eat resting orders priced through our skewed fair
            ask_take = 0
            if ba <= skewed_fair - half and buy_qty > 0:
                avail    = abs(od.sell_orders[ba])
                ask_take = min(avail, buy_qty)
                if ask_take > 0:
                    ords.append(Order(sym, ba, ask_take))

            bid_take = 0
            if bb >= skewed_fair + half and sell_qty > 0:
                avail    = od.buy_orders[bb]
                bid_take = min(avail, sell_qty)
                if bid_take > 0:
                    ords.append(Order(sym, bb, -bid_take))

            # Passive quotes for remaining capacity
            q_buy  = max(0, buy_qty  - ask_take)
            q_sell = max(0, sell_qty - bid_take)
            if q_buy  > 0:
                ords.append(Order(sym, bid_px,  q_buy))
            if q_sell > 0:
                ords.append(Order(sym, ask_px, -q_sell))

            if ords:
                orders[sym] = ords

        new_td = json.dumps(
            {"ewma": ewma_state, "pairs": pair_state, "tick": tick},
            separators=(",", ":"),
        )
        return orders, 0, new_td
