from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math

# ── SKIP List ────────────────────────────────────────────────────────────────
SKIP = {
    "ROBOT_MOPPING",
    "MICROCHIP_TRIANGLE",
    "PANEL_1X2",
    "SLEEP_POD_LAMB_WOOL",
    "PEBBLES_M",
}

# Products where penny_flatten beats v4 MM - use that strategy instead
PENNY_FLATTEN_PRODUCTS = {
    "PANEL_1X4",                      # 1m: +8805, v4: -594
    "PEBBLES_L",                      # 1m: +5597, v4: -3521  
    "OXYGEN_SHAKE_MORNING_BREATH",    # 1m: +7175, v4: -1320
    "GALAXY_SOUNDS_PLANETARY_RINGS",  # 1m: +6776, v4: +3046
}

# ── SNACKPACK Pairs Configuration ────────────────────────────────────────────
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

PAIR_ALPHA    = 0.998
PAIR_Z_THRESH = 2.0
PAIR_MAX_LEG  = 5
PAIR_WARMUP   = 500

# ── MM Engine Configuration ───────────────────────────────────────────────────
EWMA_ALPHA   = 0.93
MR_MIN_VAR   = 4.0
MR_K         = 1.5
MR_CAP       = 4
INV_SKEW     = 0.5


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

        self.active_products = [
            p for p in self.position_limits if p not in SKIP and p not in PENNY_FLATTEN_PRODUCTS
        ]

    def _microprice(self, od: OrderDepth) -> Optional[float]:
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

    def _compute_pair_targets(
        self, state: TradingState, pair_state: dict, tick: int
    ) -> Dict[str, float]:
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
            sign = -1 if z > 0 else +1

            if op == "+":
                targets[a] += sign * leg
                targets[b] += sign * leg
            else:
                targets[a] += sign * leg
                targets[b] -= sign * leg

        for p in targets:
            targets[p] = max(-limit, min(limit, targets[p]))

        return targets

    def _quote_product(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        limit: int,
        ewma_state: dict,
        target_pos: float,
    ) -> List[Order]:
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

        prev   = ewma_state.get(product)
        if prev is None:
            ewma_m, ewma_v = fair, 0.0
        else:
            ewma_m = EWMA_ALPHA * prev["m"] + (1 - EWMA_ALPHA) * fair
            ewma_v = EWMA_ALPHA * prev["v"] + (1 - EWMA_ALPHA) * (fair - ewma_m) ** 2
        ewma_state[product] = {"m": ewma_m, "v": ewma_v}

        if ewma_v > MR_MIN_VAR:
            z      = (fair - ewma_m) / math.sqrt(ewma_v)
            mr_adj = max(-MR_CAP, min(MR_CAP, -MR_K * z))
        else:
            mr_adj = 0.0

        dyn_target  = max(-limit, min(limit, target_pos + mr_adj))
        skewed_fair = fair - INV_SKEW * (pos - dyn_target)

        half    = max(1.0, book_spread / 4.0)
        our_bid = math.floor(skewed_fair - half)
        our_ask = math.ceil(skewed_fair  + half)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap  = limit - pos
        sell_cap = limit + pos

        if target_pos == limit:
            if buy_cap > 0:
                return [Order(product, best_ask + 1, buy_cap)]
            return []

        if target_pos == -limit:
            if sell_cap > 0:
                return [Order(product, best_bid - 1, -sell_cap)]
            return []

        orders: List[Order] = []

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

        quote_buy  = max(0, buy_cap  - ask_taken)
        quote_sell = max(0, sell_cap - bid_taken)

        if quote_buy  > 0:
            orders.append(Order(product, int(our_bid),  quote_buy))
        if quote_sell > 0:
            orders.append(Order(product, int(our_ask), -quote_sell))

        return orders

    def _penny_flatten(self, product: str, od: OrderDepth, pos: int, limit: int) -> List[Order]:
        """Penny-and-flatten strategy from trader_1m - works better for some products."""
        if not od.buy_orders or not od.sell_orders:
            return []

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        inner_bid = best_bid + 1
        inner_ask = best_ask - 1
        fair = (inner_bid + inner_ask) / 2.0
        edge = (inner_ask - inner_bid) / 2.0

        orders: List[Order] = []

        # TAKE: sweep mispriced levels
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair - edge:
                break
            avail = -od.sell_orders[ask_price]
            qty = min(avail, limit - pos)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                pos += qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair + edge:
                break
            avail = od.buy_orders[bid_price]
            qty = min(avail, limit + pos)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                pos -= qty

        # CLEAR: flatten inventory at fair
        fair_int = int(round(fair))
        if pos > 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0
        elif pos < 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0

        # MAKE: balanced quotes at penny-improved prices
        buy_cap = limit - pos
        sell_cap = limit + pos
        skip_bid = pos > 0 and inner_bid >= best_bid
        skip_ask = pos < 0 and inner_ask <= best_ask

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(product, inner_bid, buy_cap))
        if sell_cap > 0 and not skip_ask:
            orders.append(Order(product, inner_ask, -sell_cap))

        return orders

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

        snack_targets = self._compute_pair_targets(state, pair_state, tick)

        # Penny flatten products
        for product in PENNY_FLATTEN_PRODUCTS:
            od = state.order_depths.get(product)
            if not od:
                continue
            pos = state.position.get(product, 0)
            limit = self.position_limits[product]
            orders = self._penny_flatten(product, od, pos, limit)
            if orders:
                result[product] = orders

        # V4 MM products
        for product in self.active_products:
            od = state.order_depths.get(product)
            if not od:
                continue

            pos   = state.position.get(product, 0)
            limit = self.position_limits[product]

            target_pos = snack_targets.get(product, 0.0)

            if product == "MICROCHIP_OVAL":
                target_pos = -limit
            elif product == "GALAXY_SOUNDS_BLACK_HOLES":
                target_pos = limit
            elif product == "PEBBLES_XL":
                target_pos = limit
            elif product == "PEBBLES_XS":
                target_pos = -limit
            elif product == "OXYGEN_SHAKE_GARLIC":
                target_pos = limit

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
