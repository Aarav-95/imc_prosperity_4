import json
import math
from typing import Any, List
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ══════════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS
# ══════════════════════════════════════════════════════════════════

def norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2)
    return 0.5 * (1.0 + sign * y)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)


def implied_vol(C_market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(0, S - K * math.exp(-r * T))
    if C_market < intrinsic - 0.5:
        return 0.01
    if C_market >= S:
        return 5.0

    sigma = max(0.01, min(math.sqrt(2 * math.pi / max(T, 0.001)) * C_market / S, 5.0))

    for _ in range(50):
        price = bs_call_price(S, K, T, sigma, r)
        v = bs_vega(S, K, T, sigma, r)
        if v < 1e-10:
            break
        sigma -= (price - C_market) / v
        sigma = max(0.001, min(sigma, 10.0))
        if abs(price - C_market) < 0.01:
            return sigma

    lo, hi = 0.001, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_call_price(S, K, T, mid, r) < C_market:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.0001:
            break
    return (lo + hi) / 2


class Trader:

    def __init__(self):
        self.position_limits = {
            "GALAXY_SOUNDS_BLACK_HOLES": 10, "GALAXY_SOUNDS_DARK_MATTER": 10,
            "GALAXY_SOUNDS_PLANETARY_RINGS": 10, "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
            "GALAXY_SOUNDS_SOLAR_WINDS": 10, "MICROCHIP_CIRCLE": 10,
            "MICROCHIP_OVAL": 10, "MICROCHIP_RECTANGLE": 10,
            "MICROCHIP_SQUARE": 10, "MICROCHIP_TRIANGLE": 10,
            "OXYGEN_SHAKE_CHOCOLATE": 10, "OXYGEN_SHAKE_EVENING_BREATH": 10,
            "OXYGEN_SHAKE_GARLIC": 10, "OXYGEN_SHAKE_MINT": 10,
            "OXYGEN_SHAKE_MORNING_BREATH": 10, "PANEL_1X2": 10,
            "PANEL_1X4": 10, "PANEL_2X2": 10, "PANEL_2X4": 10,
            "PANEL_4X4": 10, "PEBBLES_L": 10, "PEBBLES_M": 10,
            "PEBBLES_S": 10, "PEBBLES_XL": 10, "PEBBLES_XS": 10,
            "ROBOT_DISHES": 10, "ROBOT_IRONING": 10, "ROBOT_LAUNDRY": 10,
            "ROBOT_MOPPING": 10, "ROBOT_VACUUMING": 10, "SLEEP_POD_COTTON": 10,
            "SLEEP_POD_LAMB_WOOL": 10, "SLEEP_POD_NYLON": 10,
            "SLEEP_POD_POLYESTER": 10, "SLEEP_POD_SUEDE": 10,
            "SNACKPACK_CHOCOLATE": 10, "SNACKPACK_PISTACHIO": 10,
            "SNACKPACK_RASPBERRY": 10, "SNACKPACK_STRAWBERRY": 10,
            "SNACKPACK_VANILLA": 10, "TRANSLATOR_ASTRO_BLACK": 10,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10, "TRANSLATOR_GRAPHITE_MIST": 10,
            "TRANSLATOR_SPACE_GRAY": 10, "TRANSLATOR_VOID_BLUE": 10,
            "UV_VISOR_AMBER": 10, "UV_VISOR_MAGENTA": 10,
            "UV_VISOR_ORANGE": 10, "UV_VISOR_RED": 10, "UV_VISOR_YELLOW": 10,
        }

        # ── Toxic Assets for the Adverse Quoter ────────────────────
        self.SKIP = {
            "MICROCHIP_SQUARE", "PEBBLES_M", "PEBBLES_XS", "ROBOT_VACUUMING",
            "SLEEP_POD_COTTON", "SLEEP_POD_SUEDE", "TRANSLATOR_ECLIPSE_CHARCOAL",
            "UV_VISOR_MAGENTA", "UV_VISOR_YELLOW", "GALAXY_SOUNDS_DARK_MATTER",
            "GALAXY_SOUNDS_SOLAR_FLAMES", "PANEL_4X4", "ROBOT_MOPPING",
            "SLEEP_POD_POLYESTER", "SNACKPACK_STRAWBERRY", "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_SPACE_GRAY", "PANEL_2X2", "ROBOT_DISHES", "ROBOT_LAUNDRY",
            "UV_VISOR_ORANGE"
        }

        self.ALL_PRODUCTS = [p for p in self.position_limits if p not in self.SKIP]

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    def _get_mid(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def _take(self, od: OrderDepth, fair: float, edge: float, product: str, pos: int, limit: int) -> tuple:
        orders: List[Order] = []
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair - edge: break
            avail = -od.sell_orders[ask_price]
            qty = min(avail, limit - pos)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                pos += qty

        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair + edge: break
            avail = od.buy_orders[bid_price]
            qty = min(avail, limit + pos)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                pos -= qty

        return orders, pos

    def _clear(self, fair: float, product: str, pos: int, limit: int) -> tuple:
        orders: List[Order] = []
        fair_int = int(round(fair))
        if pos > 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0
        elif pos < 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0
        return orders, pos

    def _make_balanced(self, bid_price: int, ask_price: int, product: str, pos: int, limit: int, best_bid: int, best_ask: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap = limit - pos
        sell_cap = limit + pos
        skip_bid = pos > 0 and bid_price >= best_bid
        skip_ask = pos < 0 and ask_price <= best_ask

        if buy_cap > 0 and not skip_bid: orders.append(Order(product, bid_price, buy_cap))
        if sell_cap > 0 and not skip_ask: orders.append(Order(product, ask_price, -sell_cap))

        return orders

    # ══════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        trader_data = {}
        if state.traderData:
            try: trader_data = json.loads(state.traderData)
            except: pass

        last_ts = trader_data.get("last_ts", -1)
        if state.timestamp < last_ts:
            trader_data["day"] = trader_data.get("day", 0) + 1
        trader_data["last_ts"] = state.timestamp

        ewma_state = trader_data.setdefault("ewma", {})

        # ── Phase 1: Penny-and-flatten on SAFE products ───────────
        self.penny_flatten(state, result)

        # ── Phase 2: v4 MM logic on TOXIC (SKIP) products ─────────
        self.v4_maker(state, result, ewma_state)

        # Serialize persistent state
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    # ══════════════════════════════════════════════════════════════
    # STRATEGY 1: PENNY FLATTEN (Safe Assets)
    # ══════════════════════════════════════════════════════════════

    def penny_flatten(self, state: TradingState, result: dict):
        for product in self.ALL_PRODUCTS:
            od = state.order_depths.get(product)
            if not od or not od.buy_orders or not od.sell_orders: continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            book_spread = best_ask - best_bid

            # Toxicity Check 1: Order Book Imbalance (OBI)
            bid_vol = sum(od.buy_orders.values())
            ask_vol = sum(abs(qty) for qty in od.sell_orders.values())
            total_vol = bid_vol + ask_vol
            obi = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0

            inner_bid = best_bid + 1
            inner_ask = best_ask - 1
            fair = (inner_bid + inner_ask) / 2.0
            edge = (inner_ask - inner_bid) / 2.0

            pos = state.position.get(product, 0)
            limit = self.position_limits.get(product, 10)

            orders: List[Order] = []

            # Phase 1: Take mispriced liquidity
            take_orders, pos = self._take(od, fair, edge, product, pos, limit)
            orders.extend(take_orders)

            # Phase 2: Clear any accumulated inventory instantly at fair value
            clear_orders, pos = self._clear(fair, product, pos, limit)
            orders.extend(clear_orders)

            # Phase 3: Make balanced quotes (Speculative Spread Capture)
            make_orders = self._make_balanced(inner_bid, inner_ask, product, pos, limit, best_bid, best_ask)

            for order in make_orders:
                # GUARDRAIL: If we are already long (pos > 0) AND there is extreme sell pressure, do not buy more.
                if order.quantity > 0 and pos > 0 and obi < -0.80:
                    continue
                # GUARDRAIL: If we are already short (pos < 0) AND there is extreme buy pressure, do not sell more.
                if order.quantity < 0 and pos < 0 and obi > 0.80:
                    continue
                orders.append(order)

            if orders: result[product] = orders

    def v4_maker(self, state: TradingState, result: dict, ewma_state: dict):
        EWMA_ALPHA = 0.93
        MR_MIN_VAR = 4.0
        MR_K = 1.5
        MR_CAP = 4
        INV_SKEW = 0.5

        for product in self.SKIP:
            od = state.order_depths.get(product)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            best_bid   = max(od.buy_orders.keys())
            best_ask   = min(od.sell_orders.keys())
            book_spread = best_ask - best_bid
            if book_spread <= 0:
                continue

            # Microprice calculation
            bid_vol = od.buy_orders[best_bid]
            ask_vol = abs(od.sell_orders[best_ask])
            total   = bid_vol + ask_vol
            if total <= 0:
                fair = (best_bid + best_ask) / 2.0
            else:
                fair = (best_bid * ask_vol + best_ask * bid_vol) / total

            # Update EWMA mean + variance
            prev = ewma_state.get(product)
            if prev is None:
                ewma_m, ewma_v = fair, 0.0
            else:
                ewma_m = EWMA_ALPHA * prev["m"] + (1 - EWMA_ALPHA) * fair
                ewma_v = EWMA_ALPHA * prev["v"] + (1 - EWMA_ALPHA) * (fair - ewma_m) ** 2
            ewma_state[product] = {"m": ewma_m, "v": ewma_v}

            pos = state.position.get(product, 0)
            limit = self.position_limits.get(product, 10)

            # Mean-reversion target nudge
            if ewma_v > MR_MIN_VAR:
                z      = (fair - ewma_m) / math.sqrt(ewma_v)
                mr_adj = max(-MR_CAP, min(MR_CAP, -MR_K * z))
            else:
                mr_adj = 0.0

            dyn_target = max(-limit, min(limit, mr_adj))

            # Inventory skew anchored at dynamic target
            deviation   = pos - dyn_target
            skewed_fair = fair - INV_SKEW * deviation

            # Half-spread logic
            half = max(1.0, book_spread / 4.0)

            our_bid = math.floor(skewed_fair - half)
            our_ask = math.ceil(skewed_fair  + half)
            if our_ask <= our_bid:
                our_ask = our_bid + 1

            buy_cap  = limit - pos
            sell_cap = limit + pos

            orders: List[Order] = []

            # 1. Take mispriced liquidity immediately
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

            # 2. Quote residual capacity passively
            quote_buy  = max(0, buy_cap  - ask_taken)
            quote_sell = max(0, sell_cap - bid_taken)

            if quote_buy > 0:
                orders.append(Order(product, int(our_bid), quote_buy))
            if quote_sell > 0:
                orders.append(Order(product, int(our_ask), -quote_sell))

            if orders:
                result[product] = orders