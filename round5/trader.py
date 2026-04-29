import json
import math
from typing import Any
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
    """Standard normal CDF (Abramowitz & Stegun approximation)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * math.exp(-x*x/2)
    return 0.5 * (1.0 + sign * y)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes delta for a European call."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes vega (dC/dσ)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.sqrt(T) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi)


def implied_vol(C_market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    """Solve for implied volatility via Newton-Raphson + bisection fallback."""
    intrinsic = max(0, S - K * math.exp(-r * T))
    if C_market < intrinsic - 0.5:
        return 0.01
    if C_market >= S:
        return 5.0

    # Initial guess
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

    # Bisection fallback
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
            "GALAXY_SOUNDS_BLACK_HOLES": 10,
            "GALAXY_SOUNDS_DARK_MATTER": 10,
            "GALAXY_SOUNDS_PLANETARY_RINGS": 10,
            "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
            "GALAXY_SOUNDS_SOLAR_WINDS": 10,
            "MICROCHIP_CIRCLE": 10,
            "MICROCHIP_OVAL": 10,
            "MICROCHIP_RECTANGLE": 10,
            "MICROCHIP_SQUARE": 10,
            "MICROCHIP_TRIANGLE": 10,
            "OXYGEN_SHAKE_CHOCOLATE": 10,
            "OXYGEN_SHAKE_EVENING_BREATH": 10,
            "OXYGEN_SHAKE_GARLIC": 10,
            "OXYGEN_SHAKE_MINT": 10,
            "OXYGEN_SHAKE_MORNING_BREATH": 10,
            "PANEL_1X2": 10,
            "PANEL_1X4": 10,
            "PANEL_2X2": 10,
            "PANEL_2X4": 10,
            "PANEL_4X4": 10,
            "PEBBLES_L": 10,
            "PEBBLES_M": 10,
            "PEBBLES_S": 10,
            "PEBBLES_XL": 10,
            "PEBBLES_XS": 10,
            "ROBOT_DISHES": 10,
            "ROBOT_IRONING": 10,
            "ROBOT_LAUNDRY": 10,
            "ROBOT_MOPPING": 10,
            "ROBOT_VACUUMING": 10,
            "SLEEP_POD_COTTON": 10,
            "SLEEP_POD_LAMB_WOOL": 10,
            "SLEEP_POD_NYLON": 10,
            "SLEEP_POD_POLYESTER": 10,
            "SLEEP_POD_SUEDE": 10,
            "SNACKPACK_CHOCOLATE": 10,
            "SNACKPACK_PISTACHIO": 10,
            "SNACKPACK_RASPBERRY": 10,
            "SNACKPACK_STRAWBERRY": 10,
            "SNACKPACK_VANILLA": 10,
            "TRANSLATOR_ASTRO_BLACK": 10,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_GRAPHITE_MIST": 10,
            "TRANSLATOR_SPACE_GRAY": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "UV_VISOR_AMBER": 10,
            "UV_VISOR_MAGENTA": 10,
            "UV_VISOR_ORANGE": 10,
            "UV_VISOR_RED": 10,
            "UV_VISOR_YELLOW": 10,
        }

        # ── Products to SKIP ───────────────────────────────────────
        # These products consistently lose with penny-and-flatten:
        # either wide spreads make crossing too expensive, structural
        # drift overwhelms the spread capture, or low MR signal means
        # we just accumulate inventory into adverse moves.
        self.SKIP = {
            # Consistent --- losers (negative all 3 days)
            "MICROCHIP_SQUARE",             # -83K, wide spread + high vol
            "PEBBLES_M",                    # -59K, wide spread
            "PEBBLES_XS",                   # -52K, wide spread
            "ROBOT_VACUUMING",              # -35K, consistent bleeder
            "SLEEP_POD_COTTON",             # -87K, biggest loser
            "SLEEP_POD_SUEDE",              # -26K, consistent bleeder
            "TRANSLATOR_ECLIPSE_CHARCOAL",  # -15K, consistent bleeder
            "UV_VISOR_MAGENTA",             # -55K, wide spread
            "UV_VISOR_YELLOW",              # -152K, catastrophic
            # Large net losers (negative 2/3 days)
            "GALAXY_SOUNDS_DARK_MATTER",    # -33K
            "GALAXY_SOUNDS_SOLAR_FLAMES",   # -11K
            "PANEL_4X4",                    # -79K
            "ROBOT_MOPPING",                # -23K
            "SLEEP_POD_POLYESTER",          # -49K
            "SNACKPACK_STRAWBERRY",         # -21K
            "TRANSLATOR_GRAPHITE_MIST",     # -39K
            "TRANSLATOR_SPACE_GRAY",        # -30K
            # Borderline net losers (negative 2/3 days, small total)
            "PANEL_2X2",                    # -8K
            "ROBOT_DISHES",                 # -5K
            "ROBOT_LAUNDRY",                # -4K
            "UV_VISOR_ORANGE",              # -5K
        }

        self.ALL_PRODUCTS = [p for p in self.position_limits if p not in self.SKIP]

    # ══════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════

    def _get_mid(self, order_depth: OrderDepth) -> float:
        """Get mid-price from a two-sided order book. Returns None if one-sided."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    # ══════════════════════════════════════════════════════════════
    # CORE: TAKE → CLEAR → MAKE  (penny-and-flatten)
    #
    # Phase 1 — TAKE: sweep any book level priced better than our
    #   inner quote (best_bid+1 / best_ask-1). This captures any
    #   NPC mispricing that exceeds the penny improvement.
    #
    # Phase 2 — CLEAR: post the remaining inventory at fair value
    #   (the mid-price). This immediately flattens any position
    #   acquired during TAKE, limiting directional exposure.
    #
    # Phase 3 — MAKE: post balanced passive quotes at the penny-
    #   improved prices (bid+1 / ask-1). Skip the side that would
    #   worsen an existing position to avoid accumulating risk.
    #
    # The edge comes from the penny improvement: we buy 1 tick
    # above best_bid and sell 1 tick below best_ask, capturing
    # ~(spread - 2) ticks per round trip. The CLEAR step ensures
    # we never hold directional risk for more than one tick.
    # ══════════════════════════════════════════════════════════════

    def _take(self, od: OrderDepth, fair: float, edge: float,
              product: str, pos: int, limit: int) -> tuple:
        """
        Phase 1: sweep mispriced book levels.
        Buy anything priced below (fair - edge), sell above (fair + edge).
        Returns (orders, updated_pos).
        """
        orders: List[Order] = []

        # Buy underpriced asks
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price >= fair - edge:
                break
            avail = -od.sell_orders[ask_price]
            qty = min(avail, limit - pos)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                pos += qty

        # Sell overpriced bids
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price <= fair + edge:
                break
            avail = od.buy_orders[bid_price]
            qty = min(avail, limit + pos)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                pos -= qty

        return orders, pos

    def _clear(self, fair: float, product: str, pos: int, limit: int) -> tuple:
        """
        Phase 2: flatten inventory at fair value.
        If long, sell at fair. If short, buy at fair.
        Returns (orders, updated_pos).
        """
        orders: List[Order] = []
        fair_int = int(round(fair))

        if pos > 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0
        elif pos < 0:
            orders.append(Order(product, fair_int, -pos))
            pos = 0

        return orders, pos

    def _make_balanced(self, bid_price: int, ask_price: int,
                       product: str, pos: int, limit: int,
                       best_bid: int, best_ask: int) -> List[Order]:
        """
        Phase 3: post passive quotes inside the spread.
        Skip the side that would worsen an existing position
        (don't bid when long if our bid >= best_bid, etc.)
        Returns orders.
        """
        orders: List[Order] = []
        buy_cap = limit - pos
        sell_cap = limit + pos

        # When long, don't bid if our bid would join/improve the NPC best
        # bid — that just adds to our long exposure at a bad price.
        skip_bid = pos > 0 and bid_price >= best_bid
        skip_ask = pos < 0 and ask_price <= best_ask

        if buy_cap > 0 and not skip_bid:
            orders.append(Order(product, bid_price, buy_cap))
        if sell_cap > 0 and not skip_ask:
            orders.append(Order(product, ask_price, -sell_cap))

        return orders

    # ══════════════════════════════════════════════════════════════
    # RUN
    # ══════════════════════════════════════════════════════════════

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # deserialize persistent state
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                pass

        # track day via timestamp wrap
        last_ts = trader_data.get("last_ts", -1)
        if state.timestamp < last_ts:
            trader_data["day"] = trader_data.get("day", 0) + 1
        trader_data["last_ts"] = state.timestamp

        # ── Penny-and-flatten on every product ────────────────────
        self.penny_flatten(state, result)
        
        # serialize persistent state
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def penny_flatten(self, state: TradingState, result):
        for product in self.ALL_PRODUCTS:
            od = state.order_depths.get(product)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())

            # Penny-improved inner quotes
            inner_bid = best_bid + 1
            inner_ask = best_ask - 1

            # Fair value = midpoint of our inner quotes
            fair = (inner_bid + inner_ask) / 2.0
            edge = (inner_ask - inner_bid) / 2.0

            pos = state.position.get(product, 0)
            limit = self.position_limits.get(product, 10)

            orders: List[Order] = []

            # Phase 1: take mispriced levels
            take_orders, pos = self._take(od, fair, edge, product, pos, limit)
            orders.extend(take_orders)

            # Phase 2: clear inventory at fair
            clear_orders, pos = self._clear(fair, product, pos, limit)
            orders.extend(clear_orders)

            # Phase 3: make balanced quotes at penny-improved prices
            make_orders = self._make_balanced(
                inner_bid, inner_ask, product, pos, limit, best_bid, best_ask
            )
            orders.extend(make_orders)

            if orders:
                result[product] = orders