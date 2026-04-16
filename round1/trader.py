import json
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

class Trader:

    def __init__(self):

        self.position_limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80
        }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # ── Deserialize persistent state ──────────────────────────
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                pass

        result["ASH_COATED_OSMIUM"] = self.trade_osmium(state)
        result["INTARIAN_PEPPER_ROOT"] = self.trade_pepper(state, trader_data)

        # ── Serialize persistent state ────────────────────────────
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_osmium(self, state: TradingState) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        FAIR_VALUE = 10_000
        LIMIT = self.position_limits[product]

        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        buy_capacity = LIMIT - position    # how much more we can buy
        sell_capacity = LIMIT + position    # how much more we can sell

        # ── Phase 1: Take all mispriced orders ──────────────────────

        # Buy from anyone selling below fair value (walk asks low → high)
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price < FAIR_VALUE and buy_capacity > 0:
                qty = min(-ask_vol, buy_capacity)  # ask_vol is negative
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                logger.print(f"TAKE BUY {qty}x @ {ask_price}")

        # Sell to anyone buying above fair value (walk bids high → low)
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > FAIR_VALUE and sell_capacity > 0:
                qty = min(bid_vol, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                logger.print(f"TAKE SELL {qty}x @ {bid_price}")

        # ── Phase 2: Post passive quotes that penny the book ─────────

        if buy_capacity > 0:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else (FAIR_VALUE - 2)
            our_bid = min(best_bid + 1, FAIR_VALUE - 1)  # overbid by 1, but stay below fair
            orders.append(Order(product, our_bid, buy_capacity))
            logger.print(f"POST BID {buy_capacity}x @ {our_bid}")

        if sell_capacity > 0:
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else (FAIR_VALUE + 2)
            our_ask = max(best_ask - 1, FAIR_VALUE + 1)  # undercut by 1, but stay above fair
            orders.append(Order(product, our_ask, -sell_capacity))
            logger.print(f"POST ASK {sell_capacity}x @ {our_ask}")

        return orders

    def trade_pepper(self, state: TradingState, trader_data: dict) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        LIMIT = self.position_limits[product]
        SLOPE = 0.001  # price increase per timestamp unit (~+1,000/day)

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)

        # ── Calculate current mid from order book ─────────────────
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        # Only trust mid when BOTH sides of the book exist.
        # One-sided books give a biased mid (off by ~half the spread).
        book_healthy = best_bid is not None and best_ask is not None
        mid = (best_bid + best_ask) / 2 if book_healthy else None

        # ── Compute dynamic fair value from linear trend ──────────
        if "pepper_base" in trader_data:
            base_price = trader_data["pepper_base"]
            base_ts = trader_data["pepper_ts"]
            fair_value = base_price + SLOPE * (state.timestamp - base_ts)

            # Only recalibrate from a healthy (two-sided) book
            if book_healthy and abs(fair_value - mid) > 10:
                trader_data["pepper_base"] = mid
                trader_data["pepper_ts"] = state.timestamp
                fair_value = mid
        else:
            # First tick: wait for a healthy book to anchor
            if not book_healthy:
                return orders
            fair_value = mid
            trader_data["pepper_base"] = mid
            trader_data["pepper_ts"] = state.timestamp

        fv = round(fair_value)

        buy_capacity = LIMIT - position
        sell_capacity = LIMIT + position

        # ── Phase 1: Take mispriced orders ────────────────────────

        # Buy from anyone selling below fair value
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price < fv and buy_capacity > 0:
                qty = min(-ask_vol, buy_capacity)
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                logger.print(f"PEPPER TAKE BUY {qty}x @ {ask_price}")

        # Sell to anyone buying above fair value
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fv and sell_capacity > 0:
                qty = min(bid_vol, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                logger.print(f"PEPPER TAKE SELL {qty}x @ {bid_price}")

        # ── Phase 2: Post passive quotes (long-biased) ────────────
        #   Bid 1 below FV (tight — we WANT to get filled on the long side)
        #   Ask 2 above FV (wider — less eager to sell against trend)

        if buy_capacity > 0:
            our_bid = min(
                best_bid + 1 if best_bid else fv - 1,
                fv - 1
            )
            orders.append(Order(product, our_bid, buy_capacity))
            logger.print(f"PEPPER POST BID {buy_capacity}x @ {our_bid}")

        if sell_capacity > 0:
            our_ask = max(
                best_ask - 1 if best_ask else fv + 2,
                fv + 2  # wider ask — reluctant to sell against the uptrend
            )
            orders.append(Order(product, our_ask, -sell_capacity))
            logger.print(f"PEPPER POST ASK {sell_capacity}x @ {our_ask}")

        return orders