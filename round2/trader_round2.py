import json
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
                compressed.append([
                    trade.symbol, trade.price, trade.quantity,
                    trade.buyer, trade.seller, trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff,
                observation.sugarPrice, observation.sunlightIndex,
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

PEPPER_SLOPE = 0.001

MARKET_ACCESS_FEE = 2500


class Trader:
    def __init__(self):
        self.position_limits = {
            "ASH_COATED_OSMIUM": 80,
            "INTARIAN_PEPPER_ROOT": 80,
        }
        self.osmium_fair_value = 10_000

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                pass

        result["ASH_COATED_OSMIUM"] = self.trade_osmium(state)
        result["INTARIAN_PEPPER_ROOT"] = self.trade_pepper(state, trader_data)

        trader_data["last_ts"] = state.timestamp
        new_trader_data = json.dumps(trader_data)
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data


    def trade_osmium(self, state: TradingState) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        FAIR_VALUE = self.osmium_fair_value
        LIMIT = self.position_limits[product]

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)

        inventory_factor = 0.08
        adjusted_fv = FAIR_VALUE - position * inventory_factor

        buy_capacity = LIMIT - position
        sell_capacity = LIMIT + position

        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price < adjusted_fv and buy_capacity > 0:
                qty = min(-ask_vol, buy_capacity)
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                logger.print(f"OSM TAKE BUY {qty}x @ {ask_price}")

        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > adjusted_fv and sell_capacity > 0:
                qty = min(bid_vol, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                logger.print(f"OSM TAKE SELL {qty}x @ {bid_price}")

        if position > 40:
            bid_offset, ask_offset = 3, 1
        elif position > 0:
            bid_offset, ask_offset = 2, 1
        elif position < -40:
            bid_offset, ask_offset = 1, 3
        elif position < 0:
            bid_offset, ask_offset = 1, 2
        else:
            bid_offset, ask_offset = 1, 1

        if buy_capacity > 0:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else (FAIR_VALUE - 3)
            our_bid = min(best_bid + 1, FAIR_VALUE - bid_offset)
            orders.append(Order(product, our_bid, buy_capacity))
            logger.print(f"OSM POST BID {buy_capacity}x @ {our_bid}")

        if sell_capacity > 0:
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else (FAIR_VALUE + 3)
            our_ask = max(best_ask - 1, FAIR_VALUE + ask_offset)
            orders.append(Order(product, our_ask, -sell_capacity))
            logger.print(f"OSM POST ASK {sell_capacity}x @ {our_ask}")

        return orders

    def trade_pepper(self, state: TradingState, trader_data: dict) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        LIMIT = self.position_limits[product]

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)
        ts = state.timestamp

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders
        mid_price = (best_bid + best_ask) / 2.0

        last_ts = trader_data.get("last_ts", ts)
        new_day = ts < last_ts
        if new_day or "pepper_anchor" not in trader_data:
            trader_data["pepper_anchor"] = mid_price - PEPPER_SLOPE * ts
            logger.print(f"PPR anchor={trader_data['pepper_anchor']:.1f} (new_day={new_day})")

        anchor = trader_data["pepper_anchor"]
        fair_value = anchor + PEPPER_SLOPE * ts

        buy_capacity = LIMIT - position

        BUY_TAKE_EDGE = 3
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price <= fair_value + BUY_TAKE_EDGE and buy_capacity > 0:
                qty = min(-ask_vol, buy_capacity)
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty
                logger.print(f"PPR TAKE BUY {qty}x @ {ask_price} fv={fair_value:.1f}")

        if buy_capacity > 0:
            our_bid = min(best_bid + 1, round(fair_value) - 1)
            orders.append(Order(product, our_bid, buy_capacity))
            logger.print(f"PPR POST BID {buy_capacity}x @ {our_bid} fv={fair_value:.1f}")

        SELL_SPIKE_EDGE = 8
        sell_capacity = position #never go net short
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price >= fair_value + SELL_SPIKE_EDGE and sell_capacity > 0:
                qty = min(bid_vol, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty
                logger.print(f"PPR TAKE SELL {qty}x @ {bid_price} fv={fair_value:.1f}")

        return orders
