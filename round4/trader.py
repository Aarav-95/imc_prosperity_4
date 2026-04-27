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
            "HYDROGEL_PACK": 200,
            "VELVETFRUIT_EXTRACT": 200,
            "VEV_4000": 300,
            "VEV_4500": 300,
            "VEV_5000": 300,
            "VEV_5100": 300,
            "VEV_5200": 300,
            "VEV_5300": 300,
            "VEV_5400": 300,
            "VEV_5500": 300,
            "VEV_6000": 300,
            "VEV_6500": 300,
        }

        # ── Options config ──────────────────────────────────────────
        self.TOTAL_DAYS_TO_EXPIRY = 5         # Calibrated from IV analysis
        self.TICKS_PER_DAY = 1_000_000        # Timestamps per day (0 to 999,900)

        # Vouchers we actively trade, grouped by strategy type
        self.VOUCHER_STRIKES = {
            "VEV_4000": 4000,
            "VEV_4500": 4500,
            "VEV_5000": 5000,
            "VEV_5100": 5100,
            "VEV_5200": 5200,    # Near ATM — primary buy target
            "VEV_5300": 5300,    # Near ATM — primary buy target
            "VEV_5400": 5400,
            "VEV_5500": 5500,
            "VEV_6000": 6000,    # Deep OTM — sell target
            "VEV_6500": 6500,    # Deep OTM — sell target
        }

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
            # timestamp wrapped around -> new day
            trader_data["day"] = trader_data.get("day", 0) + 1
        trader_data["last_ts"] = state.timestamp

        result["HYDROGEL_PACK"] = self.trade_hydrogel(state, trader_data)
        self.trade_vouchers_and_hedge(state, trader_data, result)
        self._trade_vfe_passive_mm(state, trader_data, result)
        self._trade_deep_itm_mm(state, trader_data, result)

        # serialize persistent state
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_hydrogel(self, state: TradingState, trader_data: dict) -> List[Order]:
        product = "HYDROGEL_PACK"
        LIMIT = self.position_limits[product]

        # strategy params
        FIXED_MEAN = 9995
        MR_THRESHOLD = 15
        MR_STRENGTH = 0.5
        SKEW_FACTOR = 0.10     # inventory skew per unit position
        IMB_SHIFT = 2          # fair value shift from order book imbalance
        SPREAD = 1             # min quote distance from fair value

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)

        # get best bid/ask and mid
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid = best_bid
        elif best_ask is not None:
            mid = best_ask
        else:
            return orders

        # compute order book imbalance
        total_bid_vol = sum(order_depth.buy_orders.values())
        total_ask_vol = sum(-v for v in order_depth.sell_orders.values())
        total_vol = total_bid_vol + total_ask_vol

        if total_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / total_vol
        else:
            imbalance = 0

        # mean reversion adjustment
        deviation = mid - FIXED_MEAN
        if abs(deviation) > MR_THRESHOLD:
            mr_adjustment = -MR_STRENGTH * deviation
        else:
            mr_adjustment = 0

        # fair value = mid + imbalance + inventory skew + mean reversion
        imb_adjustment = round(imbalance * IMB_SHIFT)
        inv_skew = -position * SKEW_FACTOR
        fv = round(mid + imb_adjustment + inv_skew + mr_adjustment)

        buy_capacity = LIMIT - position
        sell_capacity = LIMIT + position

        # react to mark 38 flow by tightening quotes
        mark38_trades = state.market_trades.get(product, [])
        mark38_buying  = any(t.buyer  == "Mark 38" for t in mark38_trades)
        mark38_selling = any(t.seller == "Mark 38" for t in mark38_trades)
        ask_spread = 0 if mark38_buying  else SPREAD
        bid_spread = 0 if mark38_selling else SPREAD

        # phase 1: take mispriced orders
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price < fv and buy_capacity > 0:
                qty = min(-ask_vol, buy_capacity)
                orders.append(Order(product, ask_price, qty))
                buy_capacity -= qty

        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fv and sell_capacity > 0:
                qty = min(bid_vol, sell_capacity)
                orders.append(Order(product, bid_price, -qty))
                sell_capacity -= qty

        # phase 2: post passive quotes around fair value
        if buy_capacity > 0:
            book_bid = best_bid if best_bid is not None else (fv - bid_spread - 1)
            our_bid = min(book_bid + 1, fv - bid_spread)
            orders.append(Order(product, our_bid, buy_capacity))

        if sell_capacity > 0:
            book_ask = best_ask if best_ask is not None else (fv + ask_spread + 1)
            our_ask = max(book_ask - 1, fv + ask_spread)
            orders.append(Order(product, our_ask, -sell_capacity))

        return orders

    # voucher iv trading

    def _get_mid(self, order_depth: OrderDepth) -> float:
        """get mid-price from a two-sided order book; return none if one-sided."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def _compute_tte_years(self, state: TradingState, trader_data: dict) -> float:
        """time to expiry in years, decreasing over days/ticks."""
        day = trader_data.get("day", 0)
        tte_days = self.TOTAL_DAYS_TO_EXPIRY - day - (state.timestamp / self.TICKS_PER_DAY)
        return max(tte_days, 0.01) / 252  # convert to years

    def _trade_deep_itm_mm(self, state: TradingState, trader_data: dict, result: dict):
        """trade deep itm options around intrinsic value."""
        underlying = "VELVETFRUIT_EXTRACT"
        od_u = state.order_depths.get(underlying)
        if not od_u:
            return
        S = self._get_mid(od_u)
        if S is None:
            return

        # identify deep itm strikes
        MM_STRIKES = {}
        for voucher, K in self.VOUCHER_STRIKES.items():
            od_v = state.order_depths.get(voucher)
            if not od_v or not od_v.buy_orders or not od_v.sell_orders:
                continue

            v_mid = self._get_mid(od_v)
            if v_mid is None:
                continue

            intrinsic = S - K
            if intrinsic > 300:  # deep itm
                extrinsic = v_mid - intrinsic
                if extrinsic < 20:
                    MM_STRIKES[voucher] = K

        SKEW_FACTOR = 0.02  # per-unit position skew

        for voucher, K in MM_STRIKES.items():
            od_v = state.order_depths.get(voucher)

            fv = S - K
            if fv <= 0:
                continue

            position = state.position.get(voucher, 0)
            limit = self.position_limits.get(voucher, 300)

            # inventory skew for position control
            skew = -position * SKEW_FACTOR
            fv_skewed = fv + skew

            buy_cap = limit - position
            sell_cap = limit + position

            best_bid = max(od_v.buy_orders.keys())
            best_ask = min(od_v.sell_orders.keys())

            orders: List[Order] = []

            # phase 1: take mispriced orders
            for ask_price, ask_vol in sorted(od_v.sell_orders.items()):
                if ask_price < fv_skewed and buy_cap > 0:
                    qty = min(-ask_vol, buy_cap)
                    orders.append(Order(voucher, ask_price, qty))
                    buy_cap -= qty

            for bid_price, bid_vol in sorted(od_v.buy_orders.items(), reverse=True):
                if bid_price > fv_skewed and sell_cap > 0:
                    qty = min(bid_vol, sell_cap)
                    orders.append(Order(voucher, bid_price, -qty))
                    sell_cap -= qty

            # phase 2: passive quotes inside spread
            fv_int = round(fv_skewed)

            if buy_cap > 0:
                our_bid = min(best_bid + 1, fv_int - 1)
                orders.append(Order(voucher, our_bid, buy_cap))

            if sell_cap > 0:
                our_ask = max(best_ask - 1, fv_int + 1)
                orders.append(Order(voucher, our_ask, -sell_cap))

            if orders:
                result[voucher] = orders

    def trade_vouchers_and_hedge(self, state: TradingState, trader_data: dict, result: dict):
        """iv scalping using smile-fitted black-scholes fair values."""
        underlying = "VELVETFRUIT_EXTRACT"
        MIN_EXTRINSIC = 10  # min extrinsic value for reliable iv

        # get underlying price
        od_u = state.order_depths.get(underlying)
        if not od_u:
            return
        S = self._get_mid(od_u)
        if S is None:
            return

        T = self._compute_tte_years(state, trader_data)

        # compute iv for all strikes used in smile fit
        fit_ms = []
        fit_ivs = []
        strike_data = {}
        extrinsic_map = {}

        for voucher, K in self.VOUCHER_STRIKES.items():
            od_v = state.order_depths.get(voucher)
            if not od_v:
                continue
            v_mid = self._get_mid(od_v)
            if v_mid is None or v_mid <= 0.5:
                continue

            intrinsic = max(0, S - K)
            extrinsic = v_mid - intrinsic
            if extrinsic < 2:
                continue
            if intrinsic > 300:
                continue

            extrinsic_map[voucher] = extrinsic

            iv = implied_vol(v_mid, S, K, T)
            if iv is not None and 0.01 < iv < 5.0:
                m = math.log(S / K)
                fit_ms.append(m)
                fit_ivs.append(iv)
                strike_data[voucher] = (m, iv, v_mid)

        if len(fit_ms) < 3:
            return

        # select one best strike with hysteresis
        HYSTERESIS = 30
        
        scalp_candidates = []
        for voucher, K in self.VOUCHER_STRIKES.items():
            if (voucher in strike_data
                and extrinsic_map.get(voucher, 0) >= MIN_EXTRINSIC
                and S > K):
                scalp_candidates.append((abs(S - K), voucher, K))

        scalp_candidates.sort()

        SCALP_STRIKES = {}
        if scalp_candidates:
            prev_strike = trader_data.get("active_scalp_strike")
            best_dist, best_voucher, best_K = scalp_candidates[0]

            if prev_strike and prev_strike in strike_data:
                prev_K = self.VOUCHER_STRIKES[prev_strike]
                prev_dist = abs(S - prev_K)
                if prev_dist - best_dist >= HYSTERESIS:
                    SCALP_STRIKES = {best_voucher: best_K}
                    trader_data["active_scalp_strike"] = best_voucher
                else:
                    SCALP_STRIKES = {prev_strike: prev_K}
            else:
                SCALP_STRIKES = {best_voucher: best_K}
                trader_data["active_scalp_strike"] = best_voucher

        # fit quadratic smile: iv = a*m^2 + b*m + c
        n = len(fit_ms)
        s0, s1 = n, sum(fit_ms)
        s2 = sum(m * m for m in fit_ms)
        s3 = sum(m * m * m for m in fit_ms)
        s4 = sum(m * m * m * m for m in fit_ms)
        sy = sum(fit_ivs)
        smy = sum(m * iv for m, iv in zip(fit_ms, fit_ivs))
        sm2y = sum(m * m * iv for m, iv in zip(fit_ms, fit_ivs))

        det = (s4 * (s2 * s0 - s1 * s1)
             - s3 * (s3 * s0 - s1 * s2)
             + s2 * (s3 * s1 - s2 * s2))

        if abs(det) < 1e-20:
            sorted_ivs = sorted(fit_ivs)
            sa, sb, sc = 0, 0, sorted_ivs[len(sorted_ivs) // 2]
        else:
            sa = (sm2y * (s2 * s0 - s1 * s1)
                - s3 * (smy * s0 - s1 * sy)
                + s2 * (smy * s1 - s2 * sy)) / det
            sb = (s4 * (smy * s0 - s1 * sy)
                - sm2y * (s3 * s0 - s1 * s2)
                + s2 * (s3 * sy - smy * s2)) / det
            sc = (s4 * (s2 * sy - smy * s1)
                - s3 * (s3 * sy - smy * s2)
                + sm2y * (s3 * s1 - s2 * s2)) / det

        # scalp selected strike(s)
        net_option_delta = 0.0

        for voucher, K in SCALP_STRIKES.items():
            od_v = state.order_depths.get(voucher)

            if voucher not in strike_data or not od_v:
                # still track delta for existing positions
                pos = state.position.get(voucher, 0)
                if pos != 0:
                    m_approx = math.log(S / K)
                    fiv = max(sa * m_approx ** 2 + sb * m_approx + sc, 0.01)
                    net_option_delta += pos * bs_delta(S, K, T, fiv)
                continue

            m, actual_iv, market_mid = strike_data[voucher]
            fitted_iv = max(sa * m * m + sb * m + sc, 0.01)

            # bs fair price at smile-implied vol
            bs_fair = bs_call_price(S, K, T, fitted_iv)
            delta = bs_delta(S, K, T, fitted_iv)
            fv = round(bs_fair)

            position = state.position.get(voucher, 0)
            limit = self.position_limits.get(voucher, 300)
            buy_cap = limit - position
            sell_cap = limit + position

            net_option_delta += position * delta

            orders: List[Order] = []

            # phase 1: take mispriced orders
            for ask_price, ask_vol in sorted(od_v.sell_orders.items()):
                if ask_price < fv and buy_cap > 0:
                    qty = min(-ask_vol, buy_cap)
                    orders.append(Order(voucher, ask_price, qty))
                    buy_cap -= qty
                    net_option_delta += qty * delta

            for bid_price, bid_vol in sorted(od_v.buy_orders.items(), reverse=True):
                if bid_price > fv and sell_cap > 0:
                    qty = min(bid_vol, sell_cap)
                    orders.append(Order(voucher, bid_price, -qty))
                    sell_cap -= qty
                    net_option_delta -= qty * delta

            # phase 2: post passive quotes at fair +/- 1
            if buy_cap > 0:
                orders.append(Order(voucher, fv - 1, buy_cap))
            if sell_cap > 0:
                orders.append(Order(voucher, fv + 1, -sell_cap))

            if orders:
                result[voucher] = orders

        # vfe strategy is called separately from run()

    def _trade_vfe_passive_mm(self, state: TradingState, _trader_data: dict, result: dict):
        """passive mm in vfe to absorb mark 55 flow."""
        product = "VELVETFRUIT_EXTRACT"
        MARK = "Mark 55"
        MAX_POS = 75
        SKEW_FACTOR = 0.08   # inventory skew per unit

        od = state.order_depths.get(product)
        if not od:
            return

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None or best_ask is None:
            return

        mid = (best_bid + best_ask) / 2
        position = state.position.get(product, 0)
        limit = self.position_limits[product]

        # check if mark 55 was active last tick
        mark55_buying = any(t.buyer == MARK for t in state.market_trades.get(product, []))
        mark55_selling = any(t.seller == MARK for t in state.market_trades.get(product, []))

        # inventory skew pulls quotes toward flat
        skew = -position * SKEW_FACTOR
        fv = mid + skew

        buy_cap = min(MAX_POS - position, limit - position)
        sell_cap = min(MAX_POS + position, limit + position)

        orders: List[Order] = []

        # phase 1: take orders mispriced vs fv
        for ask_price, ask_vol in sorted(od.sell_orders.items()):
            if ask_price < fv and buy_cap > 0:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask_price, qty))
                buy_cap -= qty

        for bid_price, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if bid_price > fv and sell_cap > 0:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid_price, -qty))
                sell_cap -= qty

        # phase 2: passive quotes; tighten when mark 55 is active
        spread = 1 if (mark55_buying or mark55_selling) else 2

        if buy_cap > 0:
            our_bid = min(best_bid + 1, round(fv) - spread)
            orders.append(Order(product, our_bid, buy_cap))

        if sell_cap > 0:
            our_ask = max(best_ask - 1, round(fv) + spread)
            orders.append(Order(product, our_ask, -sell_cap))

        if orders:
            result[product] = orders

    def _trade_vfe_zscore_minimal(self, state: TradingState, trader_data: dict, result: dict):
        """minimal vfe mean-reversion with conservative sizing."""
        underlying = "VELVETFRUIT_EXTRACT"
        VWAP_WINDOW = 100
        Z_ENTRY = 2.0         # higher threshold = fewer entries
        Z_EXIT = 0.5          # close sooner
        MAX_POS = 50          # smaller position limit
        TRADE_SIZE = 20       # smaller increments

        od_u = state.order_depths.get(underlying)
        if not od_u:
            return

        S = self._get_mid(od_u)
        if S is None:
            return

        current_pos = state.position.get(underlying, 0)

        # update rolling vwap
        tick_pv = 0.0
        tick_vol = 0.0

        for trade in state.market_trades.get(underlying, []):
            tick_pv += trade.price * trade.quantity
            tick_vol += trade.quantity

        for trade in state.own_trades.get(underlying, []):
            tick_pv += trade.price * abs(trade.quantity)
            tick_vol += abs(trade.quantity)

        if tick_vol == 0:
            tick_pv = S
            tick_vol = 1

        vwap_pv = trader_data.get("vwap_pv", [])
        vwap_pv.append([tick_pv, tick_vol])

        if len(vwap_pv) > VWAP_WINDOW:
            vwap_pv = vwap_pv[-VWAP_WINDOW:]
        trader_data["vwap_pv"] = vwap_pv

        vwap_prices = trader_data.get("vwap_prices", [])
        vwap_prices.append(S)
        if len(vwap_prices) > VWAP_WINDOW:
            vwap_prices = vwap_prices[-VWAP_WINDOW:]
        trader_data["vwap_prices"] = vwap_prices

        if len(vwap_pv) < 20:
            return

        total_pv = sum(x[0] for x in vwap_pv)
        total_vol = sum(x[1] for x in vwap_pv)
        vwap = total_pv / total_vol if total_vol > 0 else S

        mean_p = sum(vwap_prices) / len(vwap_prices)
        var_p = sum((p - mean_p) ** 2 for p in vwap_prices) / len(vwap_prices)
        std_p = math.sqrt(var_p) if var_p > 0 else 1

        z = (S - vwap) / std_p if std_p > 0.1 else 0

        orders: List[Order] = []

        best_bid = max(od_u.buy_orders.keys()) if od_u.buy_orders else None
        best_ask = min(od_u.sell_orders.keys()) if od_u.sell_orders else None

        if z < -Z_ENTRY and current_pos < MAX_POS:
            qty = min(TRADE_SIZE, MAX_POS - current_pos)
            if qty > 0 and best_bid is not None:
                orders.append(Order(underlying, best_bid + 1, qty))

        elif z > Z_ENTRY and current_pos > -MAX_POS:
            qty = min(TRADE_SIZE, MAX_POS + current_pos)
            if qty > 0 and best_ask is not None:
                orders.append(Order(underlying, best_ask - 1, -qty))

        elif current_pos > 0 and z > Z_EXIT:
            if best_ask is not None:
                orders.append(Order(underlying, best_ask - 1, -min(current_pos, TRADE_SIZE)))

        elif current_pos < 0 and z < -Z_EXIT:
            if best_bid is not None:
                orders.append(Order(underlying, best_bid + 1, min(-current_pos, TRADE_SIZE)))

        if orders:
            result[underlying] = orders

    def _trade_butterfly(self, state: TradingState, trader_data: dict, result: dict):
        """sell butterfly spread when market is above model value."""
        BF_SELL_THRESHOLD = 1.5  # sell when market > model by this much
        MAX_BF_POSITION = 50     # max butterfly position per leg
        
        # get underlying price
        od_u = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not od_u:
            return
        S = self._get_mid(od_u)
        if S is None:
            return
        
        T = self._compute_tte_years(state, trader_data)
        
        # get option order depths
        od_5100 = state.order_depths.get("VEV_5100")
        od_5200 = state.order_depths.get("VEV_5200")
        od_5300 = state.order_depths.get("VEV_5300")
        
        if not all([od_5100, od_5200, od_5300]):
            return
        
        # get mid prices
        mid_5100 = self._get_mid(od_5100)
        mid_5200 = self._get_mid(od_5200)
        mid_5300 = self._get_mid(od_5300)
        
        if not all([mid_5100, mid_5200, mid_5300]):
            return
        
        # calculate market butterfly
        bf_market = mid_5100 - 2 * mid_5200 + mid_5300
        
        # calculate model butterfly
        IV = 0.27  # calibrated iv
        bf_theo = (bs_call_price(S, 5100, T, IV) 
                   - 2 * bs_call_price(S, 5200, T, IV) 
                   + bs_call_price(S, 5300, T, IV))
        
        mispricing = bf_market - bf_theo
        
        # current positions
        pos_5100 = state.position.get("VEV_5100", 0)
        pos_5200 = state.position.get("VEV_5200", 0)
        pos_5300 = state.position.get("VEV_5300", 0)
        
        # infer current butterfly position
        bf_pos = min(-pos_5100, pos_5200 // 2, -pos_5300) if pos_5100 <= 0 and pos_5300 <= 0 and pos_5200 >= 0 else 0
        
        orders_5100: List[Order] = []
        orders_5200: List[Order] = []
        orders_5300: List[Order] = []
        
        # sell butterfly if overpriced and we have room
        if mispricing > BF_SELL_THRESHOLD and bf_pos < MAX_BF_POSITION:
            qty = min(10, MAX_BF_POSITION - bf_pos)  # trade in small increments
            
            # check all legs fit position limits
            limit_5100 = self.position_limits.get("VEV_5100", 300)
            limit_5200 = self.position_limits.get("VEV_5200", 300)
            limit_5300 = self.position_limits.get("VEV_5300", 300)
            
            can_sell_5100 = limit_5100 + pos_5100 >= qty
            can_buy_5200 = limit_5200 - pos_5200 >= 2 * qty
            can_sell_5300 = limit_5300 + pos_5300 >= qty
            
            if can_sell_5100 and can_buy_5200 and can_sell_5300:
                # sell c5100
                best_bid_5100 = max(od_5100.buy_orders.keys()) if od_5100.buy_orders else None
                if best_bid_5100:
                    orders_5100.append(Order("VEV_5100", best_bid_5100, -qty))
                
                # buy 2x c5200
                best_ask_5200 = min(od_5200.sell_orders.keys()) if od_5200.sell_orders else None
                if best_ask_5200:
                    orders_5200.append(Order("VEV_5200", best_ask_5200, 2 * qty))
                
                # sell c5300
                best_bid_5300 = max(od_5300.buy_orders.keys()) if od_5300.buy_orders else None
                if best_bid_5300:
                    orders_5300.append(Order("VEV_5300", best_bid_5300, -qty))
        
        # close butterfly if mispricing is gone
        elif mispricing < 0.5 and bf_pos > 0:
            qty = min(10, bf_pos)
            
            # buy back c5100
            best_ask_5100 = min(od_5100.sell_orders.keys()) if od_5100.sell_orders else None
            if best_ask_5100:
                orders_5100.append(Order("VEV_5100", best_ask_5100, qty))
            
            # sell 2x c5200
            best_bid_5200 = max(od_5200.buy_orders.keys()) if od_5200.buy_orders else None
            if best_bid_5200:
                orders_5200.append(Order("VEV_5200", best_bid_5200, -2 * qty))
            
            # buy back c5300
            best_ask_5300 = min(od_5300.sell_orders.keys()) if od_5300.sell_orders else None
            if best_ask_5300:
                orders_5300.append(Order("VEV_5300", best_ask_5300, qty))
        
        # merge with existing orders
        if orders_5100:
            if "VEV_5100" not in result:
                result["VEV_5100"] = []
            result["VEV_5100"].extend(orders_5100)
        
        if orders_5200:
            if "VEV_5200" not in result:
                result["VEV_5200"] = []
            result["VEV_5200"].extend(orders_5200)
        
        if orders_5300:
            if "VEV_5300" not in result:
                result["VEV_5300"] = []
            result["VEV_5300"].extend(orders_5300)