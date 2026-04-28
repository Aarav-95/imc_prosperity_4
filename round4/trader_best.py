# official 68K final

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



def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def top_obi(order_depth):
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0.0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    q_bid = order_depth.buy_orders[best_bid]
    q_ask = -order_depth.sell_orders[best_ask]
    if q_bid + q_ask == 0:
        return 0.0
    return (q_bid - q_ask) / (q_bid + q_ask)

class ProductBook:
    def __init__(self, product, state, limit):
        self.product = product
        self.state = state
        self.limit = limit
        self.orders = []

        self.depth = state.order_depths.get(product)
        if not self.depth:
            self.buy_orders = {}
            self.sell_orders = {}
        else:
            self.buy_orders = self.depth.buy_orders
            self.sell_orders = self.depth.sell_orders

        self.best_bid = max(self.buy_orders.keys()) if self.buy_orders else None
        self.best_ask = min(self.sell_orders.keys()) if self.sell_orders else None
        self.mid = (self.best_bid + self.best_ask) / 2.0 if self.best_bid and self.best_ask else None

        self.start_pos = state.position.get(product, 0)
        self.current_pos = self.start_pos

    def buy(self, price, size):
        allowed = self.limit - self.current_pos
        actual_size = min(size, allowed)
        if actual_size > 0:
            self.orders.append(Order(self.product, price, actual_size))
            self.current_pos += actual_size

    def sell(self, price, size):
        allowed = self.limit + self.current_pos
        actual_size = min(size, allowed)
        if actual_size > 0:
            self.orders.append(Order(self.product, price, -actual_size))
            self.current_pos -= actual_size

    def projected_position(self):
        return self.current_pos

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
        self.TOTAL_DAYS_TO_EXPIRY = 4         # Round 4 = 1 day after Round 3 (was 5)
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

        self.OPTIONS_Z = [
            {"symbol": "VEV_4000", "mean": 1247.0, "sd": 17.114, "z_thresh": 1.5, "take_size": 10, "limit": 300, "prior": 10000},
            {"symbol": "VEV_4500", "mean":  747.0, "sd": 17.105, "z_thresh": 1.5, "take_size": 10, "limit": 300, "prior": 10000},
            {"symbol": "VEV_5000", "mean":  252.0, "sd": 16.381, "z_thresh": 1.0, "take_size": 10, "limit": 300, "prior": 10000},
            {"symbol": "VEV_5100", "mean":  163.0, "sd": 15.327, "z_thresh": 1.0, "take_size": 50, "limit": 300, "prior": 10000},
            {"symbol": "VEV_5300", "mean":   43.0, "sd":  8.976, "z_thresh": 0.5, "take_size": 10, "limit": 300, "prior": 200},
            {"symbol": "VEV_5400", "mean":   14.0, "sd":  4.608, "z_thresh": 0.5, "take_size": 25, "limit": 300, "prior": 500},
        ]

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

        # ── Track which day we're on ──────────────────────────────
        last_ts = trader_data.get("last_ts", -1)
        if state.timestamp < last_ts:
            # Timestamp wrapped around → new day
            trader_data["day"] = trader_data.get("day", 0) + 1
        trader_data["last_ts"] = state.timestamp

        result["HYDROGEL_PACK"] = self.trade_hydrogel(state, trader_data)
        self.trade_vouchers_and_hedge(state, trader_data, result)
        self._trade_vfe_predator(state, trader_data, result)
        self._trade_z_take(state, trader_data, result)

        # ── Serialize persistent state ────────────────────────────
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_hydrogel(self, state: TradingState, trader_data: dict) -> List[Order]:
        product = "HYDROGEL_PACK"
        LIMIT = self.position_limits[product]

        # strategy params
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

        # dynamic mean (slow EMA) to prevent structural loss if fair value shifts
        ema_mean = trader_data.get("hydrogel_ema", mid)
        # alpha = 0.0005 adapts slowly to true market shifts without ruining short-term MR
        ema_mean = (mid * 0.0005) + (ema_mean * 0.9995)
        trader_data["hydrogel_ema"] = ema_mean

        # compute order book imbalance
        total_bid_vol = sum(order_depth.buy_orders.values())
        total_ask_vol = sum(-v for v in order_depth.sell_orders.values())
        total_vol = total_bid_vol + total_ask_vol

        if total_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / total_vol
        else:
            imbalance = 0

        # Mark 14 is a directional taker. When they sweep the book, price follows.
        # We shift fair value +1 or -1 to front-run their momentum.
        mark14_buying  = any(t.buyer  == "Mark 14" for t in state.market_trades.get(product, []))
        mark14_selling = any(t.seller == "Mark 14" for t in state.market_trades.get(product, []))

        cp_adjustment = 0
        if mark14_buying:
            cp_adjustment = 1
        elif mark14_selling:
            cp_adjustment = -1

        # mean reversion adjustment
        deviation = mid - ema_mean
        if abs(deviation) > MR_THRESHOLD:
            mr_adjustment = -MR_STRENGTH * deviation
        else:
            mr_adjustment = 0

        # fair value = mid + imbalance + inventory skew + mean reversion + counterparty momentum
        imb_adjustment = round(imbalance * IMB_SHIFT)
        inv_skew = -position * SKEW_FACTOR
        fv = round(mid + imb_adjustment + inv_skew + mr_adjustment + cp_adjustment)

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


    # ══════════════════════════════════════════════════════════════
    # VOUCHER IV TRADING + DELTA HEDGE
    # ══════════════════════════════════════════════════════════════

    def _get_mid(self, order_depth: OrderDepth) -> float:
        """Get mid-price from a two-sided order book. Returns None if one-sided."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

    def _compute_tte_years(self, state: TradingState, trader_data: dict) -> float:
        """Time to expiry in years, decreasing as we progress through days/ticks."""
        day = trader_data.get("day", 0)
        tte_days = self.TOTAL_DAYS_TO_EXPIRY - day - (state.timestamp / self.TICKS_PER_DAY)
        return max(tte_days, 0.01) / 252  # Convert to years

    def trade_vouchers_and_hedge(self, state: TradingState, trader_data: dict, result: dict):
        """
        IV Scalping — exploits negative autocorrelation in price deviations
        from the smile-fitted BS fair value.

        Pipeline:
        1. Compute IV for each voucher from its market mid-price
        2. Fit a quadratic smile (IV vs moneyness) across all strikes
        3. For each tradeable strike: BS_fair = BS(S, K, T, fitted_IV)
        4. Scalp deviations: buy when market < BS_fair, sell when market > BS_fair
        5. Delta-hedge the aggregate position with the underlying
        """
        underlying = "VELVETFRUIT_EXTRACT"
        MIN_EXTRINSIC = 10  # Min extrinsic value to consider a strike for scalping

        # ── Get underlying price ────────────────────────────────
        od_u = state.order_depths.get(underlying)
        if not od_u:
            return
        S = self._get_mid(od_u)
        if S is None:
            return

        T = self._compute_tte_years(state, trader_data)

        # ── Compute IV for ALL strikes (for smile fitting) ──────
        # Filter out options with < 2 ticks of extrinsic value — their
        # IV is unreliable (as per past team: "outliers were disregarded")
        fit_ms = []     # moneyness values for fitting
        fit_ivs = []    # IV values for fitting
        strike_data = {}  # voucher -> (moneyness, iv, market_mid)
        extrinsic_map = {}  # voucher -> extrinsic value (for strike selection)

        for voucher, K in self.VOUCHER_STRIKES.items():
            od_v = state.order_depths.get(voucher)
            if not od_v:
                continue
            v_mid = self._get_mid(od_v)
            if v_mid is None or v_mid <= 0.5:
                continue

            # Skip if extrinsic value too low (IV unreliable)
            intrinsic = max(0, S - K)
            extrinsic = v_mid - intrinsic
            if extrinsic < 2:
                continue
            # Skip deep ITM — their IV is unreliable and distorts the smile
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
            return  # Not enough data to fit a smile

        # ── Multi-strike scalping ─────────────────────────────────
        # Mark 22 systematically sells options across ALL OTM/near-ATM
        # strikes. Scalp ALL eligible strikes to capture this flow.
        # Exclude strikes where intrinsic > 150: these are in the
        # "dead zone" (VEV_5100) — too deep for smile accuracy,
        # too shallow for intrinsic-only pricing (_trade_deep_itm_mm).
        SCALP_STRIKES = {}
        for voucher, K in self.VOUCHER_STRIKES.items():
            if voucher in [cfg["symbol"] for cfg in self.OPTIONS_Z]:
                continue
            intrinsic_k = max(0, S - K)
            if (voucher in strike_data
                and extrinsic_map.get(voucher, 0) >= MIN_EXTRINSIC
                and S > K
                and intrinsic_k < 150
                and voucher != "VEV_5100"):  # Hardcode exclude: loses -22K in backtest
                SCALP_STRIKES[voucher] = K

        # ── Fit quadratic smile: IV = a*m^2 + b*m + c ──────────
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

        # ── Scalp each tradeable strike ─────────────────────────
        net_option_delta = 0.0

        for voucher, K in SCALP_STRIKES.items():
            od_v = state.order_depths.get(voucher)

            if voucher not in strike_data or not od_v:
                # Still track delta for existing positions
                pos = state.position.get(voucher, 0)
                if pos != 0:
                    m_approx = math.log(S / K)
                    fiv = max(sa * m_approx ** 2 + sb * m_approx + sc, 0.01)
                    net_option_delta += pos * bs_delta(S, K, T, fiv)
                continue

            m, actual_iv, market_mid = strike_data[voucher]
            fitted_iv = max(sa * m * m + sb * m + sc, 0.01)

            # BS fair price at the SMILE-implied vol
            bs_fair = bs_call_price(S, K, T, fitted_iv)
            delta = bs_delta(S, K, T, fitted_iv)
            fv = round(bs_fair)

            position = state.position.get(voucher, 0)
            limit = self.position_limits.get(voucher, 300)
            buy_cap = limit - position
            sell_cap = limit + position

            net_option_delta += position * delta

            orders: List[Order] = []

            # ── Phase 1: Take mispriced orders ──────────────────
            # No threshold — trade EVERY deviation. The negative AC(-0.4)
            # means deviations reverse, so we scalp both sides.
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

            # ── Phase 2: Post passive quotes at fair ± 1 ────────
            if buy_cap > 0:
                orders.append(Order(voucher, fv - 1, buy_cap))
            if sell_cap > 0:
                orders.append(Order(voucher, fv + 1, -sell_cap))

            if orders:
                result[voucher] = orders

    def _trade_z_take(self, state: TradingState, trader_data: dict, result: dict):
        zt = trader_data.setdefault("_ema", {})

        S = None
        od_u = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if od_u and od_u.buy_orders and od_u.sell_orders:
            S = (max(od_u.buy_orders) + min(od_u.sell_orders)) / 2.0

        ema_S = zt.get("VELVETFRUIT_EXTRACT", S if S else 5247.0)
        if S is not None:
            ema_S = 0.9 * ema_S + 0.1 * S
            zt["VELVETFRUIT_EXTRACT"] = ema_S

        vfe_deviation = (S - ema_S) if S is not None else 0.0

        for cfg in self.OPTIONS_Z:
            sym = cfg["symbol"]

            # 5100 only acts like a deep ITM option when S > 5175
            if sym == "VEV_5100" and S is not None and S < 5175:
                continue

            depth = state.order_depths.get(sym)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
            static_mean = float(cfg["mean"])
            prior = float(cfg["prior"])
            alpha = 1.0 / prior if prior > 0 else 0.002

            ema = zt.get(sym, static_mean)
            eff_mean = (1.0 - alpha) * ema + alpha * mid
            zt[sym] = eff_mean

            sd = float(cfg["sd"])
            z = (mid - eff_mean) / sd if sd > 0 else 0
            if abs(z) < cfg["z_thresh"]:
                continue

            # Guardrails for high-risk OTM options
            if sym in ["VEV_5300", "VEV_5400"]:
                # 1. Do not catch falling knives if underlying VFE is also dumping
                if z < 0 and vfe_deviation < -0.5:
                    continue
                # 2. Do not short a zooming option if underlying VFE is ripping
                if z > 0 and vfe_deviation > 0.5:
                    continue
                # 3. Theta Trap: do not buy deeply depreciated options near expiry value
                if z < 0 and mid <= 3.0:
                    continue

            pos = state.position.get(sym, 0)
            limit = cfg["limit"]
            take_size = cfg["take_size"]
            orders = []

            if z > 0:
                room = max(0, min(take_size, limit + pos))
                if room > 0:
                    filled = 0
                    for px in sorted(depth.buy_orders, reverse=True):
                        if filled >= room or px < eff_mean: break
                        qty = min(depth.buy_orders[px], room - filled)
                        if qty > 0:
                            orders.append(Order(sym, px, -qty))
                            filled += qty
            elif z < 0:
                room = max(0, min(take_size, limit - pos))
                if room > 0:
                    filled = 0
                    for px in sorted(depth.sell_orders):
                        if filled >= room or px > eff_mean: break
                        qty = min(-depth.sell_orders[px], room - filled)
                        if qty > 0:
                            orders.append(Order(sym, px, qty))
                            filled += qty

            if orders:
                result[sym] = orders

        # NOTE: Delta hedge replaced by VFE z-score mean-reversion
        # (called separately from run())

    def _trade_vfe_predator(self, state: TradingState, trader_data: dict, result: dict):
        VELVET = "VELVETFRUIT_EXTRACT"
        VF_AR2_ALPHA = trader_data.get("VF_AR2_ALPHA", 0.0)
        VF_AR2_BETA1 = trader_data.get("VF_AR2_BETA1", 1.0)
        VF_AR2_BETA2 = trader_data.get("VF_AR2_BETA2", -0.5)
        VF_MU = trader_data.get("VF_MU", 5000.0)
        VF_STD = trader_data.get("VF_STD", 15.0)

        if "last_mid" not in trader_data: trader_data["last_mid"] = {}
        if "vf_prev_mid" not in trader_data: trader_data["vf_prev_mid"] = 5000.0
        if "vf_prev_prev" not in trader_data: trader_data["vf_prev_prev"] = 5000.0

        book = ProductBook(VELVET, state, self.position_limits[VELVET])
        if book.mid is None or book.best_bid is None or book.best_ask is None:
            return

        prev_mid = float(trader_data["last_mid"].get(VELVET, book.mid))
        prev_prev = trader_data["vf_prev_prev"]
        last_prev = trader_data["vf_prev_mid"]

        trader_data["vf_prev_prev"] = last_prev
        trader_data["vf_prev_mid"] = book.mid
        trader_data["last_mid"][VELVET] = book.mid

        ar2_fair = None
        if isinstance(last_prev, (int, float)) and isinstance(prev_prev, (int, float)):
            ar2_fair = VF_AR2_ALPHA + VF_AR2_BETA1 * float(last_prev) + VF_AR2_BETA2 * float(prev_prev)

        base_fair = book.mid if ar2_fair is None else 0.65 * book.mid + 0.35 * ar2_fair
        momentum = book.mid - prev_mid
        obi = top_obi(book.depth)
        fair = base_fair + 0.10 * (VF_MU - base_fair) - 0.12 * momentum + 0.75 * obi

        for ask_price, ask_volume in book.sell_orders.items():
            if ask_price > fair - 1.4:
                break
            size = min(abs(ask_volume), 32, int(10 + 5 * (fair - ask_price)))
            book.buy(ask_price, size)

        for bid_price, bid_volume in book.buy_orders.items():
            if bid_price < fair + 1.4:
                break
            size = min(abs(bid_volume), 32, int(10 + 5 * (bid_price - fair)))
            book.sell(bid_price, size)

        projected = book.projected_position()
        mean_dev = (book.mid - VF_MU) / VF_STD
        skew_ticks = int(round(clamp(mean_dev * 3.0 + 0.9 * obi, -5.0, 5.0)))
        inv_lean = 6.0 * (projected / book.limit)

        thick_bid = next((price for price, vol in book.buy_orders.items() if price <= fair and abs(vol) > 1), None)
        thick_ask = next((price for price, vol in book.sell_orders.items() if price >= fair and abs(vol) > 1), None)
        base_bid = (thick_bid + 1) if thick_bid is not None else int(fair - 3)
        base_ask = (thick_ask - 1) if thick_ask is not None else int(fair + 3)
        bid_quote = min(base_bid - skew_ticks - int(round(inv_lean)), math.floor(fair - 1))
        ask_quote = max(base_ask - skew_ticks - int(round(inv_lean)), math.ceil(fair + 1))

        normalized = min(abs(mean_dev) / 3.0, 1.0)
        if mean_dev > 0:
            bid_size = min(22, max(4, int(6 + 18 * normalized)))
            ask_size = 6
        else:
            bid_size = 6
            ask_size = min(22, max(4, int(6 + 18 * normalized)))

        if bid_quote < book.best_ask:
            book.buy(int(bid_quote), bid_size)
        if ask_quote > book.best_bid:
            book.sell(int(ask_quote), ask_size)

        logger.print(f"VF mid={book.mid:.1f} fair={fair:.1f} mean_dev={mean_dev:.2f} obi={obi:.2f} pos={book.projected_position()}")

        if book.orders:
            result[VELVET] = book.orders

