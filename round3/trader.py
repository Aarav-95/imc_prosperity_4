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
        self._trade_vfe_zscore(state, trader_data, result)
        self._trade_deep_itm_mm(state, result)

        # ── Serialize persistent state ────────────────────────────
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_hydrogel(self, state: TradingState, trader_data: dict) -> List[Order]:
        product = "HYDROGEL_PACK"
        LIMIT = self.position_limits[product]

        # ── Tunable parameters ──────────────────────────────────
        SKEW_FACTOR = 0.10     # Inventory skew per unit position
        IMB_SHIFT = 2          # Shift FV by this many ticks when imbalance detected
        SPREAD = 1             # Min distance from FV for passive quotes

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)

        # ── Calculate best bid/ask and mid ───────────────────────
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

        # ── Compute order book imbalance ─────────────────────────
        # Sum all bid volume and ask volume across ALL levels
        total_bid_vol = sum(order_depth.buy_orders.values())     # positive
        total_ask_vol = sum(-v for v in order_depth.sell_orders.values())  # make positive
        total_vol = total_bid_vol + total_ask_vol

        if total_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / total_vol  # +1 = all bids, -1 = all asks
        else:
            imbalance = 0

        # ── Fair value = mid + imbalance signal + inventory skew ─
        # When bid_vol >> ask_vol: price likely to go UP → raise FV → more eager to buy
        # When ask_vol >> bid_vol: price likely to go DOWN → lower FV → more eager to sell
        imb_adjustment = round(imbalance * IMB_SHIFT)
        inv_skew = -position * SKEW_FACTOR
        fv = round(mid + imb_adjustment + inv_skew)

        buy_capacity = LIMIT - position
        sell_capacity = LIMIT + position

        # ── Phase 1: Take mispriced orders ───────────────────────
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

        # ── Phase 2: Post passive quotes around FV ───────────────
        if buy_capacity > 0:
            book_bid = best_bid if best_bid is not None else (fv - SPREAD - 1)
            our_bid = min(book_bid + 1, fv - SPREAD)
            orders.append(Order(product, our_bid, buy_capacity))

        if sell_capacity > 0:
            book_ask = best_ask if best_ask is not None else (fv + SPREAD + 1)
            our_ask = max(book_ask - 1, fv + SPREAD)
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

    def _trade_deep_itm_mm(self, state: TradingState, result: dict):
        """
        Market-make deep ITM options (VEV_4000) where spreads are 20+ ticks.
        FV = S - K (intrinsic value; extrinsic ≈ 0 for deep ITM).
        Strategy: overbid/undercut the book, take mispriced orders.
        """
        underlying = "VELVETFRUIT_EXTRACT"
        od_u = state.order_depths.get(underlying)
        if not od_u:
            return
        S = self._get_mid(od_u)
        if S is None:
            return

        MM_STRIKES = {}
        for voucher, K in self.VOUCHER_STRIKES.items():
            od_v = state.order_depths.get(voucher)
            if not od_v or not od_v.buy_orders or not od_v.sell_orders:
                continue

            v_mid = self._get_mid(od_v)
            if v_mid is None:
                continue

            intrinsic = S - K
            if intrinsic > 300: # Deep ITM
                extrinsic = v_mid - intrinsic
                if extrinsic < 15: # Very little option juice left
                    MM_STRIKES[voucher] = K

        SKEW_FACTOR = 0.02  # Per-unit position skew

        for voucher, K in MM_STRIKES.items():
            od_v = state.order_depths.get(voucher)

            # Fair value = intrinsic value
            fv = S - K
            if fv <= 0:
                continue

            position = state.position.get(voucher, 0)
            limit = self.position_limits.get(voucher, 300)

            # Inventory skew — push FV to encourage flattening
            skew = -position * SKEW_FACTOR
            fv_skewed = fv + skew

            buy_cap = limit - position
            sell_cap = limit + position

            best_bid = max(od_v.buy_orders.keys())
            best_ask = min(od_v.sell_orders.keys())

            orders: List[Order] = []

            # ── Phase 1: Take mispriced orders ───────────────────
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

            # ── Phase 2: Passive quotes inside the spread ────────
            fv_int = round(fv_skewed)

            if buy_cap > 0:
                # Overbid: post just above best_bid but below FV
                our_bid = min(best_bid + 1, fv_int - 1)
                orders.append(Order(voucher, our_bid, buy_cap))

            if sell_cap > 0:
                # Undercut: post just below best_ask but above FV
                our_ask = max(best_ask - 1, fv_int + 1)
                orders.append(Order(voucher, our_ask, -sell_cap))

            if orders:
                result[voucher] = orders

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

        # ── Dynamically select scalp strike with hysteresis ─────
        # Pick the nearest-to-ATM strike with enough extrinsic value.
        # Use hysteresis: only switch from current strike if a new one
        # is closer by >= HYSTERESIS ticks (prevents flip-flopping when
        # underlying sits between two strikes, e.g. VFE at 5250).
        HYSTERESIS = 30

        scalp_candidates = []
        for voucher, K in self.VOUCHER_STRIKES.items():
            # Only scalp ITM strikes (S > K) — OTM/ATM strikes like VEV_5300
            # are systematically mispriced by the smile fit
            if (voucher in strike_data
                and extrinsic_map.get(voucher, 0) >= MIN_EXTRINSIC
                and S > K):
                scalp_candidates.append((abs(S - K), voucher, K))

        scalp_candidates.sort()  # Closest to ATM first

        SCALP_STRIKES = {}
        if scalp_candidates:
            # Check if we have a previously active strike
            prev_strike = trader_data.get("active_scalp_strike")
            best_dist, best_voucher, best_K = scalp_candidates[0]

            if prev_strike and prev_strike in strike_data:
                prev_K = self.VOUCHER_STRIKES[prev_strike]
                prev_dist = abs(S - prev_K)
                # Only switch if new strike is closer by >= HYSTERESIS
                if prev_dist - best_dist >= HYSTERESIS:
                    SCALP_STRIKES = {best_voucher: best_K}
                    trader_data["active_scalp_strike"] = best_voucher
                else:
                    # Stick with current strike
                    SCALP_STRIKES = {prev_strike: prev_K}
            else:
                # Fresh start — use nearest ATM
                SCALP_STRIKES = {best_voucher: best_K}
                trader_data["active_scalp_strike"] = best_voucher

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

        # NOTE: Delta hedge replaced by VFE z-score mean-reversion
        # (called separately from run())

    def _trade_vfe_zscore(self, state: TradingState, trader_data: dict, result: dict):
        """
        Mean-reversion on VELVETFRUIT_EXTRACT using VWAP z-score.

        Replaces delta hedging — instead of neutralizing option delta,
        we trade VFE directionally based on its deviation from VWAP.
        Analysis showed this generates +114K vs -30K from hedging.

        Signal:
          z < -Z_ENTRY → BUY  (price below VWAP, expect rebound)
          z > +Z_ENTRY → SELL (price above VWAP, expect drop)
          z crosses 0  → FLATTEN (reverted to mean)
        """
        underlying = "VELVETFRUIT_EXTRACT"
        VWAP_WINDOW = 100
        Z_ENTRY = 1.5         # Enter when |z| > this
        Z_EXIT = 0.0          # Exit when z crosses zero
        TRADE_SIZE = 50       # Scale into position gradually

        od_u = state.order_depths.get(underlying)
        if not od_u:
            return

        S = self._get_mid(od_u)
        if S is None:
            return

        limit = self.position_limits.get(underlying, 200)
        current_pos = state.position.get(underlying, 0)

        # ── Update rolling VWAP from trade data ──────────────────
        # Use market_trades + own_trades to compute tick VWAP
        tick_pv = 0.0
        tick_vol = 0.0

        for trade in state.market_trades.get(underlying, []):
            tick_pv += trade.price * trade.quantity
            tick_vol += trade.quantity

        for trade in state.own_trades.get(underlying, []):
            tick_pv += trade.price * abs(trade.quantity)
            tick_vol += abs(trade.quantity)

        # If no trades this tick, use mid as proxy
        if tick_vol == 0:
            tick_pv = S
            tick_vol = 1

        # Persist rolling VWAP components in trader_data
        vwap_pv = trader_data.get("vwap_pv", [])   # list of (pv, vol) per tick
        vwap_pv.append([tick_pv, tick_vol])

        # Keep only last VWAP_WINDOW ticks
        if len(vwap_pv) > VWAP_WINDOW:
            vwap_pv = vwap_pv[-VWAP_WINDOW:]
        trader_data["vwap_pv"] = vwap_pv

        # Also track recent prices for std calculation
        vwap_prices = trader_data.get("vwap_prices", [])
        vwap_prices.append(S)
        if len(vwap_prices) > VWAP_WINDOW:
            vwap_prices = vwap_prices[-VWAP_WINDOW:]
        trader_data["vwap_prices"] = vwap_prices

        # Need enough history before trading
        if len(vwap_pv) < 20:
            return

        # ── Compute VWAP and z-score ─────────────────────────────
        total_pv = sum(x[0] for x in vwap_pv)
        total_vol = sum(x[1] for x in vwap_pv)
        vwap = total_pv / total_vol if total_vol > 0 else S

        mean_p = sum(vwap_prices) / len(vwap_prices)
        var_p = sum((p - mean_p) ** 2 for p in vwap_prices) / len(vwap_prices)
        std_p = math.sqrt(var_p) if var_p > 0 else 1

        z = (S - vwap) / std_p if std_p > 0.1 else 0

        # ── Generate orders based on z-score ──────────────────────
        orders: List[Order] = []

        best_bid = max(od_u.buy_orders.keys()) if od_u.buy_orders else None
        best_ask = min(od_u.sell_orders.keys()) if od_u.sell_orders else None

        if z < -Z_ENTRY and current_pos < limit:
            # Price below VWAP → BUY (expect rebound)
            qty = min(TRADE_SIZE, limit - current_pos)
            if qty > 0 and best_bid is not None:
                # Post PASSIVE bid — don't cross the spread
                orders.append(Order(underlying, best_bid + 1, qty))

        elif z > Z_ENTRY and current_pos > -limit:
            # Price above VWAP → SELL (expect drop)
            qty = min(TRADE_SIZE, limit + current_pos)
            if qty > 0 and best_ask is not None:
                # Post PASSIVE ask — don't cross the spread
                orders.append(Order(underlying, best_ask - 1, -qty))

        elif current_pos > 0 and z > Z_EXIT:
            # Long position has reverted — close passively
            if best_ask is not None:
                orders.append(Order(underlying, best_ask - 1, -current_pos))

        elif current_pos < 0 and z < -Z_EXIT:
            # Short position has reverted — close passively
            if best_bid is not None:
                orders.append(Order(underlying, best_bid + 1, -current_pos))

        if orders:
            result[underlying] = orders