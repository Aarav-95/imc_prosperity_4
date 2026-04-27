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
        self._trade_deep_itm_mm(state, result)

        # ── Serialize persistent state ────────────────────────────
        traderData = json.dumps(trader_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def trade_hydrogel(self, state: TradingState, trader_data: dict) -> List[Order]:
        """
        Market-making HYDROGEL_PACK with counterparty detection.

        Mark 14 + Mark 38 form a closed market-making loop on HYDROGEL.
        Mark 14 is the more aggressive taker. When we detect Mark 14
        buying (from market_trades), we lean our FV upward to front-run
        the momentum. Vice versa for selling.
        """
        product = "HYDROGEL_PACK"
        LIMIT = self.position_limits[product]

        SKEW_FACTOR = 0.10
        IMB_SHIFT = 2
        SPREAD = 1
        CP_SHIFT = 1           # Additional FV shift when counterparty detected

        orders: List[Order] = []
        order_depth = state.order_depths.get(product)
        if not order_depth:
            return orders

        position = state.position.get(product, 0)

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

        # ── Detect Mark 14 / Mark 38 counterparty flow ───────────
        # Mark 14 is the aggressive taker in HYDROGEL.
        # If Mark 14 was buying last tick, price likely going UP.
        cp_bias = 0
        for trade in state.market_trades.get(product, []):
            if hasattr(trade, 'buyer'):
                if trade.buyer == "Mark 14":
                    cp_bias += trade.quantity   # Mark 14 buying → bullish
                elif trade.seller == "Mark 14":
                    cp_bias -= trade.quantity   # Mark 14 selling → bearish

        # Normalize: positive = bullish, negative = bearish
        if cp_bias > 0:
            cp_adjustment = CP_SHIFT
        elif cp_bias < 0:
            cp_adjustment = -CP_SHIFT
        else:
            cp_adjustment = 0

        # ── Order book imbalance ─────────────────────────────────
        total_bid_vol = sum(order_depth.buy_orders.values())
        total_ask_vol = sum(-v for v in order_depth.sell_orders.values())
        total_vol = total_bid_vol + total_ask_vol

        if total_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / total_vol
        else:
            imbalance = 0

        # ── Fair value = mid + signals ───────────────────────────
        imb_adjustment = round(imbalance * IMB_SHIFT)
        inv_skew = -position * SKEW_FACTOR
        fv = round(mid + imb_adjustment + inv_skew + cp_adjustment)

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

        # ── Multi-strike scalping ─────────────────────────────────
        # Mark 22 systematically sells options across ALL OTM/near-ATM
        # strikes. Scalp ALL eligible strikes to capture this flow.
        # Exclude strikes where intrinsic > 150: these are in the
        # "dead zone" (VEV_5100) — too deep for smile accuracy,
        # too shallow for intrinsic-only pricing (_trade_deep_itm_mm).
        SCALP_STRIKES = {}
        for voucher, K in self.VOUCHER_STRIKES.items():
            intrinsic_k = max(0, S - K)
            if (voucher in strike_data
                and extrinsic_map.get(voucher, 0) >= MIN_EXTRINSIC
                and S > K
                and intrinsic_k < 150):
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

        # NOTE: Delta hedge replaced by VFE z-score mean-reversion
        # (called separately from run())

    def _trade_vfe_predator(self, state: TradingState, trader_data: dict, result: dict):
        """
        Predatory VFE strategy targeting Mark 67 — the informed accumulator.

        Mark 67 profile (from data analysis):
        - Buys 500 lots/day of VFE, NEVER sells (net +1510 over 3 days)
        - Always lifts the ask (aggressive taker)
        - After M67 buys: passive bid earns +2.78 ticks (82% WR)
        - After M67 buys: crossing spread LOSES -2.38 ticks (21% WR)
        - Price drifts +2 ticks over next 1000 ticks after M67 buy

        Strategy (data-validated):
        1. Z-score mean-reversion as BASE signal (proven +9K PnL)
        2. DETECT Mark 67 buying → BIAS toward long:
           - Increase buy size, suppress sell pressure
           - Don't flatten longs while M67 is active
        3. NEVER cross the spread — always post passively
        4. Always post passive ask to sell TO Mark 67 when it lifts
        """
        underlying = "VELVETFRUIT_EXTRACT"
        VWAP_WINDOW = 100

        # ── Parameters ──────────────────────────────────────────
        M67_COOLDOWN = 5000    # Ticks to maintain M67 bullish bias (signal decays over ~5K ticks)
        M67_BUY_SIZE = 80      # Bigger passive bid when M67 is active
        PASSIVE_ASK_SIZE = 30  # Permanent ask to sell to M67
        BASE_TRADE_SIZE = 50   # Normal z-score trade size
        Z_ENTRY = 1.5          # Enter on z-score extreme
        Z_ENTRY_M67 = 0.5      # Lower threshold when M67 bias is on (lean long easier)
        Z_EXIT = 0.0

        od_u = state.order_depths.get(underlying)
        if not od_u:
            return

        S = self._get_mid(od_u)
        if S is None:
            return

        limit = self.position_limits.get(underlying, 200)
        current_pos = state.position.get(underlying, 0)

        best_bid = max(od_u.buy_orders.keys()) if od_u.buy_orders else None
        best_ask = min(od_u.sell_orders.keys()) if od_u.sell_orders else None

        if best_bid is None or best_ask is None:
            return

        # ── Detect Mark 67 & Mark 49 activity ────────────────────
        m67_detected = False
        m67_buy_vol = 0
        m49_detected = False

        for trade in state.market_trades.get(underlying, []):
            if hasattr(trade, 'buyer'):
                if trade.buyer == "Mark 67":
                    m67_detected = True
                    m67_buy_vol += trade.quantity
                if trade.seller == "Mark 49":
                    m49_detected = True

        # Persist detection with cooldown
        last_m67_ts = trader_data.get("last_m67_ts", -999999)
        if m67_detected:
            last_m67_ts = state.timestamp
        trader_data["last_m67_ts"] = last_m67_ts

        # M67 bias is active for M67_COOLDOWN ticks after last detection
        m67_bias = (state.timestamp - last_m67_ts) < M67_COOLDOWN

        # ── Update rolling VWAP ──────────────────────────────────
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

        # ── Compute VWAP and z-score ─────────────────────────────
        total_pv = sum(x[0] for x in vwap_pv)
        total_vol_hist = sum(x[1] for x in vwap_pv)
        vwap = total_pv / total_vol_hist if total_vol_hist > 0 else S

        mean_p = sum(vwap_prices) / len(vwap_prices)
        var_p = sum((p - mean_p) ** 2 for p in vwap_prices) / len(vwap_prices)
        std_p = math.sqrt(var_p) if var_p > 0 else 1

        z = (S - vwap) / std_p if std_p > 0.1 else 0

        # ── Select z-score thresholds based on M67 bias ──────────
        # When M67 bias is on: lower the buy threshold (easier to go long),
        # raise the sell threshold (harder to go short)
        z_buy = Z_ENTRY_M67 if m67_bias else Z_ENTRY
        z_sell = Z_ENTRY + 1.0 if m67_bias else Z_ENTRY  # harder to sell during M67

        # ── Generate orders ──────────────────────────────────────
        orders: List[Order] = []
        buy_capacity = limit - current_pos
        sell_capacity = limit + current_pos

        # ── Z-score entry with M67-adjusted thresholds ───────────
        if z < -z_buy and buy_capacity > 0:
            # When M67 bias: use larger size since we have conviction
            size = M67_BUY_SIZE if m67_bias else BASE_TRADE_SIZE
            qty = min(size, buy_capacity)
            # ALWAYS passive — data shows crossing loses money
            orders.append(Order(underlying, best_bid + 1, qty))
            buy_capacity -= qty

        elif z > z_sell and sell_capacity > 0:
            qty = min(BASE_TRADE_SIZE, sell_capacity)
            orders.append(Order(underlying, best_ask - 1, -qty))
            sell_capacity -= qty

        # ── Flatten when z reverts — but NOT during M67 bias ─────
        elif current_pos > 0 and z > Z_EXIT and not m67_bias:
            # Only flatten longs when M67 bias is OFF
            qty = min(current_pos, sell_capacity)
            if qty > 0:
                orders.append(Order(underlying, best_ask - 1, -qty))
                sell_capacity -= qty

        elif current_pos < 0 and z < -Z_EXIT:
            qty = min(-current_pos, buy_capacity)
            if qty > 0:
                orders.append(Order(underlying, best_bid + 1, qty))
                buy_capacity -= qty

        # ── ALWAYS: Post passive ask to capture Mark 67 flow ─────
        # Mark 67 lifts the ask ~55 times/day (avg 7-10 lots).
        # By posting best_ask-1, we undercut Mark 22 and get filled first.
        if sell_capacity > 0:
            ask_qty = min(PASSIVE_ASK_SIZE, sell_capacity)
            orders.append(Order(underlying, best_ask - 1, -ask_qty))

        # ── Post passive bid to buy from Mark 49 / Mark 22 ───────
        # Mark 49 sells ~1,071 lots at the ask. Mark 55 provides 
        # two-sided liquidity. Our bid collects cheap inventory
        # that Mark 67 later lifts from our ask.
        if buy_capacity > 0:
            bid_qty = min(PASSIVE_ASK_SIZE, buy_capacity)
            orders.append(Order(underlying, best_bid + 1, bid_qty))

        if orders:
            result[underlying] = orders