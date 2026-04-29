from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from collections import deque
import numpy as np
import json

LIMIT = 10

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

# Johansen cointegrating vector (normalized to CHOCOLATE=1):
#   CHOCOLATE + (−2.1259)*PISTACHIO + (−0.2320)*STRAWBERRY = stationary spread
SNACK_COINT  = ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY")
COINT_BETA   = (1.0, -2.1259, -0.2320)
COINT_SCALE  = 4      # lot size multiplier — yields [+4, -8, -1] / [-4, +8, +1]
COINT_WINDOW = 200    # rolling window for z-score
COINT_ENTER  = 2.0    # open when |z| crosses this
COINT_EXIT   = 0.5    # close when |z| falls below this

# Pre-compute integer leg targets (rounded from COINT_SCALE * COINT_BETA)
_LONG_TGTS  = tuple(int(round(COINT_SCALE * b)) for b in COINT_BETA)   # (+4, -8, -1)
_SHORT_TGTS = tuple(-t for t in _LONG_TGTS)                             # (-4, +8, +1)

GALAXY = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
]
GALAXY_BH           = "GALAXY_SOUNDS_BLACK_HOLES"
GALAXY_BH_BIAS      = 3     # persistent long target for BH (upward drift +34% over 3 days)
GALAXY_SIGNAL_DECAY = 1000  # ticks before momentum signal expires

MR_PRODUCTS = {
    "ROBOT_DISHES": -0.23,
    "ROBOT_IRONING": -0.13,
    "OXYGEN_SHAKE_EVENING_BREATH": -0.12,
    "OXYGEN_SHAKE_CHOCOLATE": -0.09,
}

ALL_PRODUCTS = [
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    *PEBBLES,
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
    "ROBOT_LAUNDRY", "ROBOT_IRONING",
    "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

def best(od: OrderDepth):
    bb = max(od.buy_orders)  if od.buy_orders  else None
    ba = min(od.sell_orders) if od.sell_orders else None
    bbv = od.buy_orders[bb]   if bb is not None else 0
    bav = -od.sell_orders[ba] if ba is not None else 0
    return bb, bbv, ba, bav


def mid(od: OrderDepth):
    bb, _, ba, _ = best(od)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2


def passive_quote(result, prod, od, lpos, fair_val, size):
    """Post passive buy/sell around fair_val strictly inside current spread."""
    bb, _, ba, _ = best(od)
    if bb is None or ba is None:
        return
    want_buy  = int(np.floor(fair_val)) - 1
    want_sell = int(np.ceil(fair_val))  + 1
    if bb < want_buy < ba and lpos < LIMIT:
        result[prod].append(Order(prod, want_buy,  min(size, LIMIT - lpos)))
    if bb < want_sell < ba and lpos > -LIMIT:
        result[prod].append(Order(prod, want_sell, -min(size, LIMIT + lpos)))


class Trader:
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        spread_hist        = deque(saved.get("spread_hist", []), maxlen=COINT_WINDOW)
        galaxy_signal      = saved.get("galaxy_signal", 0)
        galaxy_signal_age  = saved.get("galaxy_signal_age", 0)

        # Detect new day (timestamp resets) — clear cross-day contamination
        prev_ts = saved.get("prev_ts", -1)
        if int(state.timestamp) < int(prev_ts):
            spread_hist.clear()
            galaxy_signal     = 0
            galaxy_signal_age = 0

        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}
        pos = state.position

        # ── 1) PEBBLES: delta-neutral aggressive arb ──────────────────────────
        peb_mids = {p: mid(state.order_depths[p])
                    for p in PEBBLES if p in state.order_depths}
        fair: Dict[str, float] = {}
        if len(peb_mids) == 5 and all(m is not None for m in peb_mids.values()):
            for leg in PEBBLES:
                fair[leg] = 50000 - sum(peb_mids[o] for o in PEBBLES if o != leg)

            buy_cands, sell_cands = [], []
            for leg in PEBBLES:
                od = state.order_depths.get(leg)
                if od is None:
                    continue
                bb, bbv, ba, bav = best(od)
                lpos = pos.get(leg, 0)
                if ba is not None and ba < fair[leg] - 1 and lpos < LIMIT:
                    buy_cands.append((ba - fair[leg], leg, ba, bav, lpos))
                if bb is not None and bb > fair[leg] + 1 and lpos > -LIMIT:
                    sell_cands.append((bb - fair[leg], leg, bb, bbv, lpos))

            buy_cands.sort()
            sell_cands.sort(reverse=True)

            for (_, bl, bpx, bvol, bpos), (_, sl, spx, svol, spos) in zip(buy_cands, sell_cands):
                if bl == sl:
                    continue
                qty = min(bvol, svol, LIMIT - bpos, LIMIT + spos, 5)
                if qty > 0:
                    result[bl].append(Order(bl, bpx,  qty))
                    result[sl].append(Order(sl, spx, -qty))

        # ── 2) PEBBLES: passive quoting at implied fair ────────────────────────
        for leg in PEBBLES:
            if leg not in fair:
                continue
            od = state.order_depths.get(leg)
            if od is None:
                continue
            passive_quote(result, leg, od, pos.get(leg, 0), fair[leg], 5)

        # ── 3) GALAXY SOUNDS: basket momentum + BH long bias ────────────────────
        # A hidden basket agent always trades all 5 simultaneously (+/−6 from mid).
        # After a basket BUY, prices continue up 1-2 per product for 500-10k ticks;
        # after SELL, down 1-1.5. BH additionally has a persistent +34% upward drift.
        galaxy_ods = {p: state.order_depths.get(p) for p in GALAXY}

        # Detect basket event: all 5 products have market trades in the same direction
        basket_dirs = []
        for prod in GALAXY:
            od  = galaxy_ods.get(prod)
            trs = state.market_trades.get(prod, [])
            if not trs or od is None:
                basket_dirs.append(None)
                continue
            m = mid(od)
            if m is None:
                basket_dirs.append(None)
                continue
            total_vol = sum(abs(t.quantity) for t in trs)
            vwap      = sum(t.price * abs(t.quantity) for t in trs) / total_vol
            basket_dirs.append(1 if vwap > m else -1)

        non_none = [d for d in basket_dirs if d is not None]
        if len(non_none) == 5 and len(set(non_none)) == 1:
            galaxy_signal     = non_none[0]
            galaxy_signal_age = 0

        # Age out stale signal
        if galaxy_signal != 0:
            galaxy_signal_age += 1
            if galaxy_signal_age > GALAXY_SIGNAL_DECAY:
                galaxy_signal     = 0
                galaxy_signal_age = 0

        for prod in GALAXY:
            od = galaxy_ods.get(prod)
            if od is None:
                continue
            lpos = pos.get(prod, 0)
            bb, bbv, ba, bav = best(od)
            if bb is None or ba is None:
                continue
            bh_bias = GALAXY_BH_BIAS if prod == GALAXY_BH else 0
            if galaxy_signal == 1:
                tgt = min(LIMIT, 7 + bh_bias)
            elif galaxy_signal == -1:
                tgt = max(-LIMIT, -7 + bh_bias)   # BH: -4, others: -7
            else:
                tgt = bh_bias                       # BH: hold +3, others: passive MM

            delta = tgt - lpos
            if delta > 0 and ba is not None:
                qty = min(delta, bav, LIMIT - lpos)
                if qty > 0:
                    result[prod].append(Order(prod, ba, qty))
            elif delta < 0 and bb is not None:
                qty = min(-delta, bbv, LIMIT + lpos)
                if qty > 0:
                    result[prod].append(Order(prod, bb, -qty))
            elif delta == 0:
                m = mid(od)
                if m is not None:
                    passive_quote(result, prod, od, lpos, m, 2)

        # ── 4) SNACKPACK cointegration: CHOC + (−2.1259)·PIST + (−0.2320)·STRAW ─
        # Johansen cointegrating vector is stationary AR(1). Trade mean-reversion
        # on the z-score: enter when |z| > COINT_ENTER, hold until |z| < COINT_EXIT.
        coint_ods  = [state.order_depths.get(p) for p in SNACK_COINT]
        coint_mids = [mid(o) if o is not None else None for o in coint_ods]
        if all(m is not None for m in coint_mids):
            spread_now = sum(b * m for b, m in zip(COINT_BETA, coint_mids))
            spread_hist.append(spread_now)

            if len(spread_hist) >= 30:
                arr = np.array(spread_hist)
                mu    = float(np.mean(arr))
                sigma = float(np.std(arr))

                if sigma > 0:
                    zscore = (spread_now - mu) / sigma

                    # Determine target positions with hysteresis
                    if zscore > COINT_ENTER:
                        targets = _SHORT_TGTS   # spread high → short CHOC, long PIST/STRAW
                    elif zscore < -COINT_ENTER:
                        targets = _LONG_TGTS    # spread low  → long CHOC, short PIST/STRAW
                    elif abs(zscore) < COINT_EXIT:
                        targets = (0, 0, 0)     # near mean → flatten
                    else:
                        targets = None          # in between → hold current position

                    if targets is not None:
                        for prod, od_, tgt in zip(SNACK_COINT, coint_ods, targets):
                            lpos = pos.get(prod, 0)
                            delta = tgt - lpos
                            if delta == 0 or od_ is None:
                                continue
                            bb, bbv, ba, bav = best(od_)
                            if delta > 0 and ba is not None:
                                qty = min(delta, bav, LIMIT - lpos)
                                if qty > 0:
                                    result[prod].append(Order(prod, ba, qty))
                            elif delta < 0 and bb is not None:
                                qty = min(-delta, bbv, LIMIT + lpos)
                                if qty > 0:
                                    result[prod].append(Order(prod, bb, -qty))

        # ── 5) Mean-reversion MM for high negative-AR1 products ───────────────
        for prod in MR_PRODUCTS:
            if prod not in state.order_depths:
                continue
            od = state.order_depths[prod]
            bb, bbv, ba, bav = best(od)
            if bb is None or ba is None:
                continue
            spread = ba - bb
            lpos = pos.get(prod, 0)
            if spread > 2:
                buy_px  = bb + 1
                sell_px = ba - 1
                if lpos < LIMIT - 3:
                    result[prod].append(Order(prod, buy_px,  min(3, LIMIT - lpos)))
                if lpos > -(LIMIT - 3):
                    result[prod].append(Order(prod, sell_px, -min(3, LIMIT + lpos)))

        # ── 6) Fallback spread-capture MM ─────────────────────────────────────
        already = set(PEBBLES) | set(GALAXY) | set(SNACK_COINT) | set(MR_PRODUCTS)
        for prod in ALL_PRODUCTS:
            if prod in already or prod not in state.order_depths:
                continue
            od = state.order_depths[prod]
            bb, _, ba, _ = best(od)
            if bb is None or ba is None:
                continue
            spread = ba - bb
            if spread <= 2:
                continue
            lpos = pos.get(prod, 0)
            buy_px  = bb + 1
            sell_px = ba - 1
            if lpos > 5:  buy_px  = bb
            if lpos < -5: sell_px = ba
            if lpos < LIMIT - 2:
                result[prod].append(Order(prod, buy_px,  min(2, LIMIT - lpos)))
            if lpos > -(LIMIT - 2):
                result[prod].append(Order(prod, sell_px, -min(2, LIMIT + lpos)))

        trader_data = json.dumps({
            "spread_hist":       list(spread_hist),
            "galaxy_signal":     galaxy_signal,
            "galaxy_signal_age": galaxy_signal_age,
            "prev_ts":           int(state.timestamp),
        })
        return result, 0, trader_data
