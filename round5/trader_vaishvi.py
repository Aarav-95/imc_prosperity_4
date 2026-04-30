from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from collections import deque
import numpy as np
import json

LIMIT = 10

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

# Johansen cointegrating vector (normalized to CHOCOLATE=1)
SNACK_COINT  = ("SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY")
COINT_BETA   = (1.0, -2.1259, -0.2320)
COINT_SCALE  = 4
COINT_WINDOW = 200
COINT_ENTER  = 2.0
COINT_EXIT   = 0.5
_LONG_TGTS   = tuple(int(round(COINT_SCALE * b)) for b in COINT_BETA)   # (+4, -8, -1)
_SHORT_TGTS  = tuple(-t for t in _LONG_TGTS)                             # (-4, +8, +1)

# Hold max position in products with consistent directional drift all 3 days
DIRECTIONAL_HOLDS = {
    "MICROCHIP_OVAL":            -10,  # -52.7% total, all 3 days ↓, accelerating
    "UV_VISOR_AMBER":            -10,  # -31.5% total, all 3 days ↓
    "PANEL_2X4":                 +10,  # +22.1% total, all 3 days ↑ (flattest drift)
    "OXYGEN_SHAKE_GARLIC":       +10,  # +35.6% total, all 3 days ↑
    "UV_VISOR_RED":              +10,  # +16.4% total, all 3 days ↑
    "GALAXY_SOUNDS_BLACK_HOLES": +10,  # +31.4% total, all 3 days ↑
}

# Products that consistently lose on penny-flatten — skip entirely
SKIP = {
    "MICROCHIP_SQUARE",            # -83K
    "PEBBLES_M",                   # -59K  (wide spread — basket arb still runs)
    "PEBBLES_XS",                  # -52K
    "ROBOT_VACUUMING",             # -35K
    "SLEEP_POD_COTTON",            # -87K
    "SLEEP_POD_SUEDE",             # -26K
    "TRANSLATOR_ECLIPSE_CHARCOAL", # -15K
    "UV_VISOR_MAGENTA",            # -55K
    "UV_VISOR_YELLOW",             # -152K
    "GALAXY_SOUNDS_DARK_MATTER",   # -33K
    "GALAXY_SOUNDS_SOLAR_FLAMES",  # -11K
    "PANEL_4X4",                   # -79K
    "ROBOT_MOPPING",               # -23K
    "SLEEP_POD_POLYESTER",         # -49K
    "SNACKPACK_STRAWBERRY",        # -21K
    "TRANSLATOR_GRAPHITE_MIST",    # -39K
    "TRANSLATOR_SPACE_GRAY",       # -30K
    "PANEL_2X2",                   # -8K
    "ROBOT_DISHES",                # -5K
    "ROBOT_LAUNDRY",               # -4K
    "UV_VISOR_ORANGE",             # -5K
}

# Products handled by dedicated strategies — excluded from penny-flatten
_SPECIAL = set(PEBBLES) | set(DIRECTIONAL_HOLDS) | set(SNACK_COINT) | SKIP


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
    bb, _, ba, _ = best(od)
    if bb is None or ba is None:
        return
    want_buy  = int(np.floor(fair_val)) - 1
    want_sell = int(np.ceil(fair_val))  + 1
    if bb < want_buy < ba and lpos < LIMIT:
        result[prod].append(Order(prod, want_buy,  min(size, LIMIT - lpos)))
    if bb < want_sell < ba and lpos > -LIMIT:
        result[prod].append(Order(prod, want_sell, -min(size, LIMIT + lpos)))


def penny_take(od, fair, edge, prod, pos):
    """Phase 1: sweep book levels priced better than our inner quote."""
    orders = []
    for ask_px in sorted(od.sell_orders.keys()):
        if ask_px >= fair - edge:
            break
        avail = -od.sell_orders[ask_px]
        qty = min(avail, LIMIT - pos)
        if qty > 0:
            orders.append(Order(prod, ask_px, qty))
            pos += qty
    for bid_px in sorted(od.buy_orders.keys(), reverse=True):
        if bid_px <= fair + edge:
            break
        avail = od.buy_orders[bid_px]
        qty = min(avail, LIMIT + pos)
        if qty > 0:
            orders.append(Order(prod, bid_px, -qty))
            pos -= qty
    return orders, pos


def penny_clear(fair, prod, pos):
    """Phase 2: flatten inventory at fair value."""
    if pos == 0:
        return [], pos
    return [Order(prod, int(round(fair)), -pos)], 0


def penny_make(inner_bid, inner_ask, prod, pos, best_bid, best_ask):
    """Phase 3: passive quotes at penny-improved prices, skipping the side
    that would worsen an existing directional position."""
    orders = []
    skip_bid = pos > 0 and inner_bid >= best_bid
    skip_ask = pos < 0 and inner_ask <= best_ask
    if LIMIT - pos > 0 and not skip_bid:
        orders.append(Order(prod, inner_bid,  LIMIT - pos))
    if LIMIT + pos > 0 and not skip_ask:
        orders.append(Order(prod, inner_ask, -(LIMIT + pos)))
    return orders


class Trader:
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        spread_hist = deque(saved.get("spread_hist", []), maxlen=COINT_WINDOW)

        prev_ts = saved.get("prev_ts", -1)
        if int(state.timestamp) < int(prev_ts):
            spread_hist.clear()

        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}
        pos = state.position

        # ── 0) Directional holds: reach target ASAP and hold ─────────────────
        for prod, tgt in DIRECTIONAL_HOLDS.items():
            od = state.order_depths.get(prod)
            if od is None:
                continue
            lpos = pos.get(prod, 0)
            delta = tgt - lpos
            if delta == 0:
                continue
            bb, bbv, ba, bav = best(od)
            if delta > 0 and ba is not None:
                qty = min(delta, bav, LIMIT - lpos)
                if qty > 0:
                    result[prod].append(Order(prod, ba, qty))
            elif delta < 0 and bb is not None:
                qty = min(-delta, bbv, LIMIT + lpos)
                if qty > 0:
                    result[prod].append(Order(prod, bb, -qty))

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

        # ── 2) PEBBLES: passive quoting at implied fair ───────────────────────
        for leg in PEBBLES:
            if leg not in fair:
                continue
            od = state.order_depths.get(leg)
            if od is None:
                continue
            passive_quote(result, leg, od, pos.get(leg, 0), fair[leg], 5)

        # ── 3) SNACKPACK cointegration: CHOC + (−2.1259)·PIST + (−0.2320)·STRAW
        coint_ods  = [state.order_depths.get(p) for p in SNACK_COINT]
        coint_mids = [mid(o) if o is not None else None for o in coint_ods]
        if all(m is not None for m in coint_mids):
            spread_now = sum(b * m for b, m in zip(COINT_BETA, coint_mids))
            spread_hist.append(spread_now)

            if len(spread_hist) >= 30:
                arr   = np.array(spread_hist)
                mu_s  = float(np.mean(arr))
                sigma = float(np.std(arr))

                if sigma > 0:
                    zscore = (spread_now - mu_s) / sigma
                    if zscore > COINT_ENTER:
                        targets = _SHORT_TGTS
                    elif zscore < -COINT_ENTER:
                        targets = _LONG_TGTS
                    elif abs(zscore) < COINT_EXIT:
                        targets = (0, 0, 0)
                    else:
                        targets = None

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

        # ── 4) Penny-flatten all remaining products ───────────────────────────
        for prod, od in state.order_depths.items():
            if prod in _SPECIAL:
                continue
            if not od.buy_orders or not od.sell_orders:
                continue

            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            inner_bid = bb + 1
            inner_ask = ba - 1
            fair_val  = (inner_bid + inner_ask) / 2.0
            edge      = (inner_ask - inner_bid) / 2.0

            lpos = pos.get(prod, 0)
            orders = []

            take_o, lpos = penny_take(od, fair_val, edge, prod, lpos)
            orders.extend(take_o)
            clear_o, lpos = penny_clear(fair_val, prod, lpos)
            orders.extend(clear_o)
            make_o = penny_make(inner_bid, inner_ask, prod, lpos, bb, ba)
            orders.extend(make_o)

            if orders:
                result[prod].extend(orders)

        trader_data = json.dumps({
            "spread_hist": list(spread_hist),
            "prev_ts":     int(state.timestamp),
        })
        return result, 0, trader_data
