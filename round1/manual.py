from __future__ import annotations

def clearing_price(bids: list[tuple[float, int]], asks: list[tuple[float, int]]) -> tuple[float | None, int]:
    """
    Calculate the clearing price for an order book.

    Returns:
        (clearing_price, traded_volume)  — or (None, 0) if no trade is possible
    """
    candidate_prices = sorted(set(p for p, _ in bids) | set(p for p, _ in asks))

    best_price = None
    best_volume = 0

    for price in candidate_prices:
        demand = sum(vol for bp, vol in bids if bp >= price)
        supply = sum(vol for ap, vol in asks if ap <= price)
        traded = min(demand, supply)

        if traded > best_volume or (traded == best_volume and price is not None and (best_price is None or price > best_price)):
            best_volume = traded
            best_price = price

    return best_price, best_volume


def compute_fill(bid_price: int, qty: int, original_bids: list[tuple[float, int]],
                 asks: list[tuple[float, int]], cp: float) -> int:
    """
    Compute how many units YOU get filled, given that:
      - You submitted (bid_price, qty)
      - The clearing price is cp
      - You are LAST in time priority at your price level
    """
    if bid_price < cp:
        return 0

    supply = sum(v for p, v in asks if p <= cp)
    remaining = supply

    # Walk price levels from highest to lowest (price priority)
    all_levels = sorted(set(p for p, _ in original_bids) | {bid_price}, reverse=True)

    our_fill = 0
    for level in all_levels:
        if level < cp:
            break

        # Existing orders at this level get filled first (time priority)
        existing_vol = sum(v for p, v in original_bids if p == level)
        filled_existing = min(existing_vol, remaining)
        remaining -= filled_existing

        # Our order at this level (if any) gets filled last
        if level == bid_price:
            our_fill = min(qty, remaining)
            remaining -= our_fill

    return our_fill


def find_optimal_order(bids: list[tuple[float, int]], asks: list[tuple[float, int]],
                       buyback_price: float, fee_per_unit: float = 0.0,
                       max_qty: int = 10_000_000) -> dict:
    """
    Find the optimal limit buy order to maximize profit from the guaranteed buyback.

    For each candidate bid price, we analytically determine the quantity
    thresholds where the clearing price changes, ensuring we never miss
    the optimum.
    """
    all_prices = sorted(set(p for p, _ in bids) | set(p for p, _ in asks))
    if not all_prices:
        return None

    price_lo = min(all_prices) - 2
    price_hi = max(max(all_prices), int(buyback_price)) + 1
    candidate_prices = range(price_lo, price_hi)

    best = {"order_price": None, "order_qty": 0, "clearing_price": None,
            "fill": 0, "profit": 0.0}

    # Precompute cumulative demand/supply at each price level for speed
    all_book_prices = sorted(set(p for p, _ in bids) | set(p for p, _ in asks))

    for bid_price in candidate_prices:
        # For a given bid_price, as we increase qty from 0 to max_qty,
        # the clearing price can only stay the same or increase (more demand).
        # We find the exact qty where the clearing price transitions between levels.
        #
        # Strategy: for each possible clearing price cp, compute the range of
        # qty values that produce that cp, then compute the best profit in that range.

        for target_cp in all_book_prices:
            if bid_price < target_cp:
                continue  # we wouldn't get filled
            margin = buyback_price - target_cp - fee_per_unit
            if margin <= 0:
                continue  # no profit at this clearing price

            # What qty range produces clearing price = target_cp?
            # We need: at target_cp, traded volume (with our order) >= traded at all other prices
            # AND target_cp is the highest price among those that maximize volume
            #
            # Instead of solving analytically (complex with tie-breaking),
            # binary search for the max qty that keeps cp at or below target_cp.

            # First check: does qty=1 already push cp above target_cp?
            new_bids_1 = bids + [(bid_price, 1)]
            cp_1, _ = clearing_price(new_bids_1, asks)
            if cp_1 is not None and cp_1 > target_cp and bid_price >= cp_1:
                # Even qty=1 at this bid_price gives a higher cp
                # Check if that higher cp is still profitable
                pass

            # Binary search: find max qty where clearing price == target_cp
            lo, hi = 1, max_qty
            max_qty_at_target = 0

            while lo <= hi:
                mid = (lo + hi) // 2
                new_bids_mid = bids + [(bid_price, mid)]
                cp_mid, _ = clearing_price(new_bids_mid, asks)

                if cp_mid is not None and cp_mid <= target_cp:
                    max_qty_at_target = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if max_qty_at_target == 0:
                continue

            # The best qty at this target_cp is the one that maximizes fill
            # Since fill increases with qty (up to supply), use max_qty_at_target
            qty = max_qty_at_target
            new_bids = bids + [(bid_price, qty)]
            cp, _ = clearing_price(new_bids, asks)

            if cp is None or bid_price < cp:
                continue

            fill = compute_fill(bid_price, qty, bids, asks, cp)
            if fill <= 0:
                continue

            profit = (buyback_price - cp - fee_per_unit) * fill

            if profit > best["profit"] or (
                profit == best["profit"] and fill > best["fill"]
            ):
                best = {
                    "order_price": bid_price,
                    "order_qty": qty,
                    "clearing_price": cp,
                    "fill": fill,
                    "profit": profit,
                }

    return best if best["profit"] > 0 else None


def print_price_volume_curve(bids, asks, label=""):
    """Print the demand/supply/traded curve at each price level."""
    all_prices = sorted(set(p for p, _ in bids) | set(p for p, _ in asks))
    cp, vol = clearing_price(bids, asks)

    if label:
        print(f"\n{'═' * 50}")
        print(f"  {label}")
        print(f"{'═' * 50}")
    print(f"{'Price':>8}  {'Demand':>8}  {'Supply':>8}  {'Traded':>8}")
    print("-" * 42)
    for p in all_prices:
        demand = sum(v for bp, v in bids if bp >= p)
        supply = sum(v for ap, v in asks if ap <= p)
        traded = min(demand, supply)
        marker = " ◀ clearing" if p == cp else ""
        print(f"{p:>8}  {demand:>8}  {supply:>8}  {traded:>8}{marker}")


# ══════════════════════════════════════════════════════════════════
#  ORDER BOOK DATA — Fill these in with the actual auction data
# ══════════════════════════════════════════════════════════════════

# DRYLAND_FLAX — Buyback: 30 per unit, no fees
flax_bids = [
    (30, 30000),
    (29, 5000),
    (28, 12000),
    (27, 28000)
]
flax_asks = [
    (28, 40000),
    (31, 20000),
    (32, 20000),
    (33, 30000)
]

# EMBER_MUSHROOM — Buyback: 20 per unit, fee: 0.10 per unit
mushroom_bids = [
    (20, 43000),
    (19, 17000),
    (18, 6000),
    (17, 5000),
    (16, 10000),
    (15, 5000),
    (14, 10000),
    (13, 7000),
]
mushroom_asks = [
    (12, 20000),
    (13, 25000),
    (14, 35000),
    (15, 6000),
    (16, 5000),
    (18, 10000),
    (19, 12000),
]

if __name__ == "__main__":

    products = [
        ("DRYLAND_FLAX",    flax_bids,     flax_asks,     30, 0.00),
        ("EMBER_MUSHROOM",  mushroom_bids, mushroom_asks, 20, 0.1),
    ]

    for name, bids, asks, buyback, fee in products:
        if not bids and not asks:
            print(f"\n⚠  {name}: No order book data — skipping")
            continue

        print_price_volume_curve(bids, asks, label=f"{name}  (buyback={buyback}, fee={fee})")

        result = find_optimal_order(bids, asks, buyback, fee)

        if result:
            print(f"\n  ✅ OPTIMAL ORDER:  BID {result['order_qty']}x @ {result['order_price']}")
            print(f"     Clearing price:  {result['clearing_price']}")
            print(f"     Your fill:       {result['fill']} units")
            print(f"     Gross revenue:   {buyback} × {result['fill']} = {buyback * result['fill']}")
            print(f"     Cost:            {result['clearing_price']} × {result['fill']} = {result['clearing_price'] * result['fill']}")
            if fee > 0:
                print(f"     Fees:            {fee} × {result['fill']} = {fee * result['fill']:.2f}")
            print(f"     💰 NET PROFIT:   {result['profit']:.2f}")
        else:
            print(f"\n  ❌ No profitable order exists for {name}")

        print()
