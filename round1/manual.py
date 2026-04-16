from __future__ import annotations

"""
Manual Trading Auction Optimizer

Given an order book, finds the optimal limit buy order (price, quantity) to
maximize profit from the guaranteed Merchant Guild buyback.

Clearing price rules:
  1. Maximizes total traded volume
  2. Breaks ties by choosing the higher price

You are LAST in time priority at any price level you join.
"""


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
                       max_qty: int = 50000) -> dict:
    """
    Find the optimal limit buy order to maximize profit from the guaranteed buyback.

    Args:
        bids:           existing buy orders as (price, volume)
        asks:           existing sell orders as (price, volume)
        buyback_price:  Merchant Guild buyback price per unit
        fee_per_unit:   fee charged per unit traded (deducted from profit)
        max_qty:        max quantity to search over

    Returns:
        dict with keys: order_price, order_qty, clearing_price, fill, profit
        or None if no profitable order exists
    """
    all_prices = sorted(set(p for p, _ in bids) | set(p for p, _ in asks))
    if not all_prices:
        return None

    # Search over all integer prices from below the book up to just under buyback
    price_lo = min(all_prices) - 2
    price_hi = int(buyback_price)  # bidding at buyback itself yields 0 profit
    candidate_prices = range(price_lo, price_hi)

    best = {"order_price": None, "order_qty": 0, "clearing_price": None,
            "fill": 0, "profit": 0.0}

    for bid_price in candidate_prices:
        for qty in range(1, max_qty + 1):
            # Add our order to the book
            new_bids = bids + [(bid_price, qty)]
            cp, _ = clearing_price(new_bids, asks)

            if cp is None or bid_price < cp:
                continue

            # How many units do we actually get?
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


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    products = [
        ("DRYLAND_FLAX",    flax_bids,     flax_asks,     30, 0.00),
        ("EMBER_MUSHROOM",  mushroom_bids, mushroom_asks, 20, 0.10),
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
