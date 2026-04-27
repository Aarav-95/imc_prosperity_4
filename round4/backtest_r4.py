import csv
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
sys.path.insert(0, '..')
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ConversionObservation


def read_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def to_int(v) -> Optional[int]:
    f = to_float(v)
    return int(f) if f is not None else None


class BacktestEngine:
    def __init__(self, trader_module):
        self.trader = trader_module.Trader()
        self.logger = trader_module.logger
        
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
        
        self.positions: Dict[str, int] = defaultdict(int)
        self.cash: float = 0.0
        self.trader_data: str = ""
        
        self.pnl_history: List[Tuple[int, float]] = []
        self.fill_history: List[Dict] = []
        
        self.prices_data = self._load_prices()
        self.trades_data = self._load_trades()
        
    def _load_prices(self) -> Dict[int, Dict[str, Dict]]:
        data = {}
        for day in [1, 2, 3]:
            rows = read_csv(f"../ROUND_4/prices_round_4_day_{day}.csv")
            for row in rows:
                ts = to_int(row["timestamp"])
                if ts is None:
                    continue
                key = (day - 1) * 1_000_000 + ts
                if key not in data:
                    data[key] = {}
                data[key][row["product"]] = row
        return data
    
    def _load_trades(self) -> Dict[int, List[Dict]]:
        data = defaultdict(list)
        for day in [1, 2, 3]:
            rows = read_csv(f"../ROUND_4/trades_round_4_day_{day}.csv")
            for row in rows:
                ts = to_int(row.get("timestamp"))
                if ts is None:
                    continue
                key = (day - 1) * 1_000_000 + ts
                data[key].append(row)
        return data

    def _build_order_depth(self, row: Dict) -> OrderDepth:
        od = OrderDepth()
        for i in range(1, 4):
            price = to_int(row.get(f"bid_price_{i}"))
            vol = to_int(row.get(f"bid_volume_{i}"))
            if price is not None and vol is not None:
                od.buy_orders[price] = vol
        for i in range(1, 4):
            price = to_int(row.get(f"ask_price_{i}"))
            vol = to_int(row.get(f"ask_volume_{i}"))
            if price is not None and vol is not None:
                od.sell_orders[price] = -vol
        return od

    def _build_trading_state(self, timestamp: int, products: List[str], 
                              price_rows: Dict[str, Dict], market_trades: List[Dict]) -> TradingState:
        listings = {}
        order_depths = {}
        
        for product in products:
            listings[product] = Listing(product, product, "XIRECS")
            if product in price_rows:
                order_depths[product] = self._build_order_depth(price_rows[product])
            else:
                order_depths[product] = OrderDepth()
        
        mt: Dict[Symbol, List[Trade]] = defaultdict(list)
        for t in market_trades:
            symbol = t.get("symbol", "")
            if symbol in products:
                mt[symbol].append(Trade(
                    symbol=symbol,
                    price=int(to_float(t.get("price", 0)) or 0),
                    quantity=int(to_float(t.get("quantity", 0)) or 0),
                    buyer=t.get("buyer", ""),
                    seller=t.get("seller", ""),
                    timestamp=to_int(t.get("timestamp")) or 0
                ))
        
        obs = Observation({}, {})
        
        return TradingState(
            traderData=self.trader_data,
            timestamp=timestamp % 1_000_000,
            listings=listings,
            order_depths=order_depths,
            own_trades=defaultdict(list),
            market_trades=dict(mt),
            position=dict(self.positions),
            observations=obs
        )

    def _match_orders(self, orders: List[Order], order_depth: OrderDepth, 
                       position: int, limit: int) -> List[Tuple[int, int]]:
        fills = []
        for order in orders:
            if order.quantity > 0:
                remaining = order.quantity
                current_pos = position + sum(f[1] for f in fills if f[1] > 0)
                max_buy = limit - current_pos
                remaining = min(remaining, max_buy)
                
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if remaining <= 0:
                        break
                    if order.price >= ask_price:
                        available = -order_depth.sell_orders[ask_price]
                        fill_qty = min(remaining, available)
                        if fill_qty > 0:
                            fills.append((ask_price, fill_qty))
                            remaining -= fill_qty
                            
            elif order.quantity < 0:
                remaining = -order.quantity
                current_pos = position + sum(f[1] for f in fills)
                max_sell = limit + current_pos
                remaining = min(remaining, max_sell)
                
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if remaining <= 0:
                        break
                    if order.price <= bid_price:
                        available = order_depth.buy_orders[bid_price]
                        fill_qty = min(remaining, available)
                        if fill_qty > 0:
                            fills.append((bid_price, -fill_qty))
                            remaining -= fill_qty
        return fills

    def _calculate_pnl(self, price_rows: Dict[str, Dict]) -> float:
        pnl = self.cash
        for product, pos in self.positions.items():
            if product in price_rows:
                mid = to_float(price_rows[product].get("mid_price"))
                if mid is not None and mid > 0:
                    pnl += pos * mid
        return pnl

    def run_backtest(self, verbose: bool = False) -> Dict:
        products = list(self.position_limits.keys())
        timestamps = sorted(self.prices_data.keys())
        
        total_fills = 0
        fills_by_product = defaultdict(int)
        volume_by_product = defaultdict(int)
        pnl_by_product = defaultdict(float)
        
        for ts in timestamps:
            price_rows = self.prices_data[ts]
            market_trades = self.trades_data.get(ts, [])
            
            state = self._build_trading_state(ts, products, price_rows, market_trades)
            
            try:
                result, conversions, new_trader_data = self.trader.run(state)
                self.trader_data = new_trader_data
            except Exception as e:
                if verbose:
                    print(f"Error at ts={ts}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            for product in products:
                if product not in result:
                    continue
                orders = result[product]
                if not orders:
                    continue
                
                order_depth = state.order_depths.get(product, OrderDepth())
                position = self.positions[product]
                limit = self.position_limits[product]
                
                fills = self._match_orders(orders, order_depth, position, limit)
                
                for price, qty in fills:
                    self.positions[product] += qty
                    self.cash -= price * qty
                    
                    total_fills += 1
                    fills_by_product[product] += 1
                    volume_by_product[product] += abs(qty)
                    pnl_by_product[product] -= price * qty
            
            pnl = self._calculate_pnl(price_rows)
            self.pnl_history.append((ts, pnl))
        
        final_pnl = self.pnl_history[-1][1] if self.pnl_history else 0
        
        peak = 0
        max_drawdown = 0
        for ts, pnl in self.pnl_history:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "final_pnl": final_pnl,
            "total_fills": total_fills,
            "fills_by_product": dict(fills_by_product),
            "volume_by_product": dict(volume_by_product),
            "pnl_by_product": dict(pnl_by_product),
            "final_positions": dict(self.positions),
            "max_drawdown": max_drawdown,
        }

    def print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("                    BACKTEST RESULTS")
        print("=" * 70)
        
        print(f"\n{'FINAL PnL:':<25} {results['final_pnl']:>15,.2f}")
        print(f"{'Max Drawdown:':<25} {results['max_drawdown']:>15,.2f}")
        
        print(f"\n--- Fills by Product ---")
        for product in sorted(results['fills_by_product'].keys()):
            count = results['fills_by_product'][product]
            vol = results['volume_by_product'].get(product, 0)
            pos = results['final_positions'].get(product, 0)
            print(f"  {product:<20} fills={count:>6,} vol={vol:>8,} pos={pos:>5}")
        
        print("=" * 70)


if __name__ == "__main__":
    import trader as trader_module
    
    print(f"Running Round 4 backtest...")
    
    engine = BacktestEngine(trader_module)
    
    import io
    import sys as sys_mod
    old_stdout = sys_mod.stdout
    sys_mod.stdout = io.StringIO()
    
    results = engine.run_backtest(verbose=False)
    
    sys_mod.stdout = old_stdout
    
    engine.print_summary(results)
