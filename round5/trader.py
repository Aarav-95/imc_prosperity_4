from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

class Trader:
    def __init__(self):
        self.position_limits = {
            "GALAXY_SOUNDS_DARK_MATTER": 10,
            "GALAXY_SOUNDS_BLACK_HOLES": 10,
            "GALAXY_SOUNDS_PLANETARY_RINGS": 10,
            "GALAXY_SOUNDS_SOLAR_WINDS": 10,
            "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
            "SLEEP_POD_SUEDE": 10,
            "SLEEP_POD_LAMB_WOOL": 10,
            "SLEEP_POD_POLYESTER": 10,
            "SLEEP_POD_NYLON": 10,
            "SLEEP_POD_COTTON": 10,
            "MICROCHIP_CIRCLE": 10,
            "MICROCHIP_OVAL": 10,
            "MICROCHIP_SQUARE": 10,
            "MICROCHIP_RECTANGLE": 10,
            "MICROCHIP_TRIANGLE": 10,
            "PEBBLES_XS": 10,
            "PEBBLES_S": 10,
            "PEBBLES_M": 10,
            "PEBBLES_L": 10,
            "PEBBLES_XL": 10,
            "ROBOT_VACUUMING": 10,
            "ROBOT_MOPPING": 10,
            "ROBOT_DISHES": 10,
            "ROBOT_LAUNDRY": 10,
            "ROBOT_IRONING": 10,
            "UV_VISOR_YELLOW": 10,
            "UV_VISOR_AMBER": 10,
            "UV_VISOR_ORANGE": 10,
            "UV_VISOR_RED": 10,
            "UV_VISOR_MAGENTA": 10,
            "TRANSLATOR_SPACE_GRAY": 10,
            "TRANSLATOR_ASTRO_BLACK": 10,
            "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
            "TRANSLATOR_GRAPHITE_MIST": 10,
            "TRANSLATOR_VOID_BLUE": 10,
            "PANEL_1X2": 10,
            "PANEL_2X2": 10,
            "PANEL_1X4": 10,
            "PANEL_2X4": 10,
            "PANEL_4X4": 10,
            "OXYGEN_SHAKE_MORNING_BREATH": 10,
            "OXYGEN_SHAKE_EVENING_BREATH": 10,
            "OXYGEN_SHAKE_MINT": 10,
            "OXYGEN_SHAKE_CHOCOLATE": 10,
            "OXYGEN_SHAKE_GARLIC": 10,
            "SNACKPACK_CHOCOLATE": 10,
            "SNACKPACK_VANILLA": 10,
            "SNACKPACK_PISTACHIO": 10,
            "SNACKPACK_STRAWBERRY": 10,
            "SNACKPACK_RASPBERRY": 10,
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_data = {}

        history = trader_data.get("history", {})

        for product, limit in self.position_limits.items():
            od = state.order_depths.get(product)
            if not od or not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            # Basic market making is bad if spread is constantly 1 tick
            if spread < 2:
                continue

            hist_mid = history.get(product, mid)
            fair = 0.5 * hist_mid + 0.5 * mid
            history[product] = fair

            pos = state.position.get(product, 0)

            # Inventory skew: adjust fair value against our position to reduce risk
            skewed_fair = fair - (0.5 * pos)

            edge = 1.0

            my_bid = int(math.floor(skewed_fair - edge))
            my_ask = int(math.ceil(skewed_fair + edge))

            # Only penny the best bid/ask, don't take liquidity unnecessarily
            my_bid = min(my_bid, best_bid + 1)
            my_ask = max(my_ask, best_ask - 1)

            orders = []
            buy_cap = limit - pos
            sell_cap = limit + pos

            if buy_cap > 0:
                orders.append(Order(product, my_bid, buy_cap))
            if sell_cap > 0:
                orders.append(Order(product, my_ask, -sell_cap))

            if orders:
                result[product] = orders

        traderData = json.dumps({"history": history})
        return result, conversions, traderData
