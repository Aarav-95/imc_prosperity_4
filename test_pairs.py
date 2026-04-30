import pandas as pd
import glob
import sys

files = sorted(glob.glob("data/Round 5/prices_round_5_day_*.csv"))
df_list = []
for f in files:
    df_list.append(pd.read_csv(f, sep=";"))

df = pd.concat(df_list)

pairs = [
    ("PEBBLES_XS", "UV_VISOR_AMBER"),
    ("MICROCHIP_SQUARE", "SLEEP_POD_SUEDE"),
    ("GALAXY_SOUNDS_BLACK_HOLES", "OXYGEN_SHAKE_GARLIC")
]

for p1, p2 in pairs:
    df1 = df[df["product"] == p1].set_index(["day", "timestamp"])
    df2 = df[df["product"] == p2].set_index(["day", "timestamp"])
    
    merged = df1.join(df2, lsuffix="_1", rsuffix="_2").dropna()
    
    # Spread costs
    spread_cost_1 = (merged["ask_price_1"] - merged["bid_price_1"]).mean()
    spread_cost_2 = (merged["ask_price_2"] - merged["bid_price_2"]).mean()
    total_cost = spread_cost_1 + spread_cost_2
    
    # Calculate price spread (price1 - price2)
    # Using mid price
    mid1 = (merged["bid_price_1"] + merged["ask_price_1"]) / 2
    mid2 = (merged["bid_price_2"] + merged["ask_price_2"]) / 2
    
    price_diff = mid1 - mid2
    diff_std = price_diff.std()
    
    # Calculate correlation just to verify
    corr = mid1.corr(mid2)
    
    print(f"Pair: {p1} & {p2}")
    print(f"  Correlation: {corr:.3f}")
    print(f"  Spread StdDev (Edge size): {diff_std:.1f} ticks")
    print(f"  Avg Crossing Cost (Both products): {total_cost:.1f} ticks")
    if diff_std > total_cost:
         print("  -> VIABLE: The spread variance easily overcomes the bid-ask crossing cost.")
    else:
         print("  -> NOT VIABLE: The bid-ask crossing cost destroys the edge.")
    print("")

