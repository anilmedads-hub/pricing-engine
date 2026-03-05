"""
Synthetic pharmacy transaction data generator.

Generates 1,050,000 realistic pharmacy transactions using a structural demand
model: log(units) = elasticity * log(price/list_price) + seasonality + holiday + noise
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRANSACTIONS = 1_050_000
N_PRODUCTS = 50
N_USERS = 80_000
RANDOM_SEED = 42

SEGMENTS = {
    "price_sensitive": {"elasticity": -2.5, "share": 0.35},
    "brand_loyal":     {"elasticity": -0.6, "share": 0.20},
    "chronic_buyer":   {"elasticity": -1.1, "share": 0.30},
    "one_time":        {"elasticity": -1.8, "share": 0.15},
}

HOLIDAY_WEEKS = {1, 2, 26, 27, 52}  # New Year, July 4th, Christmas


def build_product_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Create 50-product pharmacy catalog with realistic economics."""
    product_ids = list(range(1, N_PRODUCTS + 1))

    # NADAC cost: log-uniform between $3.50 and $180
    nadac_costs = np.exp(
        rng.uniform(np.log(3.50), np.log(180.0), size=N_PRODUCTS)
    )

    # Margin by product tier (cost drives tier)
    margins = np.where(
        nadac_costs < 20, 0.40,        # generics: 40% margin
        np.where(nadac_costs < 60, 0.30, 0.22)  # mid-tier: 30%, specialty: 22%
    )

    list_prices = nadac_costs / (1.0 - margins)

    # Generate NDC codes (fake but realistic format: 5-4-2)
    ndc_codes = [
        f"{rng.integers(10000, 99999)}-{rng.integers(1000, 9999)}-{rng.integers(10, 99)}"
        for _ in product_ids
    ]

    categories = rng.choice(
        ["Generic", "Brand", "Specialty", "OTC"],
        size=N_PRODUCTS,
        p=[0.45, 0.30, 0.15, 0.10],
    )

    return pd.DataFrame({
        "product_id": product_ids,
        "ndc_code": ndc_codes,
        "category": categories,
        "nadac_cost": nadac_costs.round(4),
        "list_price": list_prices.round(4),
        "margin_floor": margins,
    })


def build_user_table(rng: np.random.Generator) -> pd.DataFrame:
    """Assign 80,000 users to segments."""
    segment_names = list(SEGMENTS.keys())
    shares = [SEGMENTS[s]["share"] for s in segment_names]

    segments = rng.choice(segment_names, size=N_USERS, p=shares)

    return pd.DataFrame({
        "user_id": range(N_USERS),
        "user_segment": segments,
    })


def simulate_transactions(
    catalog: pd.DataFrame,
    users: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate N_TRANSACTIONS rows using structural demand model.

    log(units) = elasticity * log(price/list_price) + seasonality + holiday + noise
    """
    logger.info("Sampling transaction meta-data...")

    n = N_TRANSACTIONS

    # Draw user and product IDs
    user_idx = rng.integers(0, N_USERS, size=n)
    product_idx = rng.integers(0, N_PRODUCTS, size=n)

    user_segments = users["user_segment"].values[user_idx]
    elasticities = np.array([SEGMENTS[s]["elasticity"] for s in user_segments])

    prod = catalog.iloc[product_idx].reset_index(drop=True)
    nadac_costs = prod["nadac_cost"].values
    list_prices = prod["list_price"].values
    margin_floors = prod["margin_floor"].values
    ndc_codes = prod["ndc_code"].values
    product_ids = prod["product_id"].values

    # Timestamps: random seconds across 2 years (2022-01-01 to 2023-12-31)
    start_ts = pd.Timestamp("2022-01-01").timestamp()
    end_ts   = pd.Timestamp("2023-12-31 23:59:59").timestamp()
    timestamps_unix = rng.uniform(start_ts, end_ts, size=n)
    timestamps = pd.to_datetime(timestamps_unix, unit="s").round("s")

    day_of_week  = timestamps.dayofweek          # 0=Monday
    week_of_year = timestamps.isocalendar().week.astype(int).values
    is_holiday   = np.isin(week_of_year, list(HOLIDAY_WEEKS)).astype(int)

    logger.info("Simulating pricing decisions...")

    # Dynamic pricing: random discount in [-25%, +5%] around list, clipped to floor
    discount_pct = rng.uniform(-0.25, 0.05, size=n)
    price = list_prices * (1.0 + discount_pct)
    # Clip so margin >= floor
    price_floor = nadac_costs / (1.0 - margin_floors)
    price = np.maximum(price, price_floor)

    # Competitor price: list * U(0.85, 1.10)
    competitor_price = list_prices * rng.uniform(0.85, 1.10, size=n)

    # Stock level: 0-100%
    stock_level = rng.uniform(0.20, 1.0, size=n).round(2)

    logger.info("Applying structural demand model...")

    # Seasonal component: sinusoidal with period 52 weeks
    week_float = week_of_year / 52.0
    seasonality = 0.15 * np.sin(2 * np.pi * week_float)

    # Holiday uplift
    holiday_effect = 0.25 * is_holiday

    # Demand model
    log_price_ratio = np.log(price / list_prices)
    log_units = (
        elasticities * log_price_ratio
        + seasonality
        + holiday_effect
        + rng.normal(0, 0.20, size=n)   # idiosyncratic noise
    )

    # Base units: draw from log-normal centered at ~3 units median
    base_log_units = rng.normal(1.1, 0.4, size=n)
    log_units_total = base_log_units + log_units

    units_sold = np.maximum(1, np.round(np.exp(log_units_total))).astype(int)

    revenue = (price * units_sold).round(4)
    gross_margin = ((price - nadac_costs) / price * units_sold).round(4)

    logger.info("Assembling DataFrame...")

    df = pd.DataFrame({
        "transaction_id":  np.arange(n),
        "user_id":         user_idx,
        "product_id":      product_ids,
        "ndc_code":        ndc_codes,
        "timestamp":       timestamps,
        "price":           price.round(4),
        "list_price":      list_prices.round(4),
        "nadac_cost":      nadac_costs.round(4),
        "units_sold":      units_sold,
        "revenue":         revenue,
        "gross_margin":    gross_margin,
        "user_segment":    user_segments,
        "day_of_week":     day_of_week,
        "week_of_year":    week_of_year,
        "is_holiday_week": is_holiday,
        "stock_level":     stock_level,
        "competitor_price": competitor_price.round(4),
    })

    return df


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    logger.info("Building product catalog...")
    catalog = build_product_catalog(rng)

    logger.info("Building user table...")
    users = build_user_table(rng)

    logger.info(f"Simulating {N_TRANSACTIONS:,} transactions...")
    transactions = simulate_transactions(catalog, users, rng)

    out_dir = Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    txn_path = out_dir / "transactions.parquet"
    cat_path  = out_dir / "product_catalog.parquet"

    logger.info(f"Saving transactions → {txn_path}")
    transactions.to_parquet(txn_path, index=False)

    logger.info(f"Saving product catalog → {cat_path}")
    catalog.to_parquet(cat_path, index=False)

    logger.info("Data generation complete.")
    logger.info(f"  Transactions : {len(transactions):,}")
    logger.info(f"  Products     : {len(catalog)}")
    logger.info(f"  Users        : {N_USERS:,}")
    logger.info(f"  Date range   : {transactions['timestamp'].min()} → {transactions['timestamp'].max()}")

    return transactions, catalog


if __name__ == "__main__":
    main()
