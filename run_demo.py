"""
End-to-end demo for the Real-Time Dynamic Pricing & Causal Impact Engine.

Steps:
    1. Generate synthetic pharmacy transaction data
    2. Fit the Double ML causal price elasticity model
    3. Run Thompson Sampling bandit simulation for product_id=7
    4. Make 3 live pricing requests via DynamicPricingEngine
"""

import sys
import os
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np


def separator(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# STEP 1: Generate Data
# ---------------------------------------------------------------------------

separator("STEP 1: GENERATING DATA")

from src.data.generate_synthetic_data import main as generate_data

transactions, catalog = generate_data()

print(f"\nTransactions shape : {transactions.shape}")
print(f"Catalog shape      : {catalog.shape}")
print(f"Date range         : {transactions['timestamp'].min().date()} → "
      f"{transactions['timestamp'].max().date()}")
print(f"Unique products    : {transactions['product_id'].nunique()}")
print(f"Unique users       : {transactions['user_id'].nunique():,}")
print(f"Segments           : {transactions['user_segment'].value_counts().to_dict()}")
print(f"\nSample rows:")
print(
    transactions[["transaction_id","product_id","price","list_price",
                  "units_sold","revenue","user_segment"]]
    .head(5)
    .to_string(index=False)
)


# ---------------------------------------------------------------------------
# STEP 2: Fit Causal Model
# ---------------------------------------------------------------------------

separator("STEP 2: FITTING DOUBLE ML CAUSAL ELASTICITY MODEL")

from src.causal.dml_estimator import DMLPriceElasticityEstimator

data_path = Path(__file__).resolve().parent / "data" / "transactions.parquet"
df = pd.read_parquet(data_path)
print(f"Loaded {len(df):,} transactions from parquet.\n")

estimator = DMLPriceElasticityEstimator(n_folds=5, n_bootstrap=100, random_state=42)
estimator.fit(df)

print("\n--- Elasticity Results Table ---")
results_df = estimator.results_table()
print(results_df.to_string())

print("\n--- Optimal Price Examples (nadac=$25.00, list=$35.71) ---")
nadac_ex   = 25.0
list_ex    = 35.71
print(f"{'Segment':<18} {'Elasticity':>12} {'Optimal Price':>14} {'Implied Margin':>15}")
print("-" * 62)
for seg in sorted(estimator.results_):
    result = estimator.results_[seg]
    p_opt  = estimator.optimal_price(seg, nadac_cost=nadac_ex, list_price=list_ex)
    implied_margin = (p_opt - nadac_ex) / p_opt if p_opt > 0 else 0
    print(f"{seg:<18} {result.point_estimate:>+12.4f} "
          f"${p_opt:>13.2f} {implied_margin:>14.2%}")

# Save model
model_path = Path(__file__).resolve().parent / "data" / "dml_model.pkl"
estimator.save(str(model_path))
print(f"\nDML model saved → {model_path}")


# ---------------------------------------------------------------------------
# STEP 3: Bandit Simulation
# ---------------------------------------------------------------------------

separator("STEP 3: THOMPSON SAMPLING BANDIT SIMULATION (product_id=7, 5000 rounds)")

from src.bandit.thompson_sampling import simulate_bandit, ProductBandit

# Use product_id=7 from catalog
prod7_row = catalog[catalog["product_id"] == 7].iloc[0]
nadac_7   = float(prod7_row["nadac_cost"])
list_7    = float(prod7_row["list_price"])

print(f"Product 7: NADAC=${nadac_7:.2f}  List=${list_7:.2f}")

results = simulate_bandit(
    product_id=7,
    nadac_cost=nadac_7,
    list_price=list_7,
    n_rounds=5_000,
    random_state=42,
)

print(f"\nSimulation Results:")
print(f"  Rounds            : {results['n_rounds']:,}")
print(f"  Total regret      : {results['total_regret']:.4f}")
print(f"  Avg regret/round  : {results['avg_regret_per_round']:.6f}")
print(f"  Best arm index    : {results['best_arm_idx']}")
print(f"  Best arm price    : ${results['best_arm_price']:.2f}")

print(f"\n{'Arm':>4}  {'Price':>8}  {'Margin':>8}  {'Alpha':>8}  "
      f"{'Beta':>8}  {'MeanConv':>10}  {'Pulls':>6}")
print("-" * 60)
for arm in results["final_posteriors"]:
    marker = " ◄ BEST" if arm["arm_idx"] == results["best_arm_idx"] else ""
    print(
        f"{arm['arm_idx']:>4}  ${arm['price']:>7.2f}  "
        f"{arm['margin']:>7.2%}  {arm['alpha']:>8.1f}  {arm['beta']:>8.1f}  "
        f"{arm['mean_conversion']:>10.4f}  {arm['pulls']:>6}{marker}"
    )

# Show convergence: regret at 25%, 50%, 75%, 100% of rounds
cr = results["cumulative_regret"]
n  = results["n_rounds"]
print(f"\nCumulative regret at milestones:")
for pct in [0.25, 0.50, 0.75, 1.00]:
    idx = min(int(pct * n) - 1, n - 1)
    print(f"  Round {idx+1:>5,} ({pct:.0%}): {cr[idx]:.2f}")


# ---------------------------------------------------------------------------
# STEP 4: Live Price Requests via DynamicPricingEngine
# ---------------------------------------------------------------------------

separator("STEP 4: LIVE PRICING REQUESTS (DynamicPricingEngine)")

from src.bandit.thompson_sampling import DynamicPricingEngine
import json

engine = DynamicPricingEngine()

# Register a handful of products from the catalog
for _, row in catalog.head(10).iterrows():
    engine.register_product(
        product_id=int(row["product_id"]),
        nadac_cost=float(row["nadac_cost"]),
        list_price=float(row["list_price"]),
        margin_floor=float(row["margin_floor"]),
    )

print("Registered products: 1 – 10\n")
print("Making 3 price requests for product_id=7...\n")

for request_num in range(1, 4):
    response = engine.get_price(product_id=7)
    print(f"Request {request_num}:")
    print(json.dumps(response, indent=4))

    # Simulate a conversion for feedback
    converted = bool(np.random.default_rng(request_num).random() > 0.4)
    engine.record_outcome(
        product_id=7,
        arm_index=response["arm_index"],
        converted=converted,
        units=1.0,
    )
    print(f"  Outcome recorded: converted={converted}\n")

print("Making 1 price request for each of products 1–5:")
print(f"{'PID':>4}  {'Price':>8}  {'Margin':>8}  {'Guardrail':>10}  {'Latency':>10}")
print("-" * 50)
for pid in range(1, 6):
    resp = engine.get_price(product_id=pid)
    print(
        f"{resp['product_id']:>4}  ${resp['recommended_price']:>7.2f}  "
        f"{resp['margin_pct']:>7.2%}  "
        f"{'YES' if resp['guardrail_fired'] else 'no':>10}  "
        f"{resp['latency_ms']:>9.3f}ms"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

separator("DEMO COMPLETE — SUMMARY")

print("Step 1 ✓  Generated 1,050,000 transactions across 50 products & 80,000 users")
print("Step 2 ✓  Fitted Double ML elasticity model with 5-fold cross-fitting per segment")
print("Step 3 ✓  Ran 5,000-round Thompson Sampling bandit for product 7")
print("Step 4 ✓  Demonstrated live pricing via DynamicPricingEngine with guardrails")
print()
print("Key outputs:")
print(f"  data/transactions.parquet   ({len(transactions):,} rows)")
print(f"  data/product_catalog.parquet ({len(catalog)} products)")
print(f"  data/dml_model.pkl           (fitted causal model)")
print()
print("See README.md for full deployment and architecture documentation.")
