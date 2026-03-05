"""
Thompson Sampling Dynamic Pricing Bandit.

Models each price point as a Beta-Bernoulli bandit arm where the reward
signal is a margin-weighted conversion indicator. Guardrails enforce
regulatory and business constraints at inference time.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Price Arm
# ---------------------------------------------------------------------------

@dataclass
class PriceArm:
    """A single discrete price level modeled as a Beta-Bernoulli bandit arm."""
    price: float
    nadac_cost: float
    alpha: float = 1.0          # Beta distribution alpha (successes + prior)
    beta: float = 1.0           # Beta distribution beta  (failures  + prior)
    total_pulls: int = 0
    total_reward: float = 0.0

    @property
    def margin(self) -> float:
        """Gross margin as a fraction [0, 1]."""
        if self.price <= 0:
            return 0.0
        return (self.price - self.nadac_cost) / self.price

    @property
    def mean_conversion(self) -> float:
        """Posterior mean of Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    def sample_reward(self, rng: np.random.Generator) -> float:
        """
        Draw a Thompson sample: conversion probability × margin.
        This blends exploration (Beta sample) with exploitation (margin).
        """
        conversion_sample = rng.beta(self.alpha, self.beta)
        return float(conversion_sample * self.margin)

    def update(self, converted: bool, weight: float = 1.0) -> None:
        """
        Conjugate Beta-Bernoulli posterior update.

        Args:
            converted: Whether the customer purchased at this price.
            weight:    Optional observation weight (e.g., units sold).
        """
        self.total_pulls += 1
        if converted:
            self.alpha += weight
            self.total_reward += self.price * weight
        else:
            self.beta += weight

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PriceArm":
        return cls(**d)


# ---------------------------------------------------------------------------
# Guardrail Service
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    passed: bool
    reason: str
    fallback_price: Optional[float] = None


class GuardrailService:
    """
    Business and regulatory guardrails for pharmacy pricing.

    Rules:
        1. Margin floor: price must yield >= 15% gross margin.
        2. Price ceiling: price must not exceed list_price * 1.30.
        3. Price must be positive.
    """

    MARGIN_FLOOR: float = 0.15
    CEILING_MULTIPLIER: float = 1.30

    def validate(
        self,
        price: float,
        nadac_cost: float,
        list_price: float,
    ) -> GuardrailResult:
        """Return GuardrailResult indicating whether price passes all rules."""
        fallback = nadac_cost / (1.0 - self.MARGIN_FLOOR)

        if price <= 0:
            return GuardrailResult(
                passed=False,
                reason="price must be positive",
                fallback_price=round(fallback, 4),
            )

        actual_margin = (price - nadac_cost) / price
        if actual_margin < self.MARGIN_FLOOR:
            return GuardrailResult(
                passed=False,
                reason=f"margin {actual_margin:.2%} < floor {self.MARGIN_FLOOR:.2%}",
                fallback_price=round(fallback, 4),
            )

        ceiling = list_price * self.CEILING_MULTIPLIER
        if price > ceiling:
            return GuardrailResult(
                passed=False,
                reason=f"price ${price:.2f} > ceiling ${ceiling:.2f}",
                fallback_price=round(ceiling, 4),
            )

        return GuardrailResult(passed=True, reason="ok")


# ---------------------------------------------------------------------------
# Product Bandit
# ---------------------------------------------------------------------------

class ProductBandit:
    """
    Multi-armed bandit for a single product's price optimization.

    Builds N_ARMS discrete price arms evenly spaced from the margin-floor
    price to list_price * 1.30.
    """

    N_ARMS: int = 10

    def __init__(
        self,
        product_id: int,
        nadac_cost: float,
        list_price: float,
        margin_floor: float = 0.15,
        random_state: int = 42,
    ):
        self.product_id = product_id
        self.nadac_cost = nadac_cost
        self.list_price = list_price
        self.margin_floor = margin_floor
        self._rng = np.random.default_rng(random_state)

        self.arms: List[PriceArm] = self._build_arms()
        self.guardrail = GuardrailService()

    def _build_arms(self) -> List[PriceArm]:
        """Create N_ARMS price levels from floor to ceiling."""
        price_floor = self.nadac_cost / (1.0 - self.margin_floor)
        price_ceil  = self.list_price * GuardrailService.CEILING_MULTIPLIER

        prices = np.linspace(price_floor, price_ceil, self.N_ARMS)
        return [
            PriceArm(price=round(float(p), 4), nadac_cost=self.nadac_cost)
            for p in prices
        ]

    def select_arm(self) -> Tuple[int, PriceArm]:
        """
        Thompson Sampling arm selection.

        Samples θ_i ~ Beta(α_i, β_i) × margin_i for each arm and picks
        the arm with the highest sample.
        """
        samples = [arm.sample_reward(self._rng) for arm in self.arms]
        best_idx = int(np.argmax(samples))
        return best_idx, self.arms[best_idx]

    def update(self, arm_idx: int, converted: bool, units: float = 1.0) -> None:
        """Update the posterior of the selected arm."""
        self.arms[arm_idx].update(converted=converted, weight=units)

    def best_arm(self) -> Tuple[int, PriceArm]:
        """Return the arm with the highest posterior mean reward."""
        rewards = [a.mean_conversion * a.margin for a in self.arms]
        best_idx = int(np.argmax(rewards))
        return best_idx, self.arms[best_idx]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "product_id": self.product_id,
            "nadac_cost": self.nadac_cost,
            "list_price": self.list_price,
            "margin_floor": self.margin_floor,
            "arms": [arm.to_dict() for arm in self.arms],
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, s: str) -> "ProductBandit":
        d = json.loads(s)
        bandit = cls(
            product_id=d["product_id"],
            nadac_cost=d["nadac_cost"],
            list_price=d["list_price"],
            margin_floor=d["margin_floor"],
        )
        bandit.arms = [PriceArm.from_dict(a) for a in d["arms"]]
        return bandit


# ---------------------------------------------------------------------------
# Dynamic Pricing Engine (facade)
# ---------------------------------------------------------------------------

class DynamicPricingEngine:
    """
    High-level facade used by the API layer.

    Wraps ProductBandit instances per product with guardrail enforcement
    and latency instrumentation.
    """

    def __init__(self):
        self._bandits: Dict[int, ProductBandit] = {}
        self._guardrail = GuardrailService()

    def register_product(
        self,
        product_id: int,
        nadac_cost: float,
        list_price: float,
        margin_floor: float = 0.15,
    ) -> None:
        """Register a product and initialize its bandit."""
        self._bandits[product_id] = ProductBandit(
            product_id=product_id,
            nadac_cost=nadac_cost,
            list_price=list_price,
            margin_floor=margin_floor,
        )
        logger.debug(f"Registered product {product_id}")

    def get_price(
        self,
        product_id: int,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Return a pricing recommendation for the given product.

        Returns a dict with:
            recommended_price  float
            margin_pct         float
            guardrail_fired    bool
            latency_ms         float
            arm_index          int
        """
        t0 = time.perf_counter()

        if product_id not in self._bandits:
            raise KeyError(f"Product {product_id} not registered.")

        bandit = self._bandits[product_id]
        arm_idx, arm = bandit.select_arm()

        # Guardrail check
        gr = self._guardrail.validate(
            price=arm.price,
            nadac_cost=bandit.nadac_cost,
            list_price=bandit.list_price,
        )

        if gr.passed:
            recommended_price = arm.price
            guardrail_fired = False
        else:
            recommended_price = gr.fallback_price
            guardrail_fired = True
            logger.warning(
                f"Guardrail fired for product {product_id}: {gr.reason}. "
                f"Using fallback ${recommended_price:.2f}"
            )

        margin_pct = (recommended_price - bandit.nadac_cost) / recommended_price

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "product_id": product_id,
            "recommended_price": round(recommended_price, 4),
            "margin_pct": round(margin_pct, 4),
            "guardrail_fired": guardrail_fired,
            "guardrail_reason": "" if gr.passed else gr.reason,
            "arm_index": arm_idx,
            "latency_ms": round(latency_ms, 3),
        }

    def record_outcome(
        self,
        product_id: int,
        arm_index: int,
        converted: bool,
        units: float = 1.0,
    ) -> None:
        """Update the bandit posterior based on observed outcome."""
        if product_id not in self._bandits:
            raise KeyError(f"Product {product_id} not registered.")
        self._bandits[product_id].update(arm_index, converted=converted, units=units)

    def get_bandit(self, product_id: int) -> ProductBandit:
        return self._bandits[product_id]


# ---------------------------------------------------------------------------
# Simulation utility
# ---------------------------------------------------------------------------

def simulate_bandit(
    product_id: int,
    nadac_cost: float,
    list_price: float,
    n_rounds: int = 5_000,
    true_optimal_price: Optional[float] = None,
    random_state: int = 42,
) -> dict:
    """
    Run a Thompson Sampling simulation and compute cumulative regret.

    Returns:
        dict with keys: rounds, cumulative_regret, arm_pulls, final_posteriors
    """
    rng = np.random.default_rng(random_state)

    bandit = ProductBandit(
        product_id=product_id,
        nadac_cost=nadac_cost,
        list_price=list_price,
        random_state=random_state,
    )

    if true_optimal_price is None:
        # Use arm with highest margin as oracle
        true_optimal_idx = max(range(len(bandit.arms)), key=lambda i: bandit.arms[i].margin)
    else:
        diffs = [abs(a.price - true_optimal_price) for a in bandit.arms]
        true_optimal_idx = int(np.argmin(diffs))

    oracle_reward = bandit.arms[true_optimal_idx].margin

    cumulative_regret: List[float] = []
    total_regret = 0.0

    for round_idx in range(n_rounds):
        arm_idx, arm = bandit.select_arm()

        # Simulate conversion: logistic function of margin attractiveness
        # Higher price → lower conversion
        price_ratio = arm.price / list_price
        true_conversion = max(0.02, min(0.95, 0.60 * np.exp(-1.5 * (price_ratio - 0.85))))
        converted = bool(rng.random() < true_conversion)

        bandit.update(arm_idx, converted=converted, units=1.0)

        # Instant regret = oracle_reward - actual_reward
        actual_reward = arm.margin if converted else 0.0
        instant_regret = oracle_reward - actual_reward
        total_regret += instant_regret
        cumulative_regret.append(total_regret)

    arm_pulls = [arm.total_pulls for arm in bandit.arms]

    final_posteriors = [
        {
            "arm_idx": i,
            "price": arm.price,
            "margin": round(arm.margin, 4),
            "alpha": round(arm.alpha, 2),
            "beta": round(arm.beta, 2),
            "mean_conversion": round(arm.mean_conversion, 4),
            "pulls": arm.total_pulls,
        }
        for i, arm in enumerate(bandit.arms)
    ]

    return {
        "product_id": product_id,
        "n_rounds": n_rounds,
        "total_regret": round(total_regret, 4),
        "avg_regret_per_round": round(total_regret / n_rounds, 6),
        "cumulative_regret": cumulative_regret,
        "arm_pulls": arm_pulls,
        "final_posteriors": final_posteriors,
        "best_arm_idx": bandit.best_arm()[0],
        "best_arm_price": bandit.best_arm()[1].price,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Thompson Sampling Bandit Simulation ===\n")

    # Example: generic product with nadac=$15, list=$21.43 (30% margin)
    NADAC = 15.0
    LIST  = 21.43

    results = simulate_bandit(
        product_id=7,
        nadac_cost=NADAC,
        list_price=LIST,
        n_rounds=5_000,
        random_state=42,
    )

    print(f"Product 7  |  NADAC=${NADAC:.2f}  List=${LIST:.2f}")
    print(f"Rounds: {results['n_rounds']:,}")
    print(f"Total regret: {results['total_regret']:.2f}")
    print(f"Avg regret/round: {results['avg_regret_per_round']:.4f}")
    print(f"Best arm → idx={results['best_arm_idx']}  price=${results['best_arm_price']:.2f}")
    print("\nArm Posterior Summary:")
    print(f"{'Idx':>4}  {'Price':>8}  {'Margin':>8}  {'Alpha':>8}  {'Beta':>8}  {'MeanConv':>10}  {'Pulls':>6}")
    print("-" * 60)
    for arm in results["final_posteriors"]:
        print(
            f"{arm['arm_idx']:>4}  ${arm['price']:>7.2f}  "
            f"{arm['margin']:>7.2%}  {arm['alpha']:>8.1f}  {arm['beta']:>8.1f}  "
            f"{arm['mean_conversion']:>10.4f}  {arm['pulls']:>6}"
        )

    # Test JSON round-trip
    print("\n--- Testing JSON serialization ---")
    bandit = ProductBandit(product_id=7, nadac_cost=NADAC, list_price=LIST)
    json_str = bandit.to_json()
    bandit2 = ProductBandit.from_json(json_str)
    print(f"Serialized and deserialized OK. {len(bandit2.arms)} arms restored.")

    # Test DynamicPricingEngine
    print("\n--- DynamicPricingEngine demo ---")
    engine = DynamicPricingEngine()
    engine.register_product(7, nadac_cost=NADAC, list_price=LIST)
    for _ in range(3):
        resp = engine.get_price(7)
        print(f"  → ${resp['recommended_price']:.2f}  "
              f"margin={resp['margin_pct']:.2%}  "
              f"guardrail={resp['guardrail_fired']}  "
              f"latency={resp['latency_ms']:.2f}ms")
