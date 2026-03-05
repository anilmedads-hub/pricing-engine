"""
Double Machine Learning (DML) price elasticity estimator.

Uses cross-fitting with XGBRegressor to residualize treatment and outcome,
then runs OLS on the residuals to obtain an unbiased elasticity estimate
per user segment. No econml dependency required.

Reference: Chernozhukov et al. (2018) "Double/Debiased Machine Learning"
"""

import logging
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ElasticityResult:
    segment: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_obs: int
    r2_residual: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.segment:<16}] ε = {self.point_estimate:+.3f}  "
            f"95% CI [{self.ci_lower:+.3f}, {self.ci_upper:+.3f}]  "
            f"n={self.n_obs:,}"
        )


# ---------------------------------------------------------------------------
# Core DML estimator
# ---------------------------------------------------------------------------

class DMLPriceElasticityEstimator:
    """
    Cross-fit Double Machine Learning estimator for price elasticity.

    Treatment  T = log(price / list_price)   [price discount / premium]
    Outcome    Y = log1p(units_sold)

    Controls   X = [week_of_year, day_of_week, is_holiday_week,
                    stock_level, competitor_price_ratio, product_id dummies]

    Algorithm:
        1. Split data into K folds.
        2. For each fold: fit nuisance models E[Y|X] and E[T|X] on train,
           predict residuals on held-out fold.
        3. OLS: Ỹ ~ T̃  (residuals)  → elasticity coefficient.
        4. Bootstrap residual OLS for confidence intervals.
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_bootstrap: int = 200,
        xgb_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.results_: Dict[str, ElasticityResult] = {}
        self._scaler = StandardScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(df: pd.DataFrame) -> np.ndarray:
        """Construct numeric feature matrix from transaction dataframe."""
        feats = pd.DataFrame({
            "week_sin": np.sin(2 * np.pi * df["week_of_year"] / 52),
            "week_cos": np.cos(2 * np.pi * df["week_of_year"] / 52),
            "dow_sin":  np.sin(2 * np.pi * df["day_of_week"] / 7),
            "dow_cos":  np.cos(2 * np.pi * df["day_of_week"] / 7),
            "is_holiday": df["is_holiday_week"].astype(float),
            "stock_level": df["stock_level"],
            "competitor_ratio": np.log(df["competitor_price"] / df["list_price"]).clip(-1, 1),
            "log_nadac": np.log(df["nadac_cost"]),
        })
        # Product dummies (50 products → 49 dummies)
        prod_dummies = pd.get_dummies(df["product_id"], prefix="prod", drop_first=True)
        return pd.concat([feats, prod_dummies], axis=1).values.astype(np.float32)

    # ------------------------------------------------------------------
    # Cross-fitting
    # ------------------------------------------------------------------

    def _cross_fit_residuals(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return out-of-fold residuals (T_tilde, Y_tilde) via K-fold cross-fitting.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for DML estimation") from exc

        n = len(Y)
        T_tilde = np.zeros(n)
        Y_tilde = np.zeros(n)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"  Cross-fitting fold {fold_idx + 1}/{self.n_folds} "
                        f"(train={len(train_idx):,}, val={len(val_idx):,})")

            X_tr, X_val = X[train_idx], X[val_idx]
            T_tr, T_val = T[train_idx], T[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            # Nuisance model for T
            m_T = XGBRegressor(**self.xgb_params)
            m_T.fit(X_tr, T_tr)
            T_tilde[val_idx] = T_val - m_T.predict(X_val)

            # Nuisance model for Y
            m_Y = XGBRegressor(**self.xgb_params)
            m_Y.fit(X_tr, Y_tr)
            Y_tilde[val_idx] = Y_val - m_Y.predict(X_val)

        return T_tilde, Y_tilde

    # ------------------------------------------------------------------
    # OLS on residuals with bootstrap CI
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_elasticity(T_tilde: np.ndarray, Y_tilde: np.ndarray) -> float:
        """Point estimate: β = (T̃ᵀT̃)⁻¹ T̃ᵀỸ"""
        denom = np.dot(T_tilde, T_tilde)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(T_tilde, Y_tilde) / denom)

    def _bootstrap_ci(
        self, T_tilde: np.ndarray, Y_tilde: np.ndarray, alpha: float = 0.05
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(self.random_state)
        n = len(T_tilde)
        boots = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boots.append(self._ols_elasticity(T_tilde[idx], Y_tilde[idx]))
        lo = float(np.percentile(boots, 100 * alpha / 2))
        hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        return lo, hi

    @staticmethod
    def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DMLPriceElasticityEstimator":
        """Fit DML elasticity estimator per user segment."""
        logger.info(f"Fitting DML estimator on {len(df):,} rows across segments...")

        segments = df["user_segment"].unique()
        self.results_ = {}

        for seg in sorted(segments):
            seg_df = df[df["user_segment"] == seg].copy().reset_index(drop=True)
            logger.info(f"Segment '{seg}': {len(seg_df):,} observations")

            X = self._build_features(seg_df)
            T = np.log(seg_df["price"] / seg_df["list_price"]).values.astype(np.float32)
            Y = np.log1p(seg_df["units_sold"]).values.astype(np.float32)

            T_tilde, Y_tilde = self._cross_fit_residuals(X, T, Y)

            point = self._ols_elasticity(T_tilde, Y_tilde)
            ci_lo, ci_hi = self._bootstrap_ci(T_tilde, Y_tilde)

            # R² of the residual stage
            Y_hat = point * T_tilde
            r2 = self._r2(Y_tilde, Y_hat)

            result = ElasticityResult(
                segment=seg,
                point_estimate=round(point, 4),
                ci_lower=round(ci_lo, 4),
                ci_upper=round(ci_hi, 4),
                n_obs=len(seg_df),
                r2_residual=round(r2, 4),
            )
            self.results_[seg] = result
            logger.info(f"  {result}")

        self._fitted = True
        self._log_to_mlflow()
        return self

    def predict_elasticity(self, segment: str) -> ElasticityResult:
        """Return fitted elasticity for the given segment."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        if segment not in self.results_:
            raise KeyError(f"Unknown segment '{segment}'. Known: {list(self.results_)}")
        return self.results_[segment]

    def optimal_price(
        self,
        segment: str,
        nadac_cost: float,
        list_price: float,
        margin_floor: float = 0.15,
    ) -> float:
        """
        Dorfman-Steiner optimal price:
            P* = ε / (ε + 1) * nadac_cost

        Clipped so that margin >= margin_floor.
        Also clipped to not exceed list_price * 1.30 (price ceiling).
        """
        eps = self.predict_elasticity(segment).point_estimate
        if abs(eps + 1) < 1e-6:
            eps = eps - 1e-4  # avoid division by zero at unit elasticity

        raw_optimal = (eps / (eps + 1.0)) * nadac_cost

        # Guardrail: margin floor
        price_floor = nadac_cost / (1.0 - margin_floor)
        # Guardrail: price ceiling
        price_ceil = list_price * 1.30

        optimal = float(np.clip(raw_optimal, price_floor, price_ceil))
        return round(optimal, 4)

    def results_table(self) -> pd.DataFrame:
        """Return results as a formatted DataFrame."""
        rows = [asdict(r) for r in self.results_.values()]
        return pd.DataFrame(rows).set_index("segment")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the fitted estimator."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Estimator saved → {p}")

    @classmethod
    def load(cls, path: str) -> "DMLPriceElasticityEstimator":
        """Load a pickled estimator."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Estimator loaded ← {path}")
        return obj

    # ------------------------------------------------------------------
    # MLflow (optional)
    # ------------------------------------------------------------------

    def _log_to_mlflow(self) -> None:
        try:
            import mlflow
            with mlflow.start_run(run_name="dml_price_elasticity"):
                for seg, result in self.results_.items():
                    mlflow.log_metric(f"elasticity_{seg}", result.point_estimate)
                    mlflow.log_metric(f"r2_{seg}", result.r2_residual)
                mlflow.log_param("n_folds", self.n_folds)
                mlflow.log_param("n_bootstrap", self.n_bootstrap)
                mlflow.log_param("xgb_n_estimators", self.xgb_params.get("n_estimators"))
            logger.info("Results logged to MLflow.")
        except Exception:
            logger.debug("MLflow not available — skipping logging.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    data_path = Path(__file__).resolve().parents[2] / "data" / "transactions.parquet"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run  python src/data/generate_synthetic_data.py  first.")
        sys.exit(1)

    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows.")

    estimator = DMLPriceElasticityEstimator(n_folds=5, n_bootstrap=100)
    estimator.fit(df)

    print("\n=== Elasticity Results ===")
    print(estimator.results_table().to_string())

    print("\n=== Optimal Price Examples ===")
    for seg in estimator.results_:
        p_opt = estimator.optimal_price(
            segment=seg,
            nadac_cost=10.0,
            list_price=14.29,
            margin_floor=0.15,
        )
        print(f"  {seg:<16} optimal price for nadac=$10.00 → ${p_opt:.2f}")

    model_path = Path(__file__).resolve().parents[2] / "data" / "dml_model.pkl"
    estimator.save(str(model_path))
    print(f"\nModel saved to {model_path}")
