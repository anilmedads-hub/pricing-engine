"""
Airflow DAGs for the Real-Time Dynamic Pricing & Causal Impact Engine.

DAG 1: pricing_engine__weekly_dml_retrain
    Schedule: Every Sunday at 04:00 UTC
    Pipeline: refresh_nadac_costs → materialize_features → retrain_dml_model
              → validate_metrics (BranchPythonOperator)
              → promote_model  OR  rollback_model
              → notify_slack → end

DAG 2: pricing_engine__nightly_bandit_sync
    Schedule: Daily at 02:00 UTC
    Pipeline: start → [sync_bandit_posteriors, update_guardrail_cost_cache] → end

All Airflow imports are wrapped in try/except so this file is importable
even without Airflow installed (e.g., during unit tests or CI).
"""

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Airflow imports
# ---------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.utils.dates import days_ago

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.warning(
        "apache-airflow is not installed. DAG objects will not be created, "
        "but all Python callables remain importable and testable."
    )

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

DEFAULT_ARGS = {
    "owner": "pricing-team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

MODEL_DIR = Path(os.getenv("PRICING_MODEL_DIR", "/tmp/pricing_models"))
DATA_DIR  = Path(os.getenv("PRICING_DATA_DIR",  "/tmp/pricing_data"))

METRIC_IMPROVEMENT_THRESHOLD = 0.01   # challenger must show >= 1% improvement
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ---------------------------------------------------------------------------
# ─── DAG 1 CALLABLES ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def refresh_nadac_costs(**context: Any) -> None:
    """
    Pull latest NADAC (National Average Drug Acquisition Cost) data from
    the CMS public API and upsert into the product catalog.

    In production this would call:
        https://data.medicaid.gov/api/1/datastore/query/...
    Here we simulate the refresh with a stub.
    """
    logger.info("Task: refresh_nadac_costs — starting")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Simulate fetching updated costs
    try:
        import pandas as pd
        import numpy as np

        cat_path = DATA_DIR / "product_catalog.parquet"
        if cat_path.exists():
            catalog = pd.read_parquet(cat_path)
            rng = np.random.default_rng(seed=int(datetime.utcnow().timestamp()))
            # Simulate ~2% daily NADAC drift
            drift = rng.normal(1.0, 0.02, size=len(catalog))
            catalog["nadac_cost"] = (catalog["nadac_cost"] * drift).round(4)
            catalog["list_price"] = catalog["nadac_cost"] / (1.0 - catalog["margin_floor"])
            catalog.to_parquet(cat_path, index=False)
            logger.info(f"NADAC costs refreshed for {len(catalog)} products.")
        else:
            logger.warning(f"Catalog not found at {cat_path}; skipping refresh.")

    except Exception as exc:
        logger.error(f"NADAC refresh failed: {exc}", exc_info=True)
        raise

    context["ti"].xcom_push(key="nadac_refreshed_at", value=str(datetime.utcnow()))
    logger.info("Task: refresh_nadac_costs — complete")


def materialize_features(**context: Any) -> None:
    """
    Compute and cache feature matrix for DML training.

    Reads the last 90 days of transactions, runs feature engineering,
    and saves to parquet for downstream consumption.
    """
    logger.info("Task: materialize_features — starting")

    try:
        import pandas as pd

        txn_path = DATA_DIR / "transactions.parquet"
        if not txn_path.exists():
            logger.warning(f"Transactions not found at {txn_path}; generating stub.")
            return

        df = pd.read_parquet(txn_path)
        cutoff = df["timestamp"].max() - pd.Timedelta(days=90)
        df_window = df[df["timestamp"] >= cutoff].copy()

        feature_path = DATA_DIR / "training_features.parquet"
        df_window.to_parquet(feature_path, index=False)
        logger.info(
            f"Feature matrix materialized: {len(df_window):,} rows → {feature_path}"
        )
        context["ti"].xcom_push(key="feature_rows", value=len(df_window))

    except Exception as exc:
        logger.error(f"Feature materialization failed: {exc}", exc_info=True)
        raise

    logger.info("Task: materialize_features — complete")


def retrain_dml_model(**context: Any) -> None:
    """
    Fit the Double ML elasticity estimator on fresh features and
    persist the challenger model artifact.
    """
    logger.info("Task: retrain_dml_model — starting")

    try:
        import pandas as pd
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.causal.dml_estimator import DMLPriceElasticityEstimator

        feature_path = DATA_DIR / "training_features.parquet"
        if not feature_path.exists():
            feature_path = DATA_DIR / "transactions.parquet"

        if not feature_path.exists():
            logger.warning("No training data found; skipping retrain.")
            return

        df = pd.read_parquet(feature_path)
        logger.info(f"Training on {len(df):,} rows...")

        estimator = DMLPriceElasticityEstimator(n_folds=5, n_bootstrap=100)
        estimator.fit(df)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        challenger_path = MODEL_DIR / "challenger_dml_model.pkl"
        estimator.save(str(challenger_path))

        # Record aggregate metric (mean absolute elasticity as proxy)
        import numpy as np
        elasticities = [r.point_estimate for r in estimator.results_.values()]
        metric = float(np.mean([abs(e) for e in elasticities]))
        context["ti"].xcom_push(key="challenger_metric", value=metric)
        logger.info(f"Challenger metric (mean |ε|): {metric:.4f}")

    except Exception as exc:
        logger.error(f"DML retrain failed: {exc}", exc_info=True)
        raise

    logger.info("Task: retrain_dml_model — complete")


def validate_metrics(**context: Any) -> str:
    """
    BranchPythonOperator gate: compare challenger vs champion.

    Returns task_id of the next branch:
        'promote_model'   if challenger improves by >= 1%
        'rollback_model'  otherwise
    """
    logger.info("Task: validate_metrics — starting")

    ti = context["ti"]
    challenger_metric: Optional[float] = ti.xcom_pull(
        task_ids="retrain_dml_model", key="challenger_metric"
    )

    champion_metric_path = MODEL_DIR / "champion_metric.json"
    if champion_metric_path.exists():
        with open(champion_metric_path) as f:
            champion_metric = json.load(f)["metric"]
    else:
        logger.info("No champion found; auto-promoting first challenger.")
        return "promote_model"

    if challenger_metric is None:
        logger.warning("No challenger metric available; rolling back.")
        return "rollback_model"

    improvement = (challenger_metric - champion_metric) / (abs(champion_metric) + 1e-9)
    logger.info(
        f"Champion metric: {champion_metric:.4f}  "
        f"Challenger metric: {challenger_metric:.4f}  "
        f"Improvement: {improvement:+.2%}"
    )

    if improvement >= METRIC_IMPROVEMENT_THRESHOLD:
        logger.info("Challenger qualifies for promotion.")
        return "promote_model"
    else:
        logger.info("Challenger does not meet threshold; rolling back.")
        return "rollback_model"


def promote_model(**context: Any) -> None:
    """Swap challenger → champion and archive previous champion."""
    logger.info("Task: promote_model — starting")

    challenger_path = MODEL_DIR / "challenger_dml_model.pkl"
    champion_path   = MODEL_DIR / "champion_dml_model.pkl"

    if not challenger_path.exists():
        logger.warning("No challenger to promote.")
        return

    # Archive current champion
    if champion_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        archive_path = MODEL_DIR / f"archive_dml_model_{ts}.pkl"
        champion_path.rename(archive_path)
        logger.info(f"Previous champion archived → {archive_path}")

    challenger_path.rename(champion_path)
    logger.info(f"Challenger promoted → {champion_path}")

    # Update champion metric record
    ti = context["ti"]
    challenger_metric = ti.xcom_pull(
        task_ids="retrain_dml_model", key="challenger_metric"
    )
    if challenger_metric is not None:
        champion_metric_path = MODEL_DIR / "champion_metric.json"
        with open(champion_metric_path, "w") as f:
            json.dump({"metric": challenger_metric, "promoted_at": str(datetime.utcnow())}, f)

    context["ti"].xcom_push(key="promotion_status", value="promoted")
    logger.info("Task: promote_model — complete")


def rollback_model(**context: Any) -> None:
    """Delete challenger artifact and keep existing champion."""
    logger.info("Task: rollback_model — starting")

    challenger_path = MODEL_DIR / "challenger_dml_model.pkl"
    if challenger_path.exists():
        challenger_path.unlink()
        logger.info(f"Challenger removed: {challenger_path}")

    context["ti"].xcom_push(key="promotion_status", value="rolled_back")
    logger.info("Task: rollback_model — complete")


def notify_slack(**context: Any) -> None:
    """Send pipeline completion notification to Slack."""
    logger.info("Task: notify_slack — starting")

    ti = context["ti"]
    promotion_status = ti.xcom_pull(
        task_ids=["promote_model", "rollback_model"], key="promotion_status"
    )
    # xcom_pull with list returns list; flatten
    if isinstance(promotion_status, list):
        promotion_status = next((s for s in promotion_status if s), "unknown")

    dag_run = context.get("dag_run")
    run_id  = dag_run.run_id if dag_run else "local"

    message = {
        "text": (
            f":chart_with_upwards_trend: *DML Retrain Pipeline Complete*\n"
            f"Run ID: `{run_id}`\n"
            f"Status: `{promotion_status}`\n"
            f"Time: `{datetime.utcnow().isoformat()}`"
        )
    }

    if SLACK_WEBHOOK_URL:
        try:
            import urllib.request
            req = urllib.request.Request(
                SLACK_WEBHOOK_URL,
                data=json.dumps(message).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            logger.info("Slack notification sent.")
        except Exception as exc:
            logger.warning(f"Slack notification failed (non-fatal): {exc}")
    else:
        logger.info(f"Slack webhook not configured. Message would be: {message['text']}")

    logger.info("Task: notify_slack — complete")


# ---------------------------------------------------------------------------
# ─── DAG 2 CALLABLES ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


def sync_bandit_posteriors(**context: Any) -> None:
    """
    Load each product's bandit state from Redis (or local fallback),
    apply any buffered rewards accumulated since last sync, and persist
    updated posteriors back to Redis.
    """
    logger.info("Task: sync_bandit_posteriors — starting")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.bandit.thompson_sampling import ProductBandit

        # Attempt Redis connection
        try:
            import redis
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
            )
            r.ping()
            use_redis = True
            logger.info("Connected to Redis.")
        except Exception:
            use_redis = False
            logger.warning("Redis unavailable; using local file fallback.")

        cat_path = DATA_DIR / "product_catalog.parquet"
        if not cat_path.exists():
            logger.warning("Product catalog not found; skipping bandit sync.")
            return

        import pandas as pd
        catalog = pd.read_parquet(cat_path)
        synced = 0

        for _, row in catalog.iterrows():
            pid = int(row["product_id"])
            key = f"bandit:product:{pid}"

            if use_redis and r.exists(key):
                json_str = r.get(key)
                bandit = ProductBandit.from_json(json_str)
            else:
                bandit = ProductBandit(
                    product_id=pid,
                    nadac_cost=float(row["nadac_cost"]),
                    list_price=float(row["list_price"]),
                )

            # In production: apply buffered rewards from event stream here
            # bandit.update(arm_idx, converted, units) for each buffered event

            if use_redis:
                r.setex(key, timedelta(days=7), bandit.to_json())

            synced += 1

        logger.info(f"Synced {synced} product bandits.")
        context["ti"].xcom_push(key="bandits_synced", value=synced)

    except Exception as exc:
        logger.error(f"Bandit sync failed: {exc}", exc_info=True)
        raise

    logger.info("Task: sync_bandit_posteriors — complete")


def update_guardrail_cost_cache(**context: Any) -> None:
    """
    Refresh the in-memory guardrail cost cache from the latest NADAC data.

    The API layer reads this cache to enforce margin floors without hitting
    the database on every request.
    """
    logger.info("Task: update_guardrail_cost_cache — starting")

    try:
        import pandas as pd

        cat_path = DATA_DIR / "product_catalog.parquet"
        if not cat_path.exists():
            logger.warning("Product catalog not found; skipping cache update.")
            return

        catalog = pd.read_parquet(cat_path)

        # Build cost map
        cost_map = {
            int(row["product_id"]): {
                "nadac_cost": float(row["nadac_cost"]),
                "list_price": float(row["list_price"]),
                "margin_floor": float(row["margin_floor"]),
            }
            for _, row in catalog.iterrows()
        }

        # Try Redis first
        try:
            import redis
            r = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                decode_responses=True,
            )
            r.ping()
            r.setex("guardrail:cost_cache", timedelta(days=2), json.dumps(cost_map))
            logger.info(f"Cost cache written to Redis: {len(cost_map)} products.")
        except Exception:
            # Fall back to local JSON file
            cache_path = DATA_DIR / "guardrail_cost_cache.json"
            with open(cache_path, "w") as f:
                json.dump(cost_map, f, indent=2)
            logger.info(f"Cost cache written to {cache_path}: {len(cost_map)} products.")

        context["ti"].xcom_push(key="cache_size", value=len(cost_map))

    except Exception as exc:
        logger.error(f"Guardrail cache update failed: {exc}", exc_info=True)
        raise

    logger.info("Task: update_guardrail_cost_cache — complete")


# ---------------------------------------------------------------------------
# ─── DAG DEFINITIONS ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if AIRFLOW_AVAILABLE:
    # -----------------------------------------------------------------------
    # DAG 1: Weekly DML Retrain
    # -----------------------------------------------------------------------
    with DAG(
        dag_id="pricing_engine__weekly_dml_retrain",
        description="Weekly Double-ML price elasticity retraining with model promotion gate",
        default_args=DEFAULT_ARGS,
        schedule_interval="0 4 * * 0",   # Every Sunday at 04:00 UTC
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=["pricing", "causal", "ml"],
    ) as dag_weekly:

        t_refresh = PythonOperator(
            task_id="refresh_nadac_costs",
            python_callable=refresh_nadac_costs,
        )

        t_features = PythonOperator(
            task_id="materialize_features",
            python_callable=materialize_features,
        )

        t_retrain = PythonOperator(
            task_id="retrain_dml_model",
            python_callable=retrain_dml_model,
        )

        t_validate = BranchPythonOperator(
            task_id="validate_metrics",
            python_callable=validate_metrics,
        )

        t_promote = PythonOperator(
            task_id="promote_model",
            python_callable=promote_model,
        )

        t_rollback = PythonOperator(
            task_id="rollback_model",
            python_callable=rollback_model,
        )

        t_slack = PythonOperator(
            task_id="notify_slack",
            python_callable=notify_slack,
            trigger_rule="none_failed_min_one_success",
        )

        t_end = EmptyOperator(
            task_id="end",
            trigger_rule="none_failed_min_one_success",
        )

        # Pipeline wiring
        t_refresh >> t_features >> t_retrain >> t_validate
        t_validate >> t_promote >> t_slack >> t_end
        t_validate >> t_rollback >> t_slack >> t_end

    # -----------------------------------------------------------------------
    # DAG 2: Nightly Bandit Sync
    # -----------------------------------------------------------------------
    with DAG(
        dag_id="pricing_engine__nightly_bandit_sync",
        description="Nightly sync of Thompson Sampling posteriors and guardrail cost cache",
        default_args=DEFAULT_ARGS,
        schedule_interval="0 2 * * *",   # Daily at 02:00 UTC
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=["pricing", "bandit", "sync"],
    ) as dag_nightly:

        t_start = EmptyOperator(task_id="start")

        t_sync_bandits = PythonOperator(
            task_id="sync_bandit_posteriors",
            python_callable=sync_bandit_posteriors,
        )

        t_update_cache = PythonOperator(
            task_id="update_guardrail_cost_cache",
            python_callable=update_guardrail_cost_cache,
        )

        t_end_nightly = EmptyOperator(task_id="end")

        # Parallel sync tasks
        t_start >> [t_sync_bandits, t_update_cache] >> t_end_nightly


# ---------------------------------------------------------------------------
# Allow running without Airflow for local testing of callable logic
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not AIRFLOW_AVAILABLE:
        print("Airflow not installed — running callable stubs in local mode.\n")

    ctx: Dict[str, Any] = {
        "ti": type(
            "FakeTI",
            (),
            {
                "xcom_push": lambda self, key, value: print(f"  xcom_push: {key}={value}"),
                "xcom_pull": lambda self, task_ids=None, key=None: None,
            },
        )(),
        "dag_run": None,
    }

    print("=== Running DAG 1 callables (stubs) ===")
    refresh_nadac_costs(**ctx)
    materialize_features(**ctx)
    retrain_dml_model(**ctx)
    branch = validate_metrics(**ctx)
    print(f"  Branch decision: {branch}")
    if branch == "promote_model":
        promote_model(**ctx)
    else:
        rollback_model(**ctx)
    notify_slack(**ctx)

    print("\n=== Running DAG 2 callables (stubs) ===")
    sync_bandit_posteriors(**ctx)
    update_guardrail_cost_cache(**ctx)

    print("\nAll callables executed successfully.")
