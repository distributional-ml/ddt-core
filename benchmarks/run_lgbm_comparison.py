import pickle
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ddt import DDTRegressor

warnings.filterwarnings("ignore")


def calc_cce(y_true, q_lower, q_upper, target_coverage):
    empirical_coverage = np.mean((y_true >= q_lower) & (y_true <= q_upper))
    return np.abs(target_coverage - empirical_coverage)


def calc_apiw(q_lower, q_upper):
    return np.mean(q_upper - q_lower)


def calc_cvar(y_true, q_threshold):
    breaches = y_true[y_true > q_threshold]
    return np.mean(breaches) if len(breaches) > 0 else 0


def calc_hei(infer_time_sec, n_samples, model_obj):
    latency_us = (infer_time_sec / n_samples) * 1e6
    size_kb = len(pickle.dumps(model_obj)) / 1024
    return latency_us * size_kb


def main():
    print("Loading Ames Housing Dataset...")
    ames = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    X = ames.data.select_dtypes(include=[np.number]).fillna(0)
    y = ames.target.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n_test = len(y_test)

    max_depth = 10
    num_leaves = 32
    n_target_bins = 32

    # Train DDT (Single Tree)
    print("\nTraining DDT (Single Tree)...")
    ddt = DDTRegressor(max_depth=max_depth, n_target_bins=n_target_bins)
    start_time = time.time()
    ddt.fit(X_train_scaled, y_train)
    ddt_fit_time = time.time() - start_time

    start_time = time.time()
    ddt_preds = ddt.predict_quantiles(X_test_scaled, [0.05, 0.90, 0.95])
    ddt_infer_time = time.time() - start_time
    ddt_q05 = ddt_preds[0.05]
    ddt_q90 = ddt_preds[0.90]
    ddt_q95 = ddt_preds[0.95]

    ddt_metrics = [
        f"{calc_cce(y_test, ddt_q05, ddt_q95, 0.90) * 100:.2f}%",
        f"{calc_apiw(ddt_q05, ddt_q95):,.2f}",
        f"{calc_cvar(y_test, ddt_q90):,.2f}",
        f"{(ddt_infer_time / n_test) * 1e6:.2f}",
        f"{len(pickle.dumps(ddt)) / 1024:.2f}",
        f"{calc_hei(ddt_infer_time, n_test, ddt):.2f}",
    ]

    def evaluate_lgbm(n_estimators):
        lgbm_p05 = LGBMRegressor(
            objective="quantile",
            alpha=0.05,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_estimators=n_estimators,
            verbose=-1,
        )
        lgbm_p90 = LGBMRegressor(
            objective="quantile",
            alpha=0.90,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_estimators=n_estimators,
            verbose=-1,
        )
        lgbm_p95 = LGBMRegressor(
            objective="quantile",
            alpha=0.95,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=42,
            n_estimators=n_estimators,
            verbose=-1,
        )

        lgbm_p05.fit(X_train_scaled, y_train)
        lgbm_p90.fit(X_train_scaled, y_train)
        lgbm_p95.fit(X_train_scaled, y_train)

        start_time = time.time()
        lgbm_q05 = lgbm_p05.predict(X_test_scaled)
        lgbm_q90 = lgbm_p90.predict(X_test_scaled)
        lgbm_q95 = lgbm_p95.predict(X_test_scaled)
        lgbm_infer_time = time.time() - start_time

        return [
            f"{calc_cce(y_test, lgbm_q05, lgbm_q95, 0.90) * 100:.2f}%",
            f"{calc_apiw(lgbm_q05, lgbm_q95):,.2f}",
            f"{calc_cvar(y_test, lgbm_q90):,.2f}",
            f"{(lgbm_infer_time / n_test) * 1e6:.2f}",
            f"{len(pickle.dumps([lgbm_p05, lgbm_p90, lgbm_p95])) / 1024:.2f}",
            f"{calc_hei(lgbm_infer_time, n_test, [lgbm_p05, lgbm_p90, lgbm_p95]):.2f}",
        ]

    print("\nTraining LightGBM Scenario A (n_estimators=1)...")
    lgbm_a_metrics = evaluate_lgbm(n_estimators=1)

    print("\nTraining LightGBM Scenario B (n_estimators=100)...")
    lgbm_b_metrics = evaluate_lgbm(n_estimators=100)

    metrics = [
        "CCE (90% CI)",
        "APIW (90% CI)",
        "Realized CVaR (Tail > P90)",
        "Latency (us/sample)",
        "Model Size (KB)",
        "HEI",
    ]

    df_a = pd.DataFrame({"Metric": metrics, "DDT (Single Tree)": ddt_metrics, "LightGBM (1 Tree)": lgbm_a_metrics})

    df_b = pd.DataFrame({"Metric": metrics, "DDT (Single Tree)": ddt_metrics, "LightGBM (100 Trees)": lgbm_b_metrics})

    print("\n=== Scenario A - Single-Tree Baseline ===")
    print(df_a.to_string(index=False))
    print("\n=== Scenario B - Production Ensemble ===")
    print(df_b.to_string(index=False))


if __name__ == "__main__":
    main()
