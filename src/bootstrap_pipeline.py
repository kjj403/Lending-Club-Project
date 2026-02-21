import train_custom
import data_loader
import config
import preprocess_pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# -----------------------------------------------------------
# Ensemble Model (Soft Voting)
# -----------------------------------------------------------
class VotingModel:
    """다중 부스팅 모델의 예측 확률 평균(Soft Voting)을 산출하는 앙상블 클래스"""
    def __init__(self, models_info):
        self.models_info = models_info

    def predict_proba(self, X_full):
        preds = []
        valid_index = None

        for m, features in self.models_info:
            X_subset = X_full[features]

            p = m.predict_proba(X_subset)
            if p.ndim == 2:
                p = p[:, 1]

            p = pd.Series(p, index=X_subset.index)
            preds.append(p)

            if valid_index is None:
                valid_index = X_subset.index
            else:
                valid_index = valid_index.intersection(X_subset.index)

        preds = [p.loc[valid_index] for p in preds]
        avg_pred = np.mean(preds, axis=0)
        return pd.Series(avg_pred, index=valid_index)


# -----------------------------------------------------------
# 포트폴리오 성과 지표 산출 (Sharpe, Excess Return, IRR)
# -----------------------------------------------------------
def get_top_k_metrics(probs, meta_df, top_k_percent=0.15):
    """지정된 승인 임계값(Top-K) 기준의 포트폴리오 성과 측정"""
    common_idx = probs.index.intersection(meta_df.index)
    if len(common_idx) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    eval_df = meta_df.loc[common_idx].copy()
    eval_df["prob"] = probs.loc[common_idx]
    eval_df = eval_df.sort_values(by="prob", ascending=True)

    n_select = int(len(eval_df) * top_k_percent)
    if n_select < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    selection = eval_df.iloc[:n_select]

    irr_values = selection["actual_irr"].to_numpy(dtype=float)
    irr_values = irr_values[np.isfinite(irr_values)]

    excess_returns = (
        selection["actual_irr"].to_numpy(dtype=float)
        - selection["risk_free_rate"].to_numpy(dtype=float)
    )
    excess_returns = excess_returns[np.isfinite(excess_returns)]

    if len(excess_returns) < 10 or len(irr_values) < 10:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mean_excess = float(np.mean(excess_returns))
    std_excess = float(np.std(excess_returns, ddof=1))

    mean_irr = float(np.mean(irr_values))
    std_irr = float(np.std(irr_values, ddof=1))

    if std_excess < 1e-9:
        return np.nan, mean_excess, std_excess, mean_irr, std_irr

    sharpe = mean_excess / std_excess
    return float(sharpe), mean_excess, std_excess, mean_irr, std_irr


# -----------------------------------------------------------
# Benchmark 지표 산출 (모든 대출 승인 시나리오)
# -----------------------------------------------------------
def get_global_full_lc_benchmark_metrics(df_full):
    if df_full is None or len(df_full) == 0:
        return np.nan, np.nan, np.nan

    excess = (
        df_full["actual_irr"].to_numpy(dtype=float)
        - df_full["risk_free_rate"].to_numpy(dtype=float)
    )
    excess = excess[np.isfinite(excess)]

    if len(excess) < 10:
        return np.nan, np.nan, np.nan

    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1))
    if std_excess < 1e-9:
        return np.nan, mean_excess, std_excess

    return float(mean_excess / std_excess), mean_excess, std_excess


def get_global_full_lc_benchmark_irr_metrics(df_full):
    if df_full is None or len(df_full) == 0:
        return np.nan, np.nan

    irr = df_full["actual_irr"].to_numpy(dtype=float)
    irr = irr[np.isfinite(irr)]

    if len(irr) < 10:
        return np.nan, np.nan

    mean_irr = float(np.mean(irr))
    std_irr = float(np.std(irr, ddof=1))
    return mean_irr, std_irr


# -----------------------------------------------------------
# Time-based Data Splitting (Out-of-Time Test Setup)
# -----------------------------------------------------------
def _parse_issue_d(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(df["issue_d"], errors="coerce")
    return dt


def _fixed_test_pool_indices(issue_dt: pd.Series, train_ratio=0.6, val_ratio=0.2):
    """가장 최근 데이터를 테스트 셋(OOT)으로 고정"""
    sorted_idx = issue_dt.dropna().sort_values().index.to_numpy()
    n = len(sorted_idx)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    n_test = max(1, int(n * test_ratio))
    if n_test >= n:
        n_test = max(1, n - 1)

    test_idx = pd.Index(sorted_idx[-n_test:])
    trainval_pool_idx = pd.Index(sorted_idx[:-n_test])
    return trainval_pool_idx, test_idx


def _stratified_train_val_indices(trainval_pool_idx: pd.Index, y: pd.Series, train_ratio=0.6, val_ratio=0.2, seed=42):
    """시뮬레이션을 위한 학습/검증 세트 층화 무작위 추출(Stratified Random Split)"""
    val_size_within = val_ratio / (train_ratio + val_ratio)
    y_pool = y.loc[trainval_pool_idx]

    try:
        train_idx, val_idx = train_test_split(
            trainval_pool_idx.to_numpy(),
            test_size=val_size_within,
            random_state=seed,
            stratify=y_pool.to_numpy(),
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            trainval_pool_idx.to_numpy(),
            test_size=val_size_within,
            random_state=seed,
            stratify=None,
        )

    return pd.Index(train_idx), pd.Index(val_idx)


def _build_xy_meta(df):
    """학습 변수(X), 타겟(y), 가중치(w), 그리고 메타데이터(meta) 분리"""
    leakage_cols = [
        "actual_irr", "roi_pct", "risk_free_rate", "int_rate_spread",
        "duration_years", "weight_loss", "weight_profit",
        "last_fico_range_high", "last_fico_range_low", "last_pymnt_amnt",
        "recoveries", "collection_recovery_fee", "total_rec_prncp", "total_rec_int",
    ]
    finance_cols = set(preprocess_pipeline.FINANCE_COLS)
    meta_cols = {"target", "sample_weight"}

    exclude_cols = set(leakage_cols) | finance_cols | meta_cols
    if "term" in exclude_cols:
        exclude_cols.remove("term")

    features = [c for c in df.columns if c not in exclude_cols]

    X = df[features].copy()
    cat_cols = X.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        X[col] = X[col].cat.codes
    X = X.select_dtypes(include=["number"])

    y = df["target"].copy()
    if "sample_weight" in df.columns:
        w = df["sample_weight"].copy()
    else:
        w = pd.Series(np.ones(len(df)), index=df.index)

    valid_meta_cols = [c for c in list(finance_cols) + leakage_cols if c in df.columns]
    meta = df[valid_meta_cols].copy()
    meta = meta.loc[:, ~meta.columns.duplicated()]

    return X, y, w, meta


def _build_fixed_model(model_type: str):
    """Optuna를 통해 도출된 최적 하이퍼파라미터가 적용된 모델 객체 생성"""
    if model_type == "XGBoost":
        return XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.08639735585703821,
            subsample=0.8542507117746058,
            colsample_bytree=0.7941646321539888,
            min_child_weight=3,
            n_jobs=-1,
            random_state=config.SEED,
            tree_method="hist",
            enable_categorical=True,
        )
    if model_type == "LightGBM":
        return LGBMClassifier(
            n_estimators=704,
            num_leaves=52,
            learning_rate=0.07645118408763842,
            min_child_samples=98,
            n_jobs=-1,
            random_state=config.SEED,
            verbose=-1,
        )
    if model_type == "CatBoost":
        return CatBoostClassifier(
            iterations=975,
            depth=7,
            learning_rate=0.09933892161118743,
            l2_leaf_reg=6.792153404846416,
            random_seed=config.SEED,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _fit_model(model, model_type, X_train, y_train, w_train, X_val, y_val):
    """비용 민감 가중치(Sample Weight)를 반영한 모형 학습"""
    fit_params = {"sample_weight": w_train}
    if model_type == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_params)
    elif model_type == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss", **fit_params)
    elif model_type == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, **fit_params)
    return model


# -----------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------
def main():
    # 논문에 명시된 시뮬레이션 반복 횟수(100회) 및 임계값(15%) 준수
    N_ITER = 100 
    TOP_K_RATIO = 0.15

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.BASE_DIR, "../reports/figures", f"monte_carlo_{now_str}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("\n" + "=" * 60)
    print("🚀 [Monte Carlo Simulation] Model Robustness Evaluation")
    print(f"   Iterations: {N_ITER} | Threshold: {TOP_K_RATIO:.2f} | Stability Score Evaluated")
    print("=" * 60)

    print("📥 [Step 1] 파생 데이터 및 훈련 가중치 로드...")
    try:
        df = data_loader.prepare_data_with_weights()
        df = df.reset_index(drop=True)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    print("\n✂️ [Step 2] Feature Selection (Top 30 핵심 변수 고정)...")
    selected_cols_by_model = {}
    for m_name in ["XGBoost", "LightGBM", "CatBoost"]:
        tmp_model, _, _, tmp_X_test, _ = train_custom.train_and_split(
            df,
            model_type=m_name,
            feature_list=None,
            use_optuna=False,
            select_top_n=30,
        )
        selected_cols_by_model[m_name] = tmp_X_test.columns.tolist()
        print(f"✅ {m_name} 고정 변수: {len(selected_cols_by_model[m_name])}개 확보")

    X_all, y_all, w_all, meta_all = _build_xy_meta(df)
    issue_dt = _parse_issue_d(df)
    valid_time_mask = issue_dt.notna()
    X_all = X_all.loc[valid_time_mask]
    y_all = y_all.loc[valid_time_mask]
    w_all = w_all.loc[valid_time_mask]
    meta_all = meta_all.loc[valid_time_mask]
    issue_dt = issue_dt.loc[valid_time_mask]

    train_ratio = float(getattr(config, "TIME_TRAIN_RATIO", 0.6))
    val_ratio = float(getattr(config, "TIME_VAL_RATIO", 0.2))

    trainval_pool_idx, test_idx = _fixed_test_pool_indices(
        issue_dt=issue_dt,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    X_test_fixed = X_all.loc[test_idx]
    meta_test_fixed = meta_all.loc[test_idx]

    global_bench_sr, global_bench_ex_mean, global_bench_ex_std = get_global_full_lc_benchmark_metrics(meta_test_fixed)
    global_bench_irr_mean, global_bench_irr_std = get_global_full_lc_benchmark_irr_metrics(meta_test_fixed)
    print(f"✅ Benchmark (Buy-All Portfolio): SR = {global_bench_sr:.4f}")

    results = {
        "XGBoost": [], "LightGBM": [], "CatBoost": [],
        "Ensemble(Cat+LGBM)": [], "Benchmark": []
    }
    excess_mean_results = {
        "XGBoost": [], "LightGBM": [], "CatBoost": [],
        "Ensemble(Cat+LGBM)": [], "Benchmark": []
    }
    excess_std_results = {
        "XGBoost": [], "LightGBM": [], "CatBoost": [],
        "Ensemble(Cat+LGBM)": [], "Benchmark": []
    }
    irr_mean_results = {
        "XGBoost": [], "LightGBM": [], "CatBoost": [],
        "Ensemble(Cat+LGBM)": [], "Benchmark": []
    }
    irr_std_results = {
        "XGBoost": [], "LightGBM": [], "CatBoost": [],
        "Ensemble(Cat+LGBM)": [], "Benchmark": []
    }

    final_model_params = {}

    print(f"\n🔥 [Step 3] Monte Carlo 시뮬레이션 수행 ({N_ITER} Iterations)...")
    for i in tqdm(range(N_ITER), desc="Running Simulation"):
        try:
            train_idx, val_idx = _stratified_train_val_indices(
                trainval_pool_idx=trainval_pool_idx,
                y=y_all,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=42 + i,
            )

            X_train_base = X_all.loc[train_idx]
            y_train_base = y_all.loc[train_idx]
            w_train_base = w_all.loc[train_idx]

            X_val_fixed = X_all.loc[val_idx]
            y_val_fixed = y_all.loc[val_idx]

            current_models = {}

            for m_name in ["XGBoost", "LightGBM", "CatBoost"]:
                model = _build_fixed_model(m_name)

                feats = selected_cols_by_model[m_name]
                X_tr = X_train_base[feats]
                y_tr = y_train_base
                w_tr = w_train_base

                X_v = X_val_fixed[feats]
                y_v = y_val_fixed
                X_te = X_test_fixed[feats]

                model = _fit_model(model, m_name, X_tr, y_tr, w_tr, X_v, y_v)

                if m_name not in final_model_params:
                    try:
                        all_params = model.get_params()
                        filtered_params = {
                            k: v for k, v in all_params.items() if k in [
                                "n_estimators", "learning_rate", "max_depth", "num_leaves",
                                "depth", "iterations", "min_child_samples", "min_child_weight",
                                "subsample", "colsample_bytree", "l2_leaf_reg"
                            ]
                        }
                        final_model_params[m_name] = filtered_params
                    except Exception:
                        final_model_params[m_name] = "Params extraction failed"

                probs = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index)

                sr, ex_mean, ex_std, irr_mean, irr_std = get_top_k_metrics(
                    probs=probs,
                    meta_df=meta_test_fixed,
                    top_k_percent=TOP_K_RATIO,
                )

                results[m_name].append(sr)
                excess_mean_results[m_name].append(ex_mean)
                excess_std_results[m_name].append(ex_std)
                irr_mean_results[m_name].append(irr_mean)
                irr_std_results[m_name].append(irr_std)

                current_models[m_name] = {"model": model, "features": feats}

            ensemble = VotingModel([
                (current_models["CatBoost"]["model"], current_models["CatBoost"]["features"]),
                (current_models["LightGBM"]["model"], current_models["LightGBM"]["features"]),
            ])
            prob_ens = ensemble.predict_proba(X_test_fixed)

            sr_ens, ex_mean_ens, ex_std_ens, irr_mean_ens, irr_std_ens = get_top_k_metrics(
                probs=prob_ens,
                meta_df=meta_test_fixed,
                top_k_percent=TOP_K_RATIO,
            )

            results["Ensemble(Cat+LGBM)"].append(sr_ens)
            excess_mean_results["Ensemble(Cat+LGBM)"].append(ex_mean_ens)
            excess_std_results["Ensemble(Cat+LGBM)"].append(ex_std_ens)
            irr_mean_results["Ensemble(Cat+LGBM)"].append(irr_mean_ens)
            irr_std_results["Ensemble(Cat+LGBM)"].append(irr_std_ens)

            results["Benchmark"].append(global_bench_sr)
            excess_mean_results["Benchmark"].append(global_bench_ex_mean)
            excess_std_results["Benchmark"].append(global_bench_ex_std)
            irr_mean_results["Benchmark"].append(global_bench_irr_mean)
            irr_std_results["Benchmark"].append(global_bench_irr_std)

        except Exception as e:
            print(f"⚠️ 시뮬레이션 {i+1}회차 오류 발생: {e}")
            continue

    res_df = pd.DataFrame(results)
    irr_mean_df = pd.DataFrame(irr_mean_results)
    irr_std_df = pd.DataFrame(irr_std_results)

    print("\n" + "=" * 130)
    print("📊 [Final Report] Monte Carlo Simulation Results")
    print("👉 Stability Score = Mean - (Std * 0.5)")
    print("-" * 130)
    print(
        f"{'Model':<20} | {'Score':<8} | {'Mean SR':<10} | {'Std Dev':<10} | "
        f"{'Mean IRR':<10} | {'Std IRR':<10} | {'Min':<8} | {'Max':<8}"
    )
    print("-" * 130)

    summary_stats = []
    for col in results.keys():
        vals = pd.to_numeric(res_df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]

        if len(finite) == 0:
            mean_val = np.nan
            std_val = np.nan
            min_val = np.nan
            max_val = np.nan
            stability_score = np.nan
        else:
            mean_val = float(np.mean(finite))
            std_val = float(np.std(finite))
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            stability_score = mean_val - (std_val * 0.5)

        irr_m_vals = pd.to_numeric(irr_mean_df[col], errors="coerce").to_numpy(dtype=float)
        irr_m_finite = irr_m_vals[np.isfinite(irr_m_vals)]

        mean_irr = float(np.mean(irr_m_finite)) if len(irr_m_finite) > 0 else np.nan
        std_irr = float(np.std(irr_m_finite)) if len(irr_m_finite) > 0 else np.nan

        summary_stats.append({
            "model": col,
            "score": stability_score,
            "mean": mean_val,
            "std": std_val,
            "mean_irr": mean_irr,
            "std_irr": std_irr,
            "min": min_val,
            "max": max_val,
        })

    summary_stats.sort(key=lambda x: np.nan_to_num(x["score"], nan=-1e18), reverse=True)

    for stat in summary_stats:
        print(
            f"{stat['model']:<20} | {stat['score']:.4f}   | {stat['mean']:.4f}     | "
            f"{stat['std']:.4f}     | {stat['mean_irr']:.4f}    | {stat['std_irr']:.4f}    | "
            f"{stat['min']:.4f}   | {stat['max']:.4f}"
        )

    print("=" * 130)

    winner = summary_stats[0]["model"]
    print(f"\n🏆 Final Recommendation: [{winner}]")
    print(
        f"   사유: 수익률(평균 Sharpe {summary_stats[0]['mean']:.4f}), "
        f"방어력(최소 Sharpe {summary_stats[0]['min']:.4f}), "
        f"절대 수익(평균 IRR {summary_stats[0]['mean_irr']:.4f}) 종합 검토 결과 최우수."
    )

    print("\n" + "=" * 90)
    print("📋 [Applied Hyperparameters]")
    print("-" * 90)
    for m_name, params in final_model_params.items():
        print(f"🔹 {m_name}:")
        if isinstance(params, dict):
            print(f"   {params}")
        else:
            print(f"   {params}")
    print("=" * 90)

    # (이하 CSV 및 시각화 Plot 저장 로직 동일)
    runs_csv = os.path.join(save_dir, "sharpe_runs.csv")
    res_df.to_csv(runs_csv, index=False)

    summary_df = pd.DataFrame(summary_stats)
    summary_csv = os.path.join(save_dir, "sharpe_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    excess_mean_df = pd.DataFrame(excess_mean_results)
    excess_std_df = pd.DataFrame(excess_std_results)
    excess_mean_runs_csv = os.path.join(save_dir, "excess_mean_runs.csv")
    excess_std_runs_csv = os.path.join(save_dir, "excess_std_runs.csv")
    excess_mean_df.to_csv(excess_mean_runs_csv, index=False)
    excess_std_df.to_csv(excess_std_runs_csv, index=False)

    excess_summary_rows = []
    for m in excess_mean_df.columns:
        s_mean = excess_mean_df[m].dropna()
        s_std = excess_std_df[m].dropna()
        excess_summary_rows.append({
            "model": m,
            "mean_excess_mean": np.mean(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_excess_std": np.std(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_excess_min": np.min(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_excess_max": np.max(s_mean) if len(s_mean) > 0 else np.nan,
            "std_excess_mean": np.mean(s_std) if len(s_std) > 0 else np.nan,
            "std_excess_std": np.std(s_std) if len(s_std) > 0 else np.nan,
        })
    pd.DataFrame(excess_summary_rows).to_csv(os.path.join(save_dir, "excess_summary.csv"), index=False)

    irr_mean_runs_csv = os.path.join(save_dir, "irr_mean_runs.csv")
    irr_std_runs_csv = os.path.join(save_dir, "irr_std_runs.csv")
    irr_mean_df.to_csv(irr_mean_runs_csv, index=False)
    irr_std_df.to_csv(irr_std_runs_csv, index=False)

    irr_summary_rows = []
    for m in irr_mean_df.columns:
        s_mean = irr_mean_df[m].dropna()
        s_std = irr_std_df[m].dropna()
        irr_summary_rows.append({
            "model": m,
            "mean_irr_mean": np.mean(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_irr_std": np.std(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_irr_min": np.min(s_mean) if len(s_mean) > 0 else np.nan,
            "mean_irr_max": np.max(s_mean) if len(s_mean) > 0 else np.nan,
            "std_irr_mean": np.mean(s_std) if len(s_std) > 0 else np.nan,
            "std_irr_std": np.std(s_std) if len(s_std) > 0 else np.nan,
        })
    pd.DataFrame(irr_summary_rows).to_csv(os.path.join(save_dir, "irr_summary.csv"), index=False)

    print(f"✅ 성과 지표 산출 결과가 {save_dir} 디렉토리에 저장되었습니다.")

    # -----------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 7))
    bench_series = res_df["Benchmark"].dropna()
    if len(bench_series) >= 2 and float(np.std(bench_series)) > 1e-12:
        sns.kdeplot(bench_series, fill=True, color="grey", alpha=0.3, label="Benchmark (Market Average)")
    elif len(bench_series) >= 1:
        plt.axvline(float(bench_series.iloc[0]), color="grey", alpha=0.7, label="Benchmark (Market Average)")

    for stat in summary_stats:
        m = stat["model"]
        if m == "Benchmark":
            continue
        s = res_df[m].dropna()
        if len(s) >= 2 and float(np.std(s)) > 1e-12:
            sns.kdeplot(s, fill=False, linewidth=2, label=f"{m} (Score: {stat['score']:.3f})")

    plt.title(f"Monte Carlo Simulation ({N_ITER} runs): Sharpe Ratio Distribution")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "monte_carlo_distribution.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    res_long = res_df.melt(var_name="Model", value_name="Sharpe Ratio").dropna()
    order_list = [stat["model"] for stat in summary_stats]
    palette_map = dict(zip(order_list, sns.color_palette("viridis", n_colors=len(order_list))))

    ax = sns.boxplot(
        data=res_long, x="Model", y="Sharpe Ratio", order=order_list,
        hue="Model", palette=palette_map, dodge=False, showmeans=True
    )
    sns.stripplot(
        data=res_long, x="Model", y="Sharpe Ratio", order=order_list,
        hue="Model", palette=palette_map, dodge=False,
        alpha=0.45, jitter=0.22, size=2.8, edgecolor="none"
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    plt.title("Monte Carlo Simulation: Sharpe Ratio Comparison (Sorted by Score)")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(save_dir, "monte_carlo_boxplot.png"))
    plt.close()

if __name__ == "__main__":
    main()