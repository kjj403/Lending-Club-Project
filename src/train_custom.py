"""
부스팅 앙상블 파이프라인 전용 하이퍼파라미터 튜닝(Optuna) 및 피처 선택 모듈
"""

import config
import optuna
import numpy as np
import pandas as pd
import preprocess_pipeline
import data_loader
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# 변수 중요도(Feature Importance) 산출 및 시각화 리포트 저장
# -----------------------------------------------------------------------------
def save_feature_importance_artifacts(model, feature_names, model_name, save_dir, top_n=30):
    if model_name == "CatBoost":
        importances = model.get_feature_importance()
    else:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            print(f"⚠️ {model_name}: feature_importances_ 속성을 탐색할 수 없습니다.")
            return

    fi_df = pd.DataFrame(
        {"feature": list(feature_names), "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    full_csv = os.path.join(save_dir, f"{model_name}_feature_importance_full.csv")
    top_csv = os.path.join(save_dir, f"{model_name}_feature_importance_top{top_n}.csv")
    fi_df.to_csv(full_csv, index=False)
    fi_df.head(top_n).to_csv(top_csv, index=False)

    plot_df = fi_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, top_n * 0.28)))
    sns.barplot(data=plot_df, x="importance", y="feature", hue="feature", legend=False, palette="viridis")
    plt.title(f"{model_name} Feature Importance (Top {top_n})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, f"{model_name}_feature_importance_top{top_n}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"🧾 저장 완료: {full_csv}")
    print(f"📈 시각화 자료 저장 완료: {fig_path}")


# -----------------------------------------------------------------------------
# 목적함수 평가 지표 (Top-K 기준 Sharpe Ratio)
# -----------------------------------------------------------------------------
def get_top_k_sharpe(prob, y_true, meta_data, top_k_percent=0.15):
    """Optuna 목적함수: 승인 임계값(Top-K) 기준 샤프지수(Sharpe Ratio) 최적화"""
    eval_df = meta_data.iloc[:len(prob)].copy()
    eval_df["prob"] = np.asarray(prob)

    eval_df = eval_df.sort_values(by="prob", ascending=True)
    n_select = int(len(eval_df) * top_k_percent)

    if n_select < 10:
        return 0.0

    selection = eval_df.iloc[:n_select]
    excess = selection["actual_irr"].to_numpy(dtype=float) - selection["risk_free_rate"].to_numpy(dtype=float)
    excess = excess[np.isfinite(excess)]

    if len(excess) < 10:
        return 0.0

    std_ex = np.std(excess, ddof=1)
    if std_ex < 1e-9:
        return 0.0

    return float(np.mean(excess) / std_ex)


# -----------------------------------------------------------------------------
# 시계열 분할(OOT) 및 블록 부트스트랩 유틸
# -----------------------------------------------------------------------------
def _parse_issue_d(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(df["issue_d"], errors="coerce")
    return dt

def _time_split_indices(issue_dt: pd.Series, train_ratio=0.6, val_ratio=0.2):
    """테스트 데이터(최근 20%) 고정 및 훈련/검증 데이터 분할 튜닝"""
    sorted_idx = issue_dt.dropna().sort_values().index.to_numpy()
    n = len(sorted_idx)

    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio 합은 1.0보다 작아야 합니다.")

    n_test = max(1, int(n * test_ratio))
    if n_test >= n:
        n_test = max(1, n - 1)

    test_idx = sorted_idx[-n_test:]
    trainval_idx = sorted_idx[:-n_test]

    if len(trainval_idx) < 2:
        return pd.Index(trainval_idx), pd.Index([]), pd.Index(test_idx)

    rng = np.random.default_rng(int(getattr(config, "SEED", 42)))
    trainval_idx = rng.permutation(trainval_idx)

    train_share_within = train_ratio / (train_ratio + val_ratio)
    n_train = int(len(trainval_idx) * train_share_within)
    n_train = max(1, min(n_train, len(trainval_idx) - 1))

    train_idx = trainval_idx[:n_train]
    val_idx = trainval_idx[n_train:]

    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)

def _build_month_groups(issue_dt_train: pd.Series):
    m = issue_dt_train.dt.to_period("M")
    month_to_idx = {}
    for mm in sorted(m.dropna().unique()):
        idx = m.index[m == mm].to_numpy()
        if len(idx) > 0:
            month_to_idx[mm] = idx
    months = list(month_to_idx.keys())
    return months, month_to_idx

def _sample_block_bootstrap_indices(months, month_to_idx, target_size, block_months, rng):
    if len(months) == 0:
        return np.array([], dtype=int)

    if len(months) <= block_months:
        base = np.concatenate([month_to_idx[m] for m in months])
        return rng.choice(base, size=target_size, replace=True)

    out = []
    max_start = len(months) - block_months
    while len(out) < target_size:
        s = rng.integers(0, max_start + 1)
        block = months[s:s + block_months]
        idx_block = np.concatenate([month_to_idx[m] for m in block])
        if len(idx_block) == 0:
            continue
        out.extend(idx_block.tolist())

    return np.asarray(out[:target_size], dtype=int)

def _precompute_bootstrap_bank(issue_dt_train, n_trials, bootstrap_n, block_months, seed_base, target_size):
    """모델 간 비교 평가의 일관성을 위한 블록 부트스트랩 인덱스 뱅크 사전 생성"""
    months, month_to_idx = _build_month_groups(issue_dt_train)
    bank = {}
    for t in range(n_trials):
        for b in range(bootstrap_n):
            rng = np.random.default_rng(seed_base + (t * 1000) + b)
            bank[(t, b)] = _sample_block_bootstrap_indices(
                months=months,
                month_to_idx=month_to_idx,
                target_size=target_size,
                block_months=block_months,
                rng=rng,
            )
    return bank

# -----------------------------------------------------------------------------
# Optuna Objective
# -----------------------------------------------------------------------------
def optimize_model(
    X_train, y_train, w_train, X_val, y_val, meta_val, issue_dt_train,
    model_type="XGBoost", n_trials=10, bootstrap_bank=None
):
    print(f"\n🧠 [{model_type}] Optuna 기반 하이퍼파라미터 공간 탐색 중... (총 {n_trials}회)")

    bootstrap_n = int(getattr(config, "TRAIN_BOOTSTRAP_N", 1))
    block_months = int(getattr(config, "TRAIN_BOOT_BLOCK_MONTHS", 3))
    bootstrap_lambda = float(getattr(config, "TRAIN_BOOTSTRAP_LAMBDA", 0))
    seed_base = int(getattr(config, "SEED", 42))

    months, month_to_idx = _build_month_groups(issue_dt_train)

    def _build_model(param):
        if model_type == "XGBoost":
            return XGBClassifier(**param)
        if model_type == "LightGBM":
            return LGBMClassifier(**param)
        if model_type == "CatBoost":
            return CatBoostClassifier(**param)
        raise ValueError(f"지원하지 않는 모형 타입: {model_type}")

    def objective(trial):
        if model_type == "XGBoost":
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 7),
                "random_state": config.SEED,
                "n_jobs": -1,
                "tree_method": "hist",
                "enable_categorical": True,
            }
        elif model_type == "LightGBM":
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 800),
                "num_leaves": trial.suggest_int("num_leaves", 40, 100),
                "learning_rate": trial.suggest_float("learning_rate", 0.07, 0.15),
                "min_child_samples": trial.suggest_int("min_child_samples", 70, 120),
                "random_state": config.SEED,
                "n_jobs": -1,
                "verbose": -1,
            }
        elif model_type == "CatBoost":
            param = {
                "iterations": trial.suggest_int("iterations", 500, 1000),
                "depth": trial.suggest_int("depth", 6, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 4, 8),
                "random_seed": config.SEED,
                "verbose": 0,
                "allow_writing_files": False,
            }
        else:
            return -999.0

        # 임계값(Approval Rate) 기반 Sharpe Ratio 단일 목적함수 스코어 도출
        target_approval_rate = 0.15
        boot_scores = []

        for b in range(bootstrap_n):
            if bootstrap_bank is not None:
                bs_idx = bootstrap_bank.get((trial.number, b), np.array([], dtype=int))
            else:
                rng = np.random.default_rng(seed_base + (trial.number * 1000) + b)
                bs_idx = _sample_block_bootstrap_indices(
                    months=months,
                    month_to_idx=month_to_idx,
                    target_size=len(X_train),
                    block_months=block_months,
                    rng=rng,
                )

            if len(bs_idx) == 0:
                continue

            X_b = X_train.loc[bs_idx]
            y_b = y_train.loc[bs_idx]
            w_b = w_train.loc[bs_idx]

            model = _build_model(param)
            model.fit(X_b, y_b, sample_weight=w_b)

            probs = model.predict_proba(X_val)[:, 1]
            sr = get_top_k_sharpe(
                probs,
                y_val,
                meta_val,
                top_k_percent=target_approval_rate,
            )

            boot_scores.append(float(sr))

        if len(boot_scores) == 0:
            return -999.0

        boot_scores = np.asarray(boot_scores, dtype=float)
        mean_s = float(np.mean(boot_scores))
        std_s = float(np.std(boot_scores, ddof=1)) if len(boot_scores) > 1 else 0.0
        final_score = mean_s - bootstrap_lambda * std_s

        if not np.isfinite(final_score):
            return -999.0
        return final_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"🏆 최적 파라미터(Best Params): {study.best_params}")
    print(f"📈 예상 목적함수 스코어(Score): {study.best_value:.4f}")
    return study.best_params


# -----------------------------------------------------------------------------
# 메인 학습/분할 파이프라인
# -----------------------------------------------------------------------------
def train_and_split(df, model_type="XGBoost", feature_list=None, use_optuna=True, select_top_n=30):
    df = df.reset_index(drop=True).copy()

    issue_dt = _parse_issue_d(df)
    valid_time_mask = issue_dt.notna()
    df = df.loc[valid_time_mask].copy()
    issue_dt = issue_dt.loc[valid_time_mask]

    train_ratio = float(getattr(config, "TIME_TRAIN_RATIO", 0.6))
    val_ratio = float(getattr(config, "TIME_VAL_RATIO", 0.2))

    # 타겟 누수 방지를 위한 사후 지표 소거
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

    if feature_list is not None:
        features = [f for f in feature_list if f in df.columns and f not in exclude_cols]
    else:
        features = [c for c in df.columns if c not in exclude_cols]

    print(f"✅ 유효 변수군 필터링 완료: {len(df.columns)} -> {len(features)}개 투입")

    X = df[features].copy()
    cat_cols = X.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        X[col] = X[col].cat.codes
    X = X.select_dtypes(include=["number"])

    y = df["target"].copy()
    if "sample_weight" in df.columns:
        weights = df["sample_weight"].copy()
    else:
        weights = pd.Series(np.ones(len(df)), index=df.index)

    valid_meta_cols = [c for c in list(finance_cols) + leakage_cols if c in df.columns]
    meta = df[valid_meta_cols].copy()
    meta = meta.loc[:, ~meta.columns.duplicated()]

    train_idx, val_idx, test_idx = _time_split_indices(
        issue_dt=issue_dt,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
    w_train, w_val, w_test = weights.loc[train_idx], weights.loc[val_idx], weights.loc[test_idx]
    meta_train, meta_val, meta_test = meta.loc[train_idx], meta.loc[val_idx], meta.loc[test_idx]
    issue_dt_train = issue_dt.loc[train_idx]

    # Two-stage Feature Selection (상위 N개 변수 추출)
    if select_top_n is not None and X_train.shape[1] > select_top_n:
        print(f"✂️ [{model_type}] 트리 분기 기준 중요도 기반 Top {select_top_n} 변수 1차 선별 중...")

        if model_type == "XGBoost":
            sel_model = XGBClassifier(
                n_estimators=100, max_depth=4, n_jobs=-1, random_state=config.SEED,
                tree_method="hist", enable_categorical=True
            )
        elif model_type == "LightGBM":
            sel_model = LGBMClassifier(
                n_estimators=100, num_leaves=31, n_jobs=-1, random_state=config.SEED, verbose=-1
            )
        elif model_type == "CatBoost":
            sel_model = CatBoostClassifier(
                iterations=100, depth=6, random_seed=config.SEED, verbose=0, allow_writing_files=False
            )
        else:
            raise ValueError(f"지원하지 않는 모형 타입: {model_type}")

        sel_model.fit(X_train, y_train, sample_weight=w_train)

        if model_type == "CatBoost":
            imp = sel_model.get_feature_importance()
        else:
            imp = sel_model.feature_importances_

        fi_series = pd.Series(imp, index=X_train.columns)
        selected_features = fi_series.sort_values(ascending=False).head(select_top_n).index.tolist()

        print(f"✅ {model_type} 최종 투입 변수: {len(selected_features)}개 확정")
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

    # Optuna 탐색 또는 사전 정의된 하이퍼파라미터 적용
    if use_optuna:
        n_trials_opt = 10
        bootstrap_n = int(getattr(config, "TRAIN_BOOTSTRAP_N", 1))
        block_months = int(getattr(config, "TRAIN_BOOT_BLOCK_MONTHS", 3))
        seed_base = int(getattr(config, "SEED", 42))

        issue_dt_train_used = issue_dt_train.loc[X_train.index]
        bootstrap_bank = _precompute_bootstrap_bank(
            issue_dt_train=issue_dt_train_used,
            n_trials=n_trials_opt,
            bootstrap_n=bootstrap_n,
            block_months=block_months,
            seed_base=seed_base,
            target_size=len(X_train),
        )

        best_params = optimize_model(
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            X_val=X_val,
            y_val=y_val,
            meta_val=meta_val,
            issue_dt_train=issue_dt_train_used,
            model_type=model_type,
            n_trials=n_trials_opt,
            bootstrap_bank=bootstrap_bank,
        )

        if model_type == "XGBoost":
            model = XGBClassifier(
                **best_params,
                n_jobs=-1,
                tree_method="hist",
                enable_categorical=True,
                random_state=config.SEED,
            )
        elif model_type == "LightGBM":
            model = LGBMClassifier(**best_params, n_jobs=-1, verbose=-1, random_state=config.SEED)
        elif model_type == "CatBoost":
            model = CatBoostClassifier(**best_params, verbose=0, allow_writing_files=False, random_seed=config.SEED)

    else:
        if model_type == "XGBoost":
            model = XGBClassifier(
                n_estimators=792,
                max_depth=5,
                learning_rate=0.07431487008170813,
                subsample=0.8735099247695728,
                colsample_bytree=0.6173309618335238,
                min_child_weight=6,
                n_jobs=-1,
                random_state=config.SEED,
                tree_method="hist",
                enable_categorical=True,
            )
        elif model_type == "LightGBM":
            model = LGBMClassifier(
                n_estimators=604,
                num_leaves=66,
                learning_rate=0.08096281886948425,
                min_child_samples=109,
                n_jobs=-1,
                random_state=config.SEED,
                verbose=-1,
            )
        elif model_type == "CatBoost":
            model = CatBoostClassifier(
                iterations=861,
                depth=8,
                learning_rate=0.05519610315390476,
                l2_leaf_reg= 6.594404091581386,
                random_seed=config.SEED,
                verbose=0,
                allow_writing_files=False,
            )

    # 최종 모형 학습
    fit_params = {"sample_weight": w_train}
    if model_type == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, **fit_params)
    elif model_type == "LightGBM":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="logloss", **fit_params)
    elif model_type == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, **fit_params)

    return model, X_val, meta_val, X_test, meta_test


# -----------------------------------------------------------------------------
# 실행부 (Execution)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ [System] 모델별 최적 하이퍼파라미터 탐색 파이프라인 가동...")
    df = data_loader.prepare_data_with_weights()

    model_names = ["XGBoost", "LightGBM", "CatBoost"]

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fi_save_dir = os.path.join(config.BASE_DIR, "../reports/feature_importance", f"run_{now_str}")
    os.makedirs(fi_save_dir, exist_ok=True)

    print("\n🚀 [START] 모형별 학습 및 튜닝 프로세스 시작")

    for m_name in model_names:
        print("\n" + "=" * 60)
        print(f"🔍 {m_name} 모델 최적화 및 평가 진행 중...")
        print("=" * 60)

        model, X_val, meta_val, X_test, meta_test = train_and_split(
            df,
            model_type=m_name,
            use_optuna=True,
            select_top_n=30,
        )

        save_feature_importance_artifacts(
            model=model,
            feature_names=X_val.columns,
            model_name=m_name,
            save_dir=fi_save_dir,
            top_n=30,
        )

        print(f"✨ {m_name} 튜닝 및 학습 파이프라인 완료!")

    print("\n🎉 모든 모형에 대한 최적화 및 변수 중요도 추출이 성공적으로 완료되었습니다!")