"""
승인 임계값(Approval Rate Threshold) 최적화 및 스윕(Sweep) 분석 모듈
- 수익률(ROI)과 안정성(Sharpe Ratio)을 동시에 고려한 복합 점수(Composite Score) 기반 최적 임계값 도출
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import data_loader
import train_custom

# -----------------------------------------------------------
# 전역 설정 (Global Configurations)
# -----------------------------------------------------------
MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "Ensemble(Cat+LGBM)"]
BASE_MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost"]
SELECT_TOP_N = 30

# 평가를 위한 승인률(Top-K) 탐색 구간 (1% ~ 80%, 1% 단위)
K_VALUES = np.arange(0.01, 0.81, 0.01)

AMOUNT_COL_CANDIDATES = ["loan_amnt", "funded_amnt"]
REPORT_K_SNAPSHOTS = [0.05, 0.10, 0.15, 0.20, 0.25]

# 포트폴리오 구성을 위한 현실적 제약 조건 (Feasible Constraints)
K_MIN = 0.30
K_MAX = 0.80
SHARPE_MIN = 0.01
N_SELECTED_MIN = 10000

# 최적 임계값 탐색을 위한 복합 점수(Composite Score) 가중치 (Sharpe Ratio 및 ROI 균형)
W_SHARPE = 0.5
W_ROI = 0.5


# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def _pick_amount_col(df: pd.DataFrame) -> str:
    """데이터프레임 내 대출 금액(Amount) 관련 컬럼 탐색 및 반환"""
    for c in AMOUNT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"금액 컬럼 부재. 후보군: {AMOUNT_COL_CANDIDATES}")


def _normalize(s: pd.Series) -> pd.Series:
    """복합 점수(Composite Score) 산출을 위한 Min-Max 정규화"""
    x = pd.to_numeric(s, errors="coerce")
    arr = x.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return pd.Series(np.zeros(len(x)), index=s.index)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - mn) / (mx - mn)


def _calc_topk_metrics(prob_s, meta_df, amount_s, k):
    """
    특정 승인률(Top-K) 기준의 포트폴리오 성과 지표(Sharpe Ratio, ROI 등) 산출
    """
    common_idx = prob_s.index.intersection(meta_df.index).intersection(amount_s.index)
    if len(common_idx) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    eval_df = meta_df.loc[common_idx, ["actual_irr", "risk_free_rate"]].copy()
    eval_df["prob"] = pd.to_numeric(prob_s.loc[common_idx], errors="coerce")
    eval_df["amount"] = pd.to_numeric(amount_s.loc[common_idx], errors="coerce")
    eval_df = eval_df.dropna(subset=["prob", "actual_irr", "risk_free_rate", "amount"])

    if len(eval_df) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, np.nan

    # 부도 확률이 낮은 순서대로 정렬하여 상위 K% 선별
    eval_df = eval_df.sort_values("prob", ascending=True)
    n_select = int(len(eval_df) * k)
    approval_rate = float(n_select / len(eval_df))

    if n_select < 10:
        return np.nan, np.nan, np.nan, np.nan, n_select, approval_rate

    sel = eval_df.iloc[:n_select]
    actual = sel["actual_irr"].to_numpy(dtype=float)
    rf = sel["risk_free_rate"].to_numpy(dtype=float)
    amount = sel["amount"].to_numpy(dtype=float)

    mask = np.isfinite(actual) & np.isfinite(rf) & np.isfinite(amount)
    actual = actual[mask]
    rf = rf[mask]
    amount = amount[mask]

    if len(actual) < 10:
        return np.nan, np.nan, np.nan, np.nan, n_select, approval_rate

    # 초과 수익률(Excess Return) 기반 Sharpe Ratio 산출
    excess = actual - rf
    std_ex = np.std(excess, ddof=1)
    sharpe = np.nan if std_ex < 1e-12 else float(np.mean(excess) / std_ex)

    # 누적 수익(Total Profit) 및 투자수익률(ROI) 산출
    total_profit = float(np.sum(actual * amount))
    invested_amount = float(np.sum(amount))
    roi = np.nan if invested_amount <= 1e-12 else float(total_profit / invested_amount)

    return sharpe, total_profit, roi, invested_amount, n_select, approval_rate


# -----------------------------------------------------------
# Model Evaluation (Validation Set)
# -----------------------------------------------------------
def _build_models_once(df: pd.DataFrame):
    """모델별 1회 학습 및 검증 세트(Validation Set)에 대한 예측 확률 산출 캐싱"""
    amount_col = _pick_amount_col(df)

    raw_probs, raw_meta, raw_amount = {}, {}, {}
    probs_map, meta_map, amount_map = {}, {}, {}

    for m in BASE_MODEL_NAMES:
        print(f"\n🔧 모델 검증 캐시 생성 중: {m}")
        model, X_val, meta_val, _, _ = train_custom.train_and_split(
            df=df,
            model_type=m,
            feature_list=None,
            use_optuna=False,
            select_top_n=SELECT_TOP_N,
        )
        raw_probs[m] = pd.Series(model.predict_proba(X_val)[:, 1], index=X_val.index, name="prob")
        raw_meta[m] = meta_val
        raw_amount[m] = df.loc[X_val.index, amount_col]

    # 공통 검증 인덱스(Validation Index) 추출을 통한 표본 동일성 확보
    common_idx = None
    for m in BASE_MODEL_NAMES:
        idx_m = raw_probs[m].index.intersection(raw_meta[m].index).intersection(raw_amount[m].index)
        common_idx = idx_m if common_idx is None else common_idx.intersection(idx_m)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("공통 검증 샘플이 존재하지 않습니다. 데이터 분할 로직을 확인하십시오.")

    for m in BASE_MODEL_NAMES:
        probs_map[m] = raw_probs[m].loc[common_idx]
        meta_map[m] = raw_meta[m].loc[common_idx]
        amount_map[m] = raw_amount[m].loc[common_idx]

    # 앙상블 모형(CatBoost + LightGBM) 확률 산출 (Soft Voting)
    idx_ens = probs_map["CatBoost"].index.intersection(probs_map["LightGBM"].index)
    idx_ens = idx_ens.intersection(meta_map["CatBoost"].index).intersection(meta_map["LightGBM"].index)

    p_ens = (
        probs_map["CatBoost"].loc[idx_ens].to_numpy(dtype=float)
        + probs_map["LightGBM"].loc[idx_ens].to_numpy(dtype=float)
    ) / 2.0

    probs_map["Ensemble(Cat+LGBM)"] = pd.Series(p_ens, index=idx_ens, name="prob")
    meta_map["Ensemble(Cat+LGBM)"] = meta_map["CatBoost"].loc[idx_ens]
    amount_map["Ensemble(Cat+LGBM)"] = df.loc[idx_ens, amount_col]

    print(f"✅ 유효 검증 샘플 수: {len(common_idx):,}")

    return probs_map, meta_map, amount_map


# -----------------------------------------------------------
# Threshold Sweep Logic
# -----------------------------------------------------------
def _run_approval_sweep(probs_map, meta_map, amount_map):
    """사전 정의된 승인률(K) 구간 전체에 대한 포트폴리오 성과 지표 스윕(Sweep) 탐색"""
    rows = []
    for model_name in MODEL_NAMES:
        prob_s = probs_map[model_name]
        meta_df = meta_map[model_name]
        amt_s = amount_map[model_name]

        for k in K_VALUES:
            sharpe, total_profit, roi, invested, n_sel, approval_rate = _calc_topk_metrics(
                prob_s=prob_s,
                meta_df=meta_df,
                amount_s=amt_s,
                k=k,
            )
            rows.append(
                {
                    "model": model_name,
                    "k_input": float(k),
                    "approval_rate": approval_rate,
                    "n_selected": int(n_sel),
                    "invested_amount": invested,
                    "sharpe": sharpe,
                    "total_profit": total_profit,
                    "roi": roi,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------
# Optimal Strategy Recommendation
# -----------------------------------------------------------
def _make_recommendation(curve_df: pd.DataFrame):
    """제약 조건(Feasible Constraints) 하에서 복합 점수가 최대화되는 개별 모델 최적 승인률 추천"""
    rec_rows = []
    score_parts = []

    for model_name, g in curve_df.groupby("model"):
        band = g[
            (g["approval_rate"] >= K_MIN) & (g["approval_rate"] <= K_MAX)
        ].copy().sort_values("approval_rate")

        if len(band) == 0:
            rec_rows.append(
                {
                    "model": model_name,
                    "recommended_k": np.nan,
                    "recommended_sharpe": np.nan,
                    "recommended_total_profit": np.nan,
                    "recommended_roi": np.nan,
                    "recommended_n_selected": np.nan,
                    "recommended_score": np.nan,
                    "method": "no_band_data",
                }
            )
            continue

        band["sharpe_norm"] = _normalize(band["sharpe"])
        band["roi_norm"] = _normalize(band["roi"])
        band["score"] = W_SHARPE * band["sharpe_norm"] + W_ROI * band["roi_norm"]

        band["is_feasible"] = (
            (band["sharpe"] >= SHARPE_MIN)
            & (band["roi"] > 0)
            & (band["n_selected"] >= N_SELECTED_MIN)
        )
        score_parts.append(band)

        feasible = band[band["is_feasible"] & band["score"].notna()].copy()
        if len(feasible) > 0:
            best = feasible.loc[feasible["score"].idxmax()]
            method = "feasible_score_max"
        else:
            fallback = band[(band["sharpe"] > 0) & (band["roi"] > 0) & band["score"].notna()].copy()
            if len(fallback) > 0:
                best = fallback.loc[fallback["score"].idxmax()]
                method = "fallback_positive_score_max"
            else:
                best = band.loc[band["score"].idxmax()]
                method = "fallback_band_score_max"

        rec_rows.append(
            {
                "model": model_name,
                "recommended_k": float(best["approval_rate"]),
                "recommended_sharpe": float(best["sharpe"]),
                "recommended_total_profit": float(best["total_profit"]),
                "recommended_roi": float(best["roi"]),
                "recommended_n_selected": int(best["n_selected"]),
                "recommended_score": float(best["score"]),
                "method": method,
            }
        )

    rec_df = pd.DataFrame(rec_rows).sort_values("model")
    score_curve_df = pd.concat(score_parts, axis=0).sort_values(["model", "approval_rate"]) if len(score_parts) > 0 else pd.DataFrame()
    return rec_df, score_curve_df


def _make_global_k_recommendation(score_curve_df: pd.DataFrame):
    """모델 간 성과 편차를 최소화하는 전역(Global) 최적 승인 임계값 추천"""
    if score_curve_df is None or len(score_curve_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    n_models = score_curve_df["model"].nunique()

    agg = (
        score_curve_df.groupby("k_input", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            mean_sharpe=("sharpe", "mean"),
            mean_total_profit=("total_profit", "mean"),
            mean_roi=("roi", "mean"),
            mean_approval_rate=("approval_rate", "mean"),
            feasible_models=("is_feasible", "sum"),
            model_count=("model", "count"),
        )
        .sort_values("k_input")
    )

    agg["all_models_covered"] = agg["model_count"] == n_models
    agg["all_models_feasible"] = agg["feasible_models"] == n_models

    c1 = agg[agg["all_models_covered"] & agg["all_models_feasible"] & agg["mean_score"].notna()]
    if len(c1) > 0:
        best = c1.loc[c1["mean_score"].idxmax()]
        method = "all_models_feasible_mean_score_max"
    else:
        c2 = agg[agg["all_models_covered"] & agg["mean_score"].notna()]
        if len(c2) > 0:
            best = c2.loc[c2["mean_score"].idxmax()]
            method = "all_models_covered_mean_score_max"
        else:
            c3 = agg[agg["mean_score"].notna()]
            if len(c3) > 0:
                best = c3.loc[c3["mean_score"].idxmax()]
                method = "fallback_mean_score_max"
            else:
                return pd.DataFrame(), agg

    global_rec = pd.DataFrame(
        [
            {
                "global_k_input": float(best["k_input"]),
                "global_approval_rate_mean": float(best["mean_approval_rate"]),
                "global_mean_score": float(best["mean_score"]),
                "global_min_score": float(best["min_score"]),
                "global_mean_sharpe": float(best["mean_sharpe"]),
                "global_mean_total_profit": float(best["mean_total_profit"]),
                "global_mean_roi": float(best["mean_roi"]),
                "feasible_models": int(best["feasible_models"]),
                "model_count": int(best["model_count"]),
                "method": method,
            }
        ]
    )
    return global_rec, agg


# -----------------------------------------------------------
# Report & Visualization
# -----------------------------------------------------------
def _build_report_tables(curve_df, rec_df, score_curve_df):
    """성과 분석 요약 및 특정 임계값(Snapshot) 기준 지표 테이블 생성"""
    detail_df = curve_df.copy()
    summary_rows = []
    snapshot_rows = []

    for model_name, g in detail_df.groupby("model"):
        gg = g.sort_values("approval_rate").copy()
        rec_row = rec_df[rec_df["model"] == model_name].iloc[0]

        g_sh = gg.loc[gg["sharpe"].idxmax()] if gg["sharpe"].notna().any() else None
        g_pf = gg.loc[gg["total_profit"].idxmax()] if gg["total_profit"].notna().any() else None
        g_roi = gg.loc[gg["roi"].idxmax()] if gg["roi"].notna().any() else None

        summary_rows.append(
            {
                "model": model_name,
                "recommended_k": rec_row["recommended_k"],
                "recommended_sharpe": rec_row["recommended_sharpe"],
                "recommended_total_profit": rec_row["recommended_total_profit"],
                "recommended_roi": rec_row["recommended_roi"],
                "recommended_n_selected": rec_row["recommended_n_selected"],
                "recommended_score": rec_row["recommended_score"],
                "recommend_method": rec_row["method"],
                "max_sharpe": g_sh["sharpe"] if g_sh is not None else np.nan,
                "k_at_max_sharpe": g_sh["approval_rate"] if g_sh is not None else np.nan,
                "max_total_profit": g_pf["total_profit"] if g_pf is not None else np.nan,
                "k_at_max_profit": g_pf["approval_rate"] if g_pf is not None else np.nan,
                "max_roi": g_roi["roi"] if g_roi is not None else np.nan,
                "k_at_max_roi": g_roi["approval_rate"] if g_roi is not None else np.nan,
            }
        )

        g_score = score_curve_df[score_curve_df["model"] == model_name].sort_values("approval_rate")
        for kk in REPORT_K_SNAPSHOTS:
            nearest = gg.iloc[(gg["approval_rate"] - kk).abs().argmin()]
            score_near = np.nan
            feasible_near = np.nan
            if len(g_score) > 0:
                near_s = g_score.iloc[(g_score["approval_rate"] - kk).abs().argmin()]
                score_near = near_s["score"]
                feasible_near = bool(near_s["is_feasible"])

            snapshot_rows.append(
                {
                    "model": model_name,
                    "k_target": kk,
                    "k_used": nearest["approval_rate"],
                    "n_selected": nearest["n_selected"],
                    "invested_amount": nearest["invested_amount"],
                    "sharpe": nearest["sharpe"],
                    "total_profit": nearest["total_profit"],
                    "roi": nearest["roi"],
                    "score": score_near,
                    "is_feasible": feasible_near,
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    snapshot_df = pd.DataFrame(snapshot_rows).sort_values(["model", "k_target"])
    return detail_df, summary_df, snapshot_df


def _save_text_summary(summary_df: pd.DataFrame, path: str):
    lines = []
    lines.append("Approval Sweep Report Summary (Validation, Sharpe+ROI Composite)")
    lines.append("=" * 90)
    lines.append(f"Score = {W_SHARPE:.2f} * Sharpe_norm + {W_ROI:.2f} * ROI_norm")
    lines.append("-" * 90)

    for _, r in summary_df.iterrows():
        lines.append(f"[{r['model']}]")
        lines.append(
            f"  recommended_k={r['recommended_k']:.4f}, "
            f"sharpe={r['recommended_sharpe']:.4f}, "
            f"total_profit={r['recommended_total_profit']:.2f}, "
            f"roi={r['recommended_roi']:.6f}, "
            f"n_selected={int(r['recommended_n_selected']) if pd.notna(r['recommended_n_selected']) else 'nan'}, "
            f"score={r['recommended_score']:.4f}"
        )
        lines.append(
            f"  max_sharpe={r['max_sharpe']:.4f} @ k={r['k_at_max_sharpe']:.2f}, "
            f"max_profit={r['max_total_profit']:.2f} @ k={r['k_at_max_profit']:.2f}, "
            f"max_roi={r['max_roi']:.6f} @ k={r['k_at_max_roi']:.2f}"
        )
        lines.append("-" * 90)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_dual_axis(curve_df: pd.DataFrame, rec_df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    rec_map = rec_df.set_index("model")["recommended_k"].to_dict()

    for i, model_name in enumerate(MODEL_NAMES):
        ax1 = axes[i]
        g = curve_df[curve_df["model"] == model_name].sort_values("approval_rate")

        ax2 = ax1.twinx()
        ax1.plot(g["approval_rate"] * 100, g["sharpe"], color="tab:blue", linewidth=2)
        ax2.plot(g["approval_rate"] * 100, g["roi"], color="tab:orange", linewidth=2)

        rk = rec_map.get(model_name, np.nan)
        if np.isfinite(rk):
            ax1.axvline(rk * 100, linestyle="--", color="gray", alpha=0.8)

        ax1.set_title(model_name)
        ax1.set_xlabel("Approval Rate (%)")
        ax1.set_ylabel("Sharpe", color="tab:blue")
        ax2.set_ylabel("ROI", color="tab:orange")
        ax1.grid(alpha=0.3)

    plt.suptitle("Approval Rate Sweep (Validation): Sharpe vs ROI", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_score_curve(score_curve_df: pd.DataFrame, rec_df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    rec_map = rec_df.set_index("model")["recommended_k"].to_dict()

    for i, model_name in enumerate(MODEL_NAMES):
        ax = axes[i]
        g = score_curve_df[score_curve_df["model"] == model_name].sort_values("approval_rate")
        if len(g) == 0:
            continue

        ax.plot(g["approval_rate"] * 100, g["score"], color="tab:green", linewidth=2, label="Composite Score")
        feas = g[g["is_feasible"]]
        if len(feas) > 0:
            ax.scatter(
                feas["approval_rate"] * 100,
                feas["score"],
                s=12,
                alpha=0.6,
                color="tab:purple",
                label="Feasible",
            )

        rk = rec_map.get(model_name, np.nan)
        if np.isfinite(rk):
            ax.axvline(rk * 100, linestyle="--", color="gray", alpha=0.8)

        ax.set_title(model_name)
        ax.set_xlabel("Approval Rate (%)")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    plt.suptitle("Normalized Composite Score by Approval Rate", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    print("📥 데이터 로드 및 가중치 매핑 시작")
    df = data_loader.prepare_data_with_weights().reset_index(drop=True)

    print("🧠 검증 세트(Validation Set) 기반 모델별 확률 캐시 구축")
    probs_map, meta_map, amount_map = _build_models_once(df)

    print("📈 승인 임계값(Approval Rate Threshold) 탐색 시뮬레이션 가동")
    curve_df = _run_approval_sweep(probs_map, meta_map, amount_map)
    rec_df, score_curve_df = _make_recommendation(curve_df)
    global_rec_df, global_curve_df = _make_global_k_recommendation(score_curve_df)
    detail_df, summary_df, snapshot_df = _build_report_tables(curve_df, rec_df, score_curve_df)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.BASE_DIR, "../reports/approval_sweep", f"run_{now}")
    os.makedirs(save_dir, exist_ok=True)

    curve_csv = os.path.join(save_dir, "approval_curve_metrics_validation.csv")
    score_csv = os.path.join(save_dir, "approval_curve_score_validation.csv")
    rec_csv = os.path.join(save_dir, "approval_recommendation_composite_score_validation.csv")
    global_rec_csv = os.path.join(save_dir, "approval_global_k_recommendation_validation.csv")
    global_curve_csv = os.path.join(save_dir, "approval_global_k_curve_validation.csv")
    detail_csv = os.path.join(save_dir, "approval_curve_detail_validation.csv")
    summary_csv = os.path.join(save_dir, "approval_report_summary_validation.csv")
    snapshot_csv = os.path.join(save_dir, "approval_report_snapshots_validation.csv")
    summary_txt = os.path.join(save_dir, "approval_report_summary_validation.txt")

    fig_dual = os.path.join(save_dir, "approval_curve_dual_axis_validation.png")
    fig_score = os.path.join(save_dir, "approval_composite_score_curve_validation.png")

    curve_df.to_csv(curve_csv, index=False)
    score_curve_df.to_csv(score_csv, index=False)
    rec_df.to_csv(rec_csv, index=False)
    global_rec_df.to_csv(global_rec_csv, index=False)
    global_curve_df.to_csv(global_curve_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    snapshot_df.to_csv(snapshot_csv, index=False)
    _save_text_summary(summary_df, summary_txt)

    _plot_dual_axis(curve_df, rec_df, fig_dual)
    _plot_score_curve(score_curve_df, rec_df, fig_score)

    print("\n✅ 임계값 최적화 분석 완료")
    print(f"🧾 분석 결과 리포트 저장 디렉토리: {save_dir}")

    print("\n[모델별 최적 승인률 추천 결과]")
    print(rec_df.to_string(index=False))

    print("\n[공통 권장 승인률 (Global Recommendation)]")
    if len(global_rec_df) > 0:
        print(global_rec_df.to_string(index=False))
    else:
        print("조건을 만족하는 공통 추천 임계값이 탐색되지 않았습니다.")

    print("\n[최종 요약 표]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()