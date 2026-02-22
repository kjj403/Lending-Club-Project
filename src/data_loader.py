import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import config

def _get_irr_cache_path() -> str:
    if hasattr(config, "IRR_CACHE_PATH"):
        return config.IRR_CACHE_PATH
    base_dir = getattr(config, "BASE_DIR", os.getcwd())
    return os.path.join(base_dir, "../data/cache/actual_irr_cache.parquet")

def load_processed_data():
    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(f"데이터 파일이 존재하지 않습니다: {config.DATA_PATH}")
        
    df = pd.read_parquet(config.DATA_PATH)
    
    # [수정됨] 데이터를 불러온 후 Pandas 환경에서 안전하게 Categorical 타입으로 변환합니다.
    cat_targets = ['home_ownership', 'purpose', 'initial_list_status', 'grade', 'sub_grade', 'verification_status']
    existing_cats = [c for c in cat_targets if c in df.columns]
    for c in existing_cats:
        df[c] = df[c].astype('category')
        
    return df

def load_treasury_rates():
    """
    무위험 이자율 산출을 위한 국채 금리 데이터(3년물, 5년물) 로드 및 보간
    """
    try:
        gs3 = pd.read_csv(config.GS3_PATH)
        gs5 = pd.read_csv(config.GS5_PATH)

        gs3.columns = [c.strip().lower() for c in gs3.columns]
        gs5.columns = [c.strip().lower() for c in gs5.columns]

        if "observation_date" in gs3.columns:
            gs3.rename(columns={"observation_date": "DATE"}, inplace=True)
        if "observation_date" in gs5.columns:
            gs5.rename(columns={"observation_date": "DATE"}, inplace=True)

        gs3["DATE"] = pd.to_datetime(gs3["DATE"], errors="coerce")
        gs5["DATE"] = pd.to_datetime(gs5["DATE"], errors="coerce")

        if "gs3" in gs3.columns:
            gs3.rename(columns={"gs3": "GS3"}, inplace=True)
        elif len(gs3.columns) > 1:
            col_name = [c for c in gs3.columns if c != "DATE"][0]
            gs3.rename(columns={col_name: "GS3"}, inplace=True)

        if "gs5" in gs5.columns:
            gs5.rename(columns={"gs5": "GS5"}, inplace=True)
        elif len(gs5.columns) > 1:
            col_name = [c for c in gs5.columns if c != "DATE"][0]
            gs5.rename(columns={col_name: "GS5"}, inplace=True)

        gs3["GS3"] = pd.to_numeric(gs3["GS3"], errors="coerce")
        gs5["GS5"] = pd.to_numeric(gs5["GS5"], errors="coerce")

        gs3 = gs3.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE").resample("D").interpolate().reset_index()
        gs5 = gs5.dropna(subset=["DATE"]).sort_values("DATE").set_index("DATE").resample("D").interpolate().reset_index()

        print(f"✅ 거시경제 지표(국채 금리) 로드 완료: GS3({len(gs3)}건), GS5({len(gs5)}건)")
        return gs3, gs5

    except Exception as e:
        print(f"⚠️ 국채 금리 로드 실패. 상세 내역: {e}")
        return None, None

def map_risk_free_rate(df):
    """
    대출 만기(Term) 및 발행일(Issue Date)에 맞춘 무위험 이자율(Risk-Free Rate) 동적 매핑
    """
    gs3, gs5 = load_treasury_rates()

    if gs3 is None or gs5 is None:
        print("⚠️ 국채 데이터 누락으로 인하여 기본 무위험 이자율(2%)을 일괄 적용합니다.")
        return df

    out = df.copy()
    if "risk_free_rate" not in out.columns:
        out["risk_free_rate"] = np.nan

    out["issue_d_parsed"] = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    out["term_str"] = out["term"].astype(str).str.strip()

    mask_36 = out["term_str"].str.contains("36", na=False) & out["issue_d_parsed"].notna()
    mask_60 = out["term_str"].str.contains("60", na=False) & out["issue_d_parsed"].notna()

    if mask_36.any():
        base_36 = out.loc[mask_36, ["issue_d_parsed"]].copy()
        base_36["__idx__"] = base_36.index
        base_36 = base_36.sort_values("issue_d_parsed")
        merged_36 = pd.merge_asof(
            base_36,
            gs3.sort_values("DATE"),
            left_on="issue_d_parsed",
            right_on="DATE",
            direction="backward",
        )
        out.loc[merged_36["__idx__"].to_numpy(), "risk_free_rate"] = merged_36["GS3"].to_numpy() / 100.0

    if mask_60.any():
        base_60 = out.loc[mask_60, ["issue_d_parsed"]].copy()
        base_60["__idx__"] = base_60.index
        base_60 = base_60.sort_values("issue_d_parsed")
        merged_60 = pd.merge_asof(
            base_60,
            gs5.sort_values("DATE"),
            left_on="issue_d_parsed",
            right_on="DATE",
            direction="backward",
        )
        out.loc[merged_60["__idx__"].to_numpy(), "risk_free_rate"] = merged_60["GS5"].to_numpy() / 100.0

    out.drop(columns=["issue_d_parsed", "term_str"], inplace=True, errors="ignore")
    return out

def _solve_monthly_rate_annuity(principal: float, pmt: float, n_months: int) -> float:
    """등가연금(Annuity) 방식의 월별 내부수익률(IRR) 산출을 위한 이분 탐색 수치해석"""
    if principal <= 0 or pmt <= 0 or n_months <= 0:
        return np.nan

    n = int(n_months)

    def npv(r):
        if r <= -0.9999:
            return np.nan
        d = 1.0 + r
        if abs(r) < 1e-12:
            pv = pmt * n
        else:
            pv = pmt * (1.0 - d ** (-n)) / r
        return -principal + pv

    lo, hi = -0.999, 5.0
    f_lo, f_hi = npv(lo), npv(hi)

    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan

    k = 0
    while f_lo * f_hi > 0 and k < 20:
        hi *= 1.5
        f_hi = npv(hi)
        if np.isnan(f_hi):
            return np.nan
        k += 1

    if f_lo * f_hi > 0:
        return np.nan

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)

def _solve_monthly_irr_optimized(row_tuple):
    """
    단일 현금흐름(Balloon Payment) 기반의 연환산 내부수익률(IRR) 최적화 연산
    """
    principal, installment, n_months, total_inflow = row_tuple

    if principal <= 0 or total_inflow <= 0 or n_months <= 0:
        return np.nan

    n = int(n_months)
    reg_n = max(0, n - 1)

    # 마지막 기일에 잔여 현금흐름이 집중된다는 가정(Balloon payment)
    balloon = total_inflow - (installment * reg_n)

    # 현금흐름 구조상 Balloon이 음수인 경우 등가연금(Annuity) 방식으로 대체 연산
    if balloon <= 0:
        pmt_new = total_inflow / n
        r_m = _solve_monthly_rate_annuity(principal, pmt_new, n)
        if np.isnan(r_m):
            return np.nan
        return (1.0 + r_m) ** 12 - 1.0

    def npv(r):
        if r <= -0.9999:
            return np.nan
        d = 1.0 + r

        if reg_n == 0:
            pv_reg = 0.0
        elif abs(r) < 1e-12:
            pv_reg = installment * reg_n
        else:
            pv_reg = installment * (1.0 - d ** (-reg_n)) / r

        pv_balloon = balloon / (d ** n)
        return -principal + pv_reg + pv_balloon

    lo, hi = -0.999, 5.0
    f_lo, f_hi = npv(lo), npv(hi)

    if np.isnan(f_lo) or np.isnan(f_hi):
        return np.nan

    k = 0
    while f_lo * f_hi > 0 and k < 20:
        hi *= 1.5
        f_hi = npv(hi)
        if np.isnan(f_hi):
            return np.nan
        k += 1

    if f_lo * f_hi > 0:
        return np.nan

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)

        if np.isnan(f_mid):
            return np.nan
        if abs(f_mid) < 1e-8:
            return (1.0 + mid) ** 12 - 1.0

        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return (1.0 + 0.5 * (lo + hi)) ** 12 - 1.0

def calculate_actual_irr(df):
    """데이터프레임 내 개별 대출 건에 대한 내부수익률(IRR) 일괄 계산"""
    print("⏳ 대출 상환 이력 기반 내부수익률(IRR) 연산 중...")

    out = df.copy()
    if "actual_irr" in out.columns:
        out.drop(columns=["actual_irr"], inplace=True)

    issue_dt = pd.to_datetime(out["issue_d"], format="%b-%Y", errors="coerce")
    last_dt = pd.to_datetime(out["last_pymnt_d"], format="%b-%Y", errors="coerce")
    out["n_months"] = ((last_dt - issue_dt).dt.days / 30.4375).round().clip(lower=1).fillna(0).astype(int)

    if "total_pymnt" not in out.columns:
        out["total_pymnt"] = (
            pd.to_numeric(out["total_rec_prncp"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["total_rec_int"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["recoveries"], errors="coerce").fillna(0.0)
        )

    out["funded_amnt"] = pd.to_numeric(out["funded_amnt"], errors="coerce")
    out["installment"] = pd.to_numeric(out["installment"], errors="coerce")
    out["total_pymnt"] = pd.to_numeric(out["total_pymnt"], errors="coerce")

    terminal = out["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])

    out["_k_principal"] = out["funded_amnt"].round(2)
    out["_k_install"] = out["installment"].round(2)
    out["_k_total"] = out["total_pymnt"].round(2)

    key_cols = ["_k_principal", "_k_install", "n_months", "_k_total"]

    valid = (
        terminal
        & out["_k_principal"].notna() & (out["_k_principal"] > 0)
        & out["_k_install"].notna() & (out["_k_install"] >= 0)
        & out["_k_total"].notna() & (out["_k_total"] > 0)
        & (out["n_months"] > 0)
    )

    target_df = out.loc[valid, key_cols].copy()
    unique_cases = target_df.drop_duplicates()

    print(f"   - 전체 연산 대상: {len(target_df):,}건 / 유니크(병합) 연산: {len(unique_cases):,}건")

    records = unique_cases.to_records(index=False)
    results = []
    for row in tqdm(records, desc="IRR Solving"):
        results.append(_solve_monthly_irr_optimized(row))

    unique_cases["actual_irr"] = np.array(results, dtype=float)

    out = out.join(unique_cases.set_index(key_cols)["actual_irr"], on=key_cols)
    out["actual_irr"] = out["actual_irr"].clip(lower=-1.0, upper=3.0)

    out.drop(columns=["_k_principal", "_k_install", "_k_total", "n_months"], inplace=True, errors="ignore")
    return out

def attach_cached_actual_irr(df, force_recompute=False):
    """연산 시간 단축을 위한 캐싱(Caching) 기법 기반 IRR 결합"""
    out = df.copy()
    cache_path = _get_irr_cache_path()

    if "id" in out.columns:
        key_col = "id"
    else:
        hash_cols = [c for c in ["issue_d", "last_pymnt_d", "funded_amnt", "installment", "loan_status"] if c in out.columns]
        if len(hash_cols) == 0:
            raise ValueError("고유 식별(Hash) 키 생성을 위한 필수 컬럼이 존재하지 않습니다.")
        out["_irr_key"] = pd.util.hash_pandas_object(out[hash_cols].astype(str), index=False).astype("int64")
        key_col = "_irr_key"

    if "actual_irr" not in out.columns:
        out["actual_irr"] = np.nan

    if (not force_recompute) and os.path.exists(cache_path):
        try:
            cache = pd.read_parquet(cache_path)
            if key_col in cache.columns and "actual_irr" in cache.columns:
                cache = cache[[key_col, "actual_irr"]].drop_duplicates(subset=[key_col], keep="last")
                out = out.merge(cache, on=key_col, how="left", suffixes=("", "_cache"))
                out["actual_irr"] = out["actual_irr"].fillna(out["actual_irr_cache"])
                out.drop(columns=["actual_irr_cache"], inplace=True, errors="ignore")
                print(f"✅ 기산출된 IRR 캐시(Cache) 메모리 로드 성공: {cache_path}")
        except Exception as e:
            print(f"⚠️ IRR 캐시 로드 실패 (재계산을 수행합니다): {e}")

    need = out["actual_irr"].isna() | ~np.isfinite(out["actual_irr"])
    if need.any():
        print(f"🔄 미산출된 데이터에 대한 신규 IRR 연산 시작: {need.sum():,}건")
        calc = calculate_actual_irr(out.loc[need].copy())
        out.loc[need, "actual_irr"] = calc["actual_irr"].values

    cache_out = out[[key_col, "actual_irr"]].dropna().drop_duplicates(subset=[key_col], keep="last")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_out.to_parquet(cache_path, index=False)

    if key_col == "_irr_key":
        out.drop(columns=["_irr_key"], inplace=True, errors="ignore")

    return out

def compute_sample_weights(df, default_penalty=1.5):
    """
    비용 민감 학습(Cost-sensitive Learning)을 위한 샘플별 가중치 산출 알고리즘
    - 고액 대출 부도 건에 대한 모델의 과적합(가중치 폭주)을 방지하기 위해 로그 스케일(Log-scale) 반영
    """
    weights = np.ones(len(df))
    mask_default = (df["target"] == 1)

    if "loan_amnt" in df.columns:
        amt = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0).clip(lower=0.0)

        median_amount = float(np.nanmedian(amt.values))
        if median_amount <= 0:
            median_amount = 1.0

        denom = np.log1p(median_amount)
        if denom <= 0:
            denom = 1.0
        amount_factor = np.log1p(amt) / denom

        # 모델 안정성을 위한 가중치 상/하한선(Clipping) 설정
        amount_factor = amount_factor.clip(lower=0.5, upper=2.0)

        weights[mask_default] = default_penalty * amount_factor[mask_default]
    else:
        weights[mask_default] = default_penalty

    return weights

def prepare_data_with_weights():
    """모델 학습 전처리 파이프라인(가중치 산출 및 거시지표 매핑)"""
    print("🔄 [Step 1] 파생 데이터 로드 및 훈련 가중치 파이프라인 초기화...")
    df = load_processed_data()

    print("💵 무위험 이자율(Risk-Free Rate) 매핑 프로세스 진행 중...")
    try:
        df = map_risk_free_rate(df)
    except Exception as e:
        print(f"⚠️ 매핑 중 예외 발생({e}). 일괄 기본값 처리합니다.")

    if "risk_free_rate" not in df.columns:
        df["risk_free_rate"] = 0.02
    else:
        df["risk_free_rate"] = df["risk_free_rate"].fillna(0.02)

    try:
        df = attach_cached_actual_irr(df, force_recompute=False)
    except Exception as e:
        print(f"❌ IRR 연산 파이프라인 오류: {e}")
        return None

    if "int_rate" in df.columns:
        if df["int_rate"].mean() > 1:
            df["int_rate_spread"] = (df["int_rate"] / 100.0) - df["risk_free_rate"]
        else:
            df["int_rate_spread"] = df["int_rate"] - df["risk_free_rate"]

    print("⚖️ 금융 특화 훈련 가중치(Cost-sensitive Weights) 부여 중...")
    df["sample_weight"] = compute_sample_weights(df, default_penalty=1.5)
    # 정규화를 통한 학습 스케일 안정화
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()

    print("✅ 데이터 준비 파이프라인 완료 (부도 패널티 및 금액 가중치 적용 완수)")
    return df