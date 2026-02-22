"""
Lending Club 데이터 전처리 파이프라인 (Polars Lazy Execution 기반)
- 결측치 제어, 파생 변수 생성, 타겟 누수(Data Leakage) 차단 로직 포함
"""

import polars as pl
import os
import config

# =============================================================================
# 1. 상수 및 변수 설정 (Constants & Columns definition)
# =============================================================================

# 성과 검증(수익률 계산 등)을 위해 파이프라인 통과 후에도 반드시 보존되어야 하는 변수군
FINANCE_COLS = [
    'id', 'member_id',
    'loan_status', 'issue_d', 'term', 
    'int_rate', 'installment', 'grade', 'sub_grade',
    'funded_amnt', 'funded_amnt_inv',
    'total_pymnt', 'total_rec_prncp', 'total_rec_int', 
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d'
]

def get_cols_to_drop():
    """학습에 불필요하거나 Data Leakage를 유발할 수 있는 변수 목록 반환"""
    drop_cols = [
        # --- 1. 기본 식별자 및 텍스트 ---
        'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code', 'addr_state',
        'policy_code', 'pymnt_plan', 'issue_d_parsed', 'earliest_cr_line_parsed',
        
        # --- 2. 미래 정보 (Data Leakage 차단을 위한 선제적 제거) ---
        'roi_pct', 'last_fico_range_high', 'last_fico_range_low', 
        'total_pymnt_inv', 'total_pymnt', 'total_rec_prncp', 'total_rec_int', 
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 
        'last_pymnt_amnt', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
        'debt_settlement_flag', 'out_prncp', 'out_prncp_inv',
        
        # --- 3. 내부 평가 등급 (예측 목적에 부합하지 않아 제거) ---
        'grade', 'sub_grade',
        
        # --- 4. 중복 지표 및 다중공선성 우려 변수 ---
        'funded_amnt', 'funded_amnt_inv', 'fico_range_high', 
        'annual_inc', 'annual_inc_joint', 'dti_joint', 'monthly_inc',
        
        # --- 5. 노이즈 및 희소(Sparse) 변수 (공동 대출자 세부 정보) ---
        'sec_app_fico_range_low', 'sec_app_fico_range_high',
        'sec_app_inq_last_6mths', 'sec_app_mort_acc',
        'sec_app_open_acc', 'sec_app_revol_util',
        'sec_app_open_act_il', 'sec_app_num_rev_accts',
        'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med',
        'revol_bal_joint',
        
        # --- 6. 단일값(Zero-variance) 혹은 분산이 매우 낮은 변수 ---
        'acc_now_delinq', 'delinq_amnt', 'chargeoff_within_12_mths',
        
        # --- 7. 대출 부도 사후 처리(Hardship/Settlement) 관련 변수 ---
        'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
        'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
        'hardship_loan_status', 'orig_projected_additional_accrued_interest',
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
        'settlement_status', 'settlement_date'
    ]
    
    # 금리(Interest Rate) 기반 모델 의존성 통제 스위치 적용
    if config.REMOVE_INT_RATE_FROM_TRAIN:
        drop_cols.append('int_rate')
        
    return list(set(drop_cols))

# =============================================================================
# 2. 전처리 파이프라인 (Polars LazyFrame 활용)
# =============================================================================
def process_pipeline(file_path, is_train=True):
    print(f"🚀 [Pipeline Start] 데이터 로드 및 전처리 초기화: {file_path}")
    
    q = pl.scan_csv(file_path, infer_schema_length=10000, ignore_errors=True)
    
    # --------------------------------------------------------------------------
    # 1. Target Labeling
    # --------------------------------------------------------------------------
    if is_train:
        target_bad = ['Charged Off']
        target_good = ['Fully Paid']
        
        q = (
            q.filter(pl.col('loan_status').is_in(target_bad + target_good))
             .with_columns(
                 pl.col('loan_status').is_in(target_bad).cast(pl.Int8).alias('target')
             )
        )
    
    # --------------------------------------------------------------------------
    # 2. String Parsing & Type Casting
    # --------------------------------------------------------------------------
    q = q.with_columns([
        pl.col('issue_d').str.strptime(pl.Date, '%b-%Y', strict=False).alias('issue_d_parsed'),
        pl.col('earliest_cr_line').str.strptime(pl.Date, '%b-%Y', strict=False).alias('earliest_cr_line_parsed'),
        pl.col('term').str.strip_chars(' months').cast(pl.Int32, strict=False),
        pl.col('int_rate').str.strip_chars(' %').cast(pl.Float32, strict=False),
        pl.col('revol_util').str.strip_chars(' %').cast(pl.Float32, strict=False),
        
        # 근속 연수(Employment Length) 수치 정규화
        pl.col('emp_length')
          .str.replace('< 1 year', '0')
          .str.replace('10\+ years', '10')
          .str.extract(r'(\d+)', 1)
          .cast(pl.Int32, strict=False)
          .fill_null(0)
          .alias('emp_length_int')
    ])
    
    # --------------------------------------------------------------------------
    # 3. Feature Engineering
    # --------------------------------------------------------------------------
    q = q.with_columns([
        # (1) 공동 차입자 정보를 고려한 유효 소득 및 DTI 통합
        pl.coalesce([pl.col('annual_inc_joint'), pl.col('annual_inc')]).cast(pl.Float32).alias('effective_annual_inc'),
        pl.coalesce([pl.col('dti_joint'), pl.col('dti')]).cast(pl.Float32).alias('effective_dti'),
        
        # (2) 월 소득 추정치
        (pl.col('annual_inc') / 12).alias('monthly_inc')
    ])
    
    q = q.with_columns([
        # (3) 신용 이력 기간(년 단위 변환)
        ((pl.col('issue_d_parsed') - pl.col('earliest_cr_line_parsed')).dt.total_days() / 365.25).cast(pl.Float32).alias('credit_hist_years'),
        
        # (4) 소득 대비 상환액 비율
        (pl.col('installment') / (pl.col('monthly_inc') + 1)).alias('installment_ratio'),

        # (5) 연 소득 대비 대출 원금 비중
        (pl.col('loan_amnt') / (pl.col('annual_inc') + 1)).alias('lti_ratio')
    ])

    # --------------------------------------------------------------------------
    # 4. 역선택(Adverse Selection) 방지용 이진 플래그(Binary Flags)
    # --------------------------------------------------------------------------
    q = q.with_columns([
        (pl.col('application_type') == 'Joint App').cast(pl.Int8).alias('is_joint_app'),
        (pl.col('tax_liens') > 0).cast(pl.Int8).alias('has_tax_liens'),
        (pl.col('pub_rec') > 0).cast(pl.Int8).alias('has_pub_rec'),
    ])

    # --------------------------------------------------------------------------
    # 5. Missing Value Imputation (결측치를 고유 정보로 보존)
    # --------------------------------------------------------------------------
    schema = q.collect_schema().names()
    mths_cols = [c for c in schema if 'mths_since' in c]
    
    for col_name in mths_cols:
         q = q.with_columns([
             pl.col(col_name).is_null().cast(pl.Int8).alias(f'is_never_{col_name}'), 
             # 트리 모델의 학습을 유도하기 위한 극단값(Outlier) 대체
             pl.col(col_name).fill_null(9999).cast(pl.Float32) 
         ])

    # --------------------------------------------------------------------------
    # 6. Feature Selection (불필요 변수 소거)
    # --------------------------------------------------------------------------
    drop_candidates = get_cols_to_drop()
    drop_candidates.extend(['application_type', 'tax_liens', 'pub_rec'])
    
    # 성과 산출용 핵심 변수(FINANCE_COLS) 보호 논리
    real_cols_to_drop = [c for c in drop_candidates if c not in FINANCE_COLS]
    
    current_cols = q.collect_schema().names()
    final_drop_list = [c for c in real_cols_to_drop if c in current_cols]
    
    q = q.drop(final_drop_list)
    
    return q

# =============================================================================
# 3. Execution Module
# =============================================================================
def main():
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    q = process_pipeline(config.RAW_DATA_PATH, is_train=True)
    
    print("🔄 [Step 2] 데이터 스트리밍 처리 및 메모리 적재...")
    df = q.collect(engine='streaming')
    
    # [수정됨] Polars 단계에서의 Categorical 변환 생략 (Pandas 호환성 에러 원천 차단)
    
    save_path = config.DATA_PATH
    print(f"💾 [Step 3] 압축 저장 중 (ZSTD Compression) -> {save_path}")
    df.write_parquet(save_path, compression='zstd')
    
    print(f"✅ 전처리 파이프라인 완수. Final Matrix Shape: {df.shape}")

if __name__ == "__main__":
    main()