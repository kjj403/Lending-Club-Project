"""
내부수익률(IRR) 단독 연산 및 캐시(Cache) 생성 모듈
- 데이터 파이프라인과 독립적으로 IRR 통계량을 확인하고 사전 연산하기 위한 스크립트
"""

import os
import data_loader
import config
import numpy as np
import pandas as pd

def main():
    print("⏳ [IRR Module] 내부수익률(IRR) 연산 및 캐시 갱신 시작...")
    df = data_loader.prepare_data_with_weights() 
    cache_path = config.IRR_CACHE_PATH            

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    key_col = "id" if "id" in df.columns else None
    
    if key_col:
        df[[key_col, "actual_irr"]].dropna().drop_duplicates(subset=[key_col]).to_parquet(cache_path, index=False)
    else:
        df[["actual_irr"]].dropna().to_parquet(cache_path, index=False)

    # IRR 산출 결과 통계량 확인
    irr_series = pd.to_numeric(df["actual_irr"], errors="coerce")
    valid_irr = irr_series[np.isfinite(irr_series)]

    print("\n✅ [IRR Cache Saved]")
    print(f" - Path   : {cache_path}")
    print(f" - Exists : {os.path.exists(cache_path)}")
    print(f" - Count  : {len(valid_irr):,}")
    print(f" - Mean   : {valid_irr.mean():.6f}")
    print(f" - Std    : {valid_irr.std(ddof=1):.6f}")
    print(f" - Min    : {valid_irr.min():.6f}")
    print(f" - Max    : {valid_irr.max():.6f}")
    print("\n📊 [Percentiles]")
    print(valid_irr.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))

if __name__ == "__main__":
    main()