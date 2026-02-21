import os
from datetime import datetime

# =========================================================
# 1. 경로 설정 (Path Settings)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1) 원본 데이터 (전처리 파이프라인 입력용)
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/lending_club_2020_train.csv')

# 2) 전처리 완료 데이터 (학습용 최종 파생변수 포함)
DATA_PATH = os.path.join(BASE_DIR, '../data/processed/train_final.parquet')

# 3) 외부 거시경제 지표 데이터 (무위험 이자율 매핑용 국채 금리)
GS3_PATH = os.path.join(BASE_DIR, '../data/external/GS3.csv')
GS5_PATH = os.path.join(BASE_DIR, '../data/external/GS5.csv')

# 내부수익률(IRR) 고속 연산을 위한 캐시 경로
IRR_CACHE_PATH = os.path.join(BASE_DIR, "../data/irr_cache.parquet")

# 4) 결과물 저장 경로 (모델 아티팩트 및 분석 리포트)
NOW_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, f'../reports/figures/run_{NOW_STR}')
MODEL_DIR = os.path.join(BASE_DIR, '../models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 2. 전역 실험 설정 (Global Experiment Switches)
# =========================================================
# 타겟 누수(Data Leakage) 방지를 위한 금리(int_rate) 변수 학습 제외 여부
REMOVE_INT_RATE_FROM_TRAIN = True 

# 실험 재현성(Reproducibility)을 위한 난수 시드 고정
SEED = 42

# (참고) 학습 변수 삭제 및 선택(Feature Selection) 로직은
# 모델의 내장 기능 활용을 위해 'src/preprocess_pipeline.py' 내부로 모듈화됨.