# -*- coding: utf-8 -*-
"""
overnight_gap_predict.py
========================
어젯밤 미국 시장(반도체) 신호로 '오늘 한국 개장 갭'을 추정한다 (CLI 버전).

매일 아침(한국 장 시작 전) 한 번 실행:
    python overnight_gap_predict.py

웹 대시보드로 보려면:
    streamlit run app.py

출력:
  - 오늘 예상 개장 갭(%)과 방향, 직전 종가 기준 예상 개장가(원)
  - 모델 신뢰도(검증 R^2, 방향 적중률)
  - 예측 로그(predictions_log.csv) 자동 누적 -> 사후 정확도 추적

주의: 이 갭은 '개장가 그 자체'에 반영되어 열린다. 갭 방향을 가늠하는
      참고용이지 자동 수익 도구가 아니다.
"""
import sys
import io
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

import gap_model as gm

KST = timezone(timedelta(hours=9))  # 한국 표준시 (DST 없음)

# Windows 콘솔 한글 깨짐 방지
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")


def main():
    print("=" * 64)
    print(f"  야간 신호 기반 개장 갭 예측  |  실행시각 {datetime.now(KST):%Y-%m-%d %H:%M} KST")
    print("=" * 64)

    print("\n미국 야간 신호 수집 중...")
    try:
        latest, latest_date, feats, results = gm.predict_all()
    except Exception as e:
        print(f"데이터 수집 실패: {e}")
        sys.exit(1)

    print(f"  최신 신호일(미국): {latest_date}  (가용 신호 {len(feats)}/{len(gm.FEATURES)})")
    print("  " + "  ".join(f"{c} {latest[c]:+.2f}%" for c in feats))

    log_rows = []
    for r in results:
        direction = "상승" if r["pred_gap"] > 0 else "하락"
        print("\n" + "-" * 64)
        print(f"  [{r['name']}] ({r['code']})")
        print(f"    직전 종가({r['last_close_date']}) : {r['last_close']:,.0f}원")
        print(f"    예상 개장 갭            : {r['pred_gap']:+.2f}%  ({direction})")
        print(f"    예상 개장가            : {r['pred_open']:,.0f}원")
        print(f"    ±1σ 범위               : {r['lo']:,.0f} ~ {r['hi']:,.0f}원")
        print(f"    모델 신뢰도(워크포워드) : R²={r['r2_wf']:.2f}, 방향적중={r['hit_wf']:.0f}% (검증 {r['n']}일)")
        print(f"    참고-장중(개장→종가)    : 방향적중={r['intraday_hit']:.0f}%, R²={r['intraday_r2']:.2f}")

        log_rows.append({
            "run_time": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
            "us_signal_date": latest_date,
            "stock": r["name"], "code": r["code"],
            "last_close_date": r["last_close_date"],
            "last_close": round(r["last_close"], 0),
            "pred_gap_pct": round(r["pred_gap"], 3),
            "pred_open": round(r["pred_open"], 0),
            "wf_r2": round(r["r2_wf"], 3),
            "wf_hit": round(r["hit_wf"], 1),
            **{f"sig_{c}": round(float(latest[c]), 3) for c in feats},
        })

    df_log = pd.DataFrame(log_rows)
    if os.path.exists(LOG_PATH):
        try:
            old = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
        except Exception:
            old = pd.DataFrame()
        # 스키마가 같으면 단순 append, 다르면 정렬 후 전체 재기록(마이그레이션)
        if list(old.columns) == list(df_log.columns):
            df_log.to_csv(LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            pd.concat([old, df_log], ignore_index=True).to_csv(
                LOG_PATH, index=False, encoding="utf-8-sig")
    else:
        df_log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 64)
    print(f"  예측 로그 저장: {LOG_PATH}")
    print("  ※ 갭은 개장가에 이미 반영되어 열립니다. 방향 가늠용 참고 지표입니다.")
    print("=" * 64)


if __name__ == "__main__":
    main()
