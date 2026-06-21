# -*- coding: utf-8 -*-
"""
overnight_gap_predict.py
========================
어젯밤 미국 시장(반도체) 신호로 '오늘 한국 개장 갭'을 추정한다.

대상: 삼성전자(005930), SK하이닉스(000660)
신호: 마이크론(MU), 필라델피아반도체(^SOX), 한국ETF(EWY), 원/달러(KRW=X)

매일 아침(한국 장 시작 전) 한 번 실행:
    python overnight_gap_predict.py

출력:
  - 오늘 예상 개장 갭(%)과 방향
  - 직전 종가 기준 예상 개장가(원)
  - 모델 신뢰도(검증 R^2, 방향 적중률)
  - 예측 로그(predictions_log.csv)에 자동 누적 -> 사후 정확도 추적

주의: 이 갭은 '개장가 그 자체'에 반영되어 열린다. 갭 방향을 가늠하는
      참고용이지 자동 수익 도구가 아니다.
"""
import sys
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

# Windows 콘솔 한글 깨짐 방지
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

START = "2023-01-01"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")

KR = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"MU": "MU", "SOX": "^SOX", "EWY": "EWY", "USDKRW": "KRW=X"}
FEATURES = list(US_TICKERS.keys())


# ----------------------------- 데이터 -----------------------------
def get_kr(code):
    df = fdr.DataReader(code, START)
    out = df[["Open", "Close"]].copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out


def get_us_returns():
    cols = {}
    for name, tk in US_TICKERS.items():
        d = yf.download(tk, start=START, progress=False, auto_adjust=True)
        if d is None or len(d) == 0:
            print(f"  [경고] {name}({tk}) 데이터 비어있음")
            continue
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ret = close.pct_change() * 100.0
        ret.index = pd.to_datetime(ret.index).normalize()
        cols[name] = ret
    return pd.DataFrame(cols)


# ----------------------------- 모델 -----------------------------
def ols(X, y):
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def r2_of(beta, X, y):
    yhat = np.column_stack([np.ones(len(X)), X]) @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def build_dataset(kr, us):
    """한국 거래일 d의 갭 <- d 직전 미국 등락 매칭."""
    kr = kr.copy()
    kr["prev_close"] = kr["Close"].shift(1)
    kr["gap"] = (kr["Open"] / kr["prev_close"] - 1) * 100.0
    us_sorted = us.sort_index()
    rows = []
    for d, gap in kr["gap"].items():
        if pd.isna(gap):
            continue
        prev = us_sorted.loc[us_sorted.index < d]
        if len(prev) == 0:
            continue
        sig = prev.iloc[-1]
        if sig.isna().any():
            continue
        rows.append((d, gap, *sig.values))
    cols = ["date", "gap"] + FEATURES
    return pd.DataFrame(rows, columns=cols).set_index("date").dropna()


def train_and_score(data):
    X = data[FEATURES].values
    y = data["gap"].values
    n = len(data)
    cut = int(n * 0.7)
    beta_tr = ols(X[:cut], y[:cut])
    r2_out = r2_of(beta_tr, X[cut:], y[cut:])
    yhat_te = np.column_stack([np.ones(n - cut), X[cut:]]) @ beta_tr
    hit = np.mean(np.sign(yhat_te) == np.sign(y[cut:])) * 100
    # 최종 예측엔 전체 데이터로 재학습
    beta_full = ols(X, y)
    resid = y - (np.column_stack([np.ones(n), X]) @ beta_full)
    resid_std = np.std(resid)
    return beta_full, r2_out, hit, resid_std, n


# ----------------------------- 실행 -----------------------------
def main():
    print("=" * 64)
    print(f"  야간 신호 기반 개장 갭 예측  |  실행시각 {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * 64)

    print("\n미국 야간 신호 수집 중...")
    us = get_us_returns()
    if us.empty:
        print("미국 데이터 수집 실패")
        sys.exit(1)

    # 예측에 쓸 '가장 최근 미국 등락' (어젯밤 신호)
    latest = us.sort_index().dropna().iloc[-1]
    latest_date = us.sort_index().dropna().index[-1].date()
    print(f"  최신 신호일(미국): {latest_date}")
    print("  " + "  ".join(f"{c} {latest[c]:+.2f}%" for c in FEATURES))

    log_rows = []
    for name, code in KR.items():
        kr = get_kr(code)
        last_close = kr["Close"].dropna().iloc[-1]
        last_close_date = kr["Close"].dropna().index[-1].date()

        data = build_dataset(kr, us)
        beta, r2_out, hit, resid_std, n = train_and_score(data)

        x = latest[FEATURES].values.astype(float)
        pred_gap = beta[0] + beta[1:] @ x
        pred_open = last_close * (1 + pred_gap / 100.0)
        lo = last_close * (1 + (pred_gap - resid_std) / 100.0)
        hi = last_close * (1 + (pred_gap + resid_std) / 100.0)
        direction = "상승" if pred_gap > 0 else "하락"

        print("\n" + "-" * 64)
        print(f"  [{name}] ({code})")
        print(f"    직전 종가({last_close_date}) : {last_close:,.0f}원")
        print(f"    예상 개장 갭            : {pred_gap:+.2f}%  ({direction})")
        print(f"    예상 개장가            : {pred_open:,.0f}원")
        print(f"    ±1σ 범위               : {lo:,.0f} ~ {hi:,.0f}원")
        print(f"    모델 신뢰도            : OOS R²={r2_out:.2f}, 방향적중={hit:.0f}% (표본 {n}일)")

        log_rows.append({
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "us_signal_date": latest_date,
            "stock": name, "code": code,
            "last_close": round(last_close, 0),
            "pred_gap_pct": round(pred_gap, 3),
            "pred_open": round(pred_open, 0),
            "oos_r2": round(r2_out, 3),
            "hit_rate": round(hit, 1),
            **{f"sig_{c}": round(float(latest[c]), 3) for c in FEATURES},
        })

    # 로그 누적
    df_log = pd.DataFrame(log_rows)
    if os.path.exists(LOG_PATH):
        df_log.to_csv(LOG_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 64)
    print(f"  예측 로그 저장: {LOG_PATH}")
    print("  ※ 갭은 개장가에 이미 반영되어 열립니다. 방향 가늠용 참고 지표입니다.")
    print("=" * 64)


if __name__ == "__main__":
    main()
