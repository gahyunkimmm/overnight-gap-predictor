# -*- coding: utf-8 -*-
"""
야간 해외 신호 -> 익일 한국 시가 갭 예측력 검증
대상: 삼성전자(005930), SK하이닉스(000660)
신호: 마이크론(MU), 필라델피아반도체(^SOX), 한국ETF(EWY), 원/달러(KRW=X)

핵심 아이디어:
  미국 장(D-1 밤)이 한국 장(D일 아침) 이전에 끝난다.
  -> D-1 미국 종가 등락률로 D일 한국 '개장 갭'을 설명할 수 있는가?

지표:
  - 상관계수 (각 신호 vs 익일 갭)
  - 다중회귀 R^2 (in-sample, out-of-sample)
  - 방향 적중률 (갭 부호 예측)
"""
import sys
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

START = "2023-01-01"

KR = {"삼성전자": "005930", "SK하이닉스": "000660"}
US_TICKERS = {"MU": "MU", "SOX": "^SOX", "EWY": "EWY", "USDKRW": "KRW=X"}


def get_kr(code):
    df = fdr.DataReader(code, START)
    # 시가/종가
    out = df[["Open", "Close"]].copy()
    out.index = pd.to_datetime(out.index).normalize()
    return out


def get_us_returns():
    """미국 신호들의 일별 종가 등락률(%) 반환. 인덱스는 거래일(미국)."""
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
    us = pd.DataFrame(cols)
    return us


def ols(X, y):
    """절편 포함 최소제곱. 반환: beta, y_hat, r2"""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    y_hat = Xb @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, y_hat, r2


def analyze(name, code, us):
    print("\n" + "=" * 60)
    print(f"  {name} ({code})")
    print("=" * 60)

    kr = get_kr(code)
    # 익일 시가 갭(%) = (당일 시가 / 전일 종가 - 1) * 100
    kr["prev_close"] = kr["Close"].shift(1)
    kr["gap"] = (kr["Open"] / kr["prev_close"] - 1) * 100.0

    # 한국 D일 갭을 설명하는 신호는 '미국 D-1일 밤' 등락률.
    # 미국 거래일 t의 종가 등락은 한국의 '다음 거래일' 아침에 반영된다.
    # 날짜 단순 정렬: 한국 거래일 d에 대해, d 직전(<d)의 가장 최근 미국 등락을 매칭.
    us_sorted = us.sort_index()

    rows = []
    for d, gap in kr["gap"].items():
        if pd.isna(gap):
            continue
        prev_us = us_sorted.loc[us_sorted.index < d]
        if len(prev_us) == 0:
            continue
        sig = prev_us.iloc[-1]
        if sig.isna().any():
            continue
        rows.append((d, gap, *sig.values))

    cols = ["date", "gap"] + list(us.columns)
    data = pd.DataFrame(rows, columns=cols).set_index("date").dropna()
    n = len(data)
    print(f"  표본 수: {n}일")
    if n < 50:
        print("  표본 부족 -> 분석 중단")
        return

    # 1) 개별 상관계수
    print("\n  [단변량 상관계수: 신호(D-1 미국 등락) vs 익일 갭]")
    for c in us.columns:
        r = np.corrcoef(data[c], data["gap"])[0, 1]
        print(f"    {c:8s}: {r:+.3f}")

    # 2) 다중회귀 (in-sample) + train/test 분할
    feat = list(us.columns)
    X = data[feat].values
    y = data["gap"].values

    _, _, r2_in = ols(X, y)
    print(f"\n  [다중회귀 in-sample R^2] {r2_in:.3f}")

    # 시계열 분할: 앞 70% 학습, 뒤 30% 검증
    cut = int(n * 0.7)
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]
    beta, _, _ = ols(Xtr, ytr)
    yhat_te = np.column_stack([np.ones(len(Xte)), Xte]) @ beta
    ss_res = np.sum((yte - yhat_te) ** 2)
    ss_tot = np.sum((yte - np.mean(yte)) ** 2)
    r2_out = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  [out-of-sample R^2]     {r2_out:.3f}   (뒤 {len(yte)}일)")

    # 3) 방향 적중률 (부호)
    hit = np.mean(np.sign(yhat_te) == np.sign(yte)) * 100
    base = max(np.mean(yte > 0), np.mean(yte < 0)) * 100  # 항상 한쪽만 찍을 때
    print(f"  [방향 적중률]           {hit:.1f}%   (기준선 {base:.1f}%)")

    # 4) 평균 갭 크기 참고
    print(f"  [참고] 평균 |갭| = {np.mean(np.abs(y)):.2f}%, 갭 표준편차 = {np.std(y):.2f}%")


def main():
    print("미국 야간 신호 다운로드 중...")
    us = get_us_returns()
    if us.empty:
        print("미국 데이터 수집 실패")
        sys.exit(1)
    print(f"  신호 {list(us.columns)} / 기간 {us.index.min().date()} ~ {us.index.max().date()}")

    for name, code in KR.items():
        analyze(name, code, us)


if __name__ == "__main__":
    main()
