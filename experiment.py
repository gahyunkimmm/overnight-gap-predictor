# -*- coding: utf-8 -*-
"""
실험: 신호 확장 + 릿지 회귀가 실제로 OOS 성능을 올리는지 검증.
baseline(현재 4신호, OLS) vs expanded(신호 추가) vs expanded+ridge 비교.
"""
import sys, io
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

START = "2022-01-01"
STOCKS = {"삼성전자": "005930", "SK하이닉스": "000660", "현대차": "005380"}

BASE = {"MU": "MU", "SOX": "^SOX", "EWY": "EWY", "USDKRW": "KRW=X"}
EXTRA = {"NVDA": "NVDA", "TSM": "TSM", "NASDAQ": "^IXIC", "VIX": "^VIX", "SP500": "^GSPC"}


def get_kr(code):
    try:
        df = fdr.DataReader(code, START)
        out = df[["Open", "Close"]].copy()
        if len(out.dropna()) >= 50:
            out.index = pd.to_datetime(out.index).normalize()
            return out
    except Exception:
        pass
    d = yf.download(f"{code}.KS", start=START, progress=False, auto_adjust=True)
    out = d[["Open", "Close"]].copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.index = pd.to_datetime(out.index).normalize()
    return out


def get_us(tickers):
    cols = {}
    for name, tk in tickers.items():
        d = yf.download(tk, start=START, progress=False, auto_adjust=True)
        if d is None or len(d) == 0:
            continue
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        r = c.pct_change() * 100.0
        r.index = pd.to_datetime(r.index).normalize()
        cols[name] = r
    return pd.DataFrame(cols)


def build(kr, us, feats):
    kr = kr.copy()
    kr["prev_close"] = kr["Close"].shift(1)
    kr["gap"] = (kr["Open"] / kr["prev_close"] - 1) * 100.0
    us = us[feats].sort_index()
    rows = []
    for d, gap in kr["gap"].items():
        if pd.isna(gap):
            continue
        prev = us.loc[us.index < d]
        if len(prev) == 0:
            continue
        sig = prev.iloc[-1]
        if sig.isna().any():
            continue
        rows.append((d, gap, *sig.values))
    df = pd.DataFrame(rows, columns=["date", "gap"] + feats).set_index("date").dropna()
    return df


def fit_predict(Xtr, ytr, Xte, ridge=0.0):
    # 표준화
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    Xb = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
    p = Xb.shape[1]
    A = Xb.T @ Xb + ridge * np.eye(p)
    A[0, 0] -= ridge  # 절편은 정규화 제외
    beta = np.linalg.solve(A, Xb.T @ ytr)
    yhat = np.column_stack([np.ones(len(Xte_s)), Xte_s]) @ beta
    return yhat


def evaluate(df, feats, ridge=0.0):
    X = df[feats].values
    y = df["gap"].values
    n = len(df)
    cut = int(n * 0.7)
    yhat = fit_predict(X[:cut], y[:cut], X[cut:], ridge)
    yte = y[cut:]
    ss_res = np.sum((yte - yhat) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    hit = np.mean(np.sign(yhat) == np.sign(yte)) * 100
    mae = np.mean(np.abs(yhat - yte))
    return r2, hit, mae, n


def main():
    base_feats = list(BASE.keys())
    all_feats = list(BASE.keys()) + list(EXTRA.keys())
    print("미국 신호 수집...")
    us = get_us({**BASE, **EXTRA})
    print(f"  수집된 신호: {list(us.columns)}")

    for name, code in STOCKS.items():
        kr = get_kr(code)
        print(f"\n{'='*60}\n  {name} ({code})\n{'='*60}")
        configs = [
            ("baseline (4신호, OLS)", base_feats, 0.0),
            ("expanded (9신호, OLS)", all_feats, 0.0),
            ("expanded (9신호, Ridge λ=10)", all_feats, 10.0),
            ("expanded (9신호, Ridge λ=30)", all_feats, 30.0),
        ]
        print(f"  {'설정':32s} {'OOS R²':>8s} {'적중%':>7s} {'MAE':>7s}")
        for label, feats, ridge in configs:
            df = build(kr, us, feats)
            r2, hit, mae, n = evaluate(df, feats, ridge)
            print(f"  {label:32s} {r2:8.3f} {hit:6.1f}% {mae:6.2f}  (n={n})")


if __name__ == "__main__":
    main()
