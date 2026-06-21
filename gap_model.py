# -*- coding: utf-8 -*-
"""
gap_model.py
============
야간 신호 -> 익일 한국 개장 갭 예측의 공용 로직.
CLI(overnight_gap_predict.py)와 대시보드(app.py)가 함께 사용한다.
"""
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

START = "2023-01-01"
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
    beta_full = ols(X, y)
    resid = y - (np.column_stack([np.ones(n), X]) @ beta_full)
    resid_std = np.std(resid)
    return beta_full, r2_out, hit, resid_std, n


def predict_all():
    """전체 예측 수행. 반환: (latest_signal, latest_date, results[list of dict])"""
    us = get_us_returns()
    if us.empty:
        raise RuntimeError("미국 신호 데이터 수집 실패")

    us_clean = us.sort_index().dropna()
    latest = us_clean.iloc[-1]
    latest_date = us_clean.index[-1].date()

    results = []
    for name, code in KR.items():
        kr = get_kr(code)
        last_close = kr["Close"].dropna().iloc[-1]
        last_close_date = kr["Close"].dropna().index[-1].date()

        data = build_dataset(kr, us)
        beta, r2_out, hit, resid_std, n = train_and_score(data)

        x = latest[FEATURES].values.astype(float)
        pred_gap = float(beta[0] + beta[1:] @ x)
        pred_open = last_close * (1 + pred_gap / 100.0)
        lo = last_close * (1 + (pred_gap - resid_std) / 100.0)
        hi = last_close * (1 + (pred_gap + resid_std) / 100.0)

        # 최근 30거래일 실제 갭 vs 모델 적합치 (차트용)
        recent = data.tail(30).copy()
        Xr = recent[FEATURES].values
        recent["fitted"] = np.column_stack([np.ones(len(Xr)), Xr]) @ beta

        results.append({
            "name": name, "code": code,
            "last_close": float(last_close), "last_close_date": last_close_date,
            "pred_gap": pred_gap, "pred_open": float(pred_open),
            "lo": float(lo), "hi": float(hi),
            "r2_out": float(r2_out), "hit": float(hit), "n": int(n),
            "recent": recent[["gap", "fitted"]],
        })

    return latest, latest_date, results
